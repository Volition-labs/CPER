from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
import pickle

# ---------- Load FAISS Index and JSON Mapping ----------
INDEX_PATH = "model/revised_faiss_index_v01.index"
MAPPING_PATH = "model/revised_json_mapping_v01.pkl"
BI_ENCODER_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

faiss_index = faiss.read_index(INDEX_PATH)
with open(MAPPING_PATH, "rb") as f:
    data_mapping = pickle.load(f)

# ---------- Load Models ----------
bi_encoder = SentenceTransformer(BI_ENCODER_MODEL_NAME)
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME)

# ---------- Preprocessing ----------
def prepare_text(item):
    product_names = " ".join(item.get("product_names", []))
    product_desc = " ".join(item.get("product_description", []))[:500]  # Truncate long text
    product_ids = " ".join(item.get("product_ids", [])) if isinstance(item.get("product_ids"), list) else ''
    combined = (product_names + " ") * 3 + (product_ids + " ") * 2 + product_desc
    return combined.strip()

# ---------- Hybrid Search ----------
def search_with_rerank(query, top_k=15, rerank_k=5, min_score=0.85):
    # Step 1: Bi-encoder + FAISS
    query_embedding = bi_encoder.encode(query, normalize_embeddings=True)
    query_embedding = np.array([query_embedding], dtype='float32')
    D, I = faiss_index.search(query_embedding, top_k)

    candidates = []
    for idx in I[0]:
        item = data_mapping[idx]
        text = prepare_text(item)
        candidates.append((item, text))

    # Step 2: Cross-encoder re-ranking
    cross_inputs = [(query, text) for _, text in candidates]
    cross_scores = cross_encoder.predict(cross_inputs)

    reranked = sorted(zip(candidates, cross_scores), key=lambda x: x[1], reverse=True)

    # Step 3: Filter by score threshold
    final_results = [(item, score) for ((item, _), score) in reranked[:rerank_k] if score >= min_score]
    return final_results

# ---------- Run Demo Queries ----------
if __name__ == "__main__":
    test_queries = [
        "Framery Q",
        "Reinforcing steel bars",
        "Aluminium window frames",
        "Pre-mix concrete",
        "Craftsman stretch trousers 2900 GWM",
        "Acoustic insulation board",
        "Plastic boards",
        "prime fix Adhesive mortar for thermal insulation systems"
    ]

    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        results = search_with_rerank(query, top_k=15, rerank_k=5, min_score=0.85)

        if not results:
            print("  ❌ No relevant results found above threshold.")
            continue

        for i, (item, score) in enumerate(results):
            print(f"[{i+1}] Score: {round(score, 4)}", end=" ")
            if i > 0:
                delta = score - results[i - 1][1]
                print(f"(Δ from prev: {round(delta, 4)})")
            else:
                print()
            print("  Product Names :", item.get("product_names"))
            print("  Product IDs   :", item.get("product_ids"))
            desc = item.get("product_description", [""])[0][:250] + "..."
            print("  Description   :", desc)
            print("  " + "-" * 50)
