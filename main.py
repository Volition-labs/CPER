from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import faiss
import pickle

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ---------- Config ----------
BI_ENCODER_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
INDEX_PATH = "model/revised_faiss_index_v01.index"
MAPPING_PATH = "model/revised_json_mapping_v01.pkl"
A_KEYS = ["A1", "A2", "A3", "A1_A3_total", "A4", "A5"]

# ---------- Load Models and Data ----------
bi_encoder = SentenceTransformer(BI_ENCODER_MODEL_NAME)
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME)

faiss_index = faiss.read_index(INDEX_PATH)
with open(MAPPING_PATH, "rb") as f:
    processed_data = pickle.load(f)


# ---------- Utilities ----------
def prepare_text(item):
    product_names = " ".join(item.get("product_names", []))
    product_desc = " ".join(item.get("product_description", []))[:500]
    product_ids = " ".join(item.get("product_ids", [])) if isinstance(item.get("product_ids"), list) else ''
    combined = (product_names + " ") * 3 + (product_ids + " ") * 2 + product_desc
    return combined.strip()


def get_embedding(text):
    return bi_encoder.encode(text, normalize_embeddings=True).astype("float32")


def search_with_rerank(query: str, top_k=15, rerank_k=5, min_score=0.85):
    query_embedding = get_embedding(query).reshape(1, -1)
    D, I = faiss_index.search(query_embedding, top_k)

    candidates = []
    for idx in I[0]:
        item = processed_data[idx]
        candidates.append((item, prepare_text(item)))

    # Cross-encoder re-ranking
    cross_inputs = [(query, text) for _, text in candidates]
    cross_scores = cross_encoder.predict(cross_inputs)

    reranked = sorted(zip(candidates, cross_scores), key=lambda x: x[1], reverse=True)

    results = []
    for ((item, _), score) in reranked[:rerank_k]:
        if score >= min_score:
            results.append((item, float(score)))
    return results


def find_exact_match(query: str):
    for product in processed_data:
        ids = [i for i in product.get("product_ids", []) if isinstance(i, str)]
        names = [n for n in product.get("product_names", []) if isinstance(n, str)]
        descs = [d for d in product.get("product_description", []) if isinstance(d, str)]
        if query.lower() in map(str.lower, ids + names + descs):
            return product
    return None


def get_impact_info(product, impact_filter='global warming'):
    for impact in product.get("epd_impacts", []):
        if impact_filter.lower() in impact.get("impact_category", "").lower():
            return impact
    return None


def fill_missing_a_values(a_values):
    for key in A_KEYS:
        a_values.setdefault(key, None)
    return a_values


def compute_weighted_average(impact_data_list, similarities):
    total_sim = sum(similarities)
    if total_sim == 0:
        return {}, None
    norm_weights = [s / total_sim for s in similarities]
    weighted = {}
    unit = None
    for impact, w in zip(impact_data_list, norm_weights):
        if not impact:
            continue
        if unit is None:
            unit = impact.get("unit")
        for key in A_KEYS:
            try:
                val = float(impact.get(key, 0))
                weighted[key] = weighted.get(key, 0) + val * w
            except (ValueError, TypeError):
                continue
    return fill_missing_a_values(weighted), unit


def extract_product_info(product):
    info = {
        "product_names": product.get("product_names", []),
        "product_description": product.get("product_description", []),
        "product_ids": product.get("product_ids", [])
    }
    impact = get_impact_info(product)
    if impact:
        info["A_values"] = fill_missing_a_values({k: impact.get(k, None) for k in A_KEYS})
    else:
        info["A_values"] = {k: None for k in A_KEYS}
    return info


# ---------- API ----------
class QueryRequest(BaseModel):
    query: str


@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/search")
def search_endpoint(request: QueryRequest):
    query = request.query.strip()
    if not query:
        return {"message": "Please enter a query", "status": 400}

    # First: exact match
    exact_product = find_exact_match(query)
    if exact_product:
        impact = get_impact_info(exact_product)
        a_values = {k: impact.get(k, None) for k in A_KEYS} if impact else {k: None for k in A_KEYS}
        return {
            "message": "Exact match found.",
            "score_type": "exact",
            "similarity_scores": [1.0],
            "impact": {"unit": impact.get("unit") if impact else None, "A_values": fill_missing_a_values(a_values)},
            "matched_products": [{"product_info": extract_product_info(exact_product), "similarity": 1.0}],
            "status": 200
        }

    # Otherwise: hybrid search
    results = search_with_rerank(query)
    if not results:
        return {"message": "No relevant results found.", "status": 404}

    matched_products = [{"product_info": extract_product_info(p), "similarity": sim} for p, sim in results]
    similarities = [sim for _, sim in results]

    top_similarity = similarities[0]

    if top_similarity == 1.0:
        impact = get_impact_info(results[0][0])
        a_values = {k: impact.get(k, None) for k in A_KEYS} if impact else {k: None for k in A_KEYS}
        return {
            "message": "Perfect match found.",
            "score_type": "top_match",
            "similarity_scores": [top_similarity],
            "impact": {"unit": impact.get("unit") if impact else None, "A_values": fill_missing_a_values(a_values)},
            "matched_products": matched_products,
            "status": 200
        }

    elif top_similarity >= 0.9:
        impact_list = [get_impact_info(p) for p, _ in results]
        weighted_avg, unit = compute_weighted_average(impact_list, similarities)
        return {
            "message": "High similarity, using weighted average.",
            "score_type": "weighted_average",
            "similarity_scores": similarities,
            "weighted_similarity_score": np.mean(similarities),
            "impact": {"unit": unit, "A_values": weighted_avg},
            "matched_products": matched_products,
            "status": 200
        }

    else:
        return {
            "message": "No sufficiently relevant match found.",
            "score_type": None,
            "similarity_scores": similarities,
            "impact": {"unit": None, "A_values": {k: None for k in A_KEYS}},
            "matched_products": matched_products,
            "status": 400
        }



# Optional: run with `python app.py`
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
