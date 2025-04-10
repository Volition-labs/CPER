from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import numpy as np
import faiss
import pickle
from transformers import AutoTokenizer, AutoModel

app = FastAPI()

# Set up the Jinja2 templates directory.
templates = Jinja2Templates(directory="templates")

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Define the list of A keys we care about.
A_KEYS = ["A1", "A2", "A3", "A1_A3_total", "A4", "A5"]


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_embedding(text: str):
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    embedding = F.normalize(embedding, p=2, dim=1)
    return embedding


INDEX_PATH = "model/faiss_index.index"  # Adjust path as needed
MAPPING_PATH = "model/json_mapping.pkl"  # Adjust path as needed

faiss_index = faiss.read_index(INDEX_PATH)
with open(MAPPING_PATH, "rb") as f:
    processed_data = pickle.load(f)


class QueryRequest(BaseModel):
    query: str


def find_exact_match(query: str):
    """
    Look for an exact match in the product data.
    Checks if the query exactly matches any string in product_ids,
    product_names, or product_description.
    Non-string entries are skipped.
    """
    for product in processed_data:
        product_ids = product.get("product_ids") or []
        product_names = product.get("product_names") or []
        product_descriptions = product.get("product_description") or []
        # Filter out non-string values
        product_ids = [pid for pid in product_ids if isinstance(pid, str)]
        product_names = [pname for pname in product_names if isinstance(pname, str)]
        product_descriptions = [desc for desc in product_descriptions if isinstance(desc, str)]

        if (any(query.lower() == pid.lower() for pid in product_ids) or
                any(query.lower() == pname.lower() for pname in product_names) or
                any(query.lower() == desc.lower() for desc in product_descriptions)):
            return product
    return None


def search_products(query: str, k: int = 3):
    query_emb = get_embedding(query).cpu().numpy().astype('float32')
    similarities, indices = faiss_index.search(query_emb, k)
    similarities = ((similarities + 1) / 2).flatten()
    similarities = [float(sim) for sim in similarities]

    results = []
    for idx, sim in zip(indices[0], similarities):
        matched_product = processed_data[idx]
        results.append((matched_product, sim))
    return results


def get_impact_info(epd_json: dict, impact_filter: str = 'global warming'):
    impacts = epd_json.get("epd_impacts", [])
    for impact in impacts:
        if impact_filter.lower() in impact.get("impact_category", "").lower():
            return impact
    return None


def compute_weighted_average(impact_data_list, similarities):
    total_sim = sum(similarities)
    if total_sim == 0:
        return {}, None
    norm_weights = [sim / total_sim for sim in similarities]
    weighted_average = {}
    unit = None
    for impact, weight in zip(impact_data_list, norm_weights):
        if impact is None:
            continue
        if unit is None:
            unit = impact.get("unit")
        for key, value in impact.items():
            if key.startswith("A"):
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    continue
                weighted_average[key] = weighted_average.get(key, 0.0) + (value * weight)
    for a_key in A_KEYS:
        if a_key not in weighted_average:
            weighted_average[a_key] = None
    return weighted_average, unit


def fill_missing_a_values(a_values: dict):
    """Ensure A_KEYS are always present in the dictionary."""
    for key in A_KEYS:
        if key not in a_values:
            a_values[key] = None
    return a_values


def extract_product_info(product: dict):
    """
    Extract minimal product info including product names, description, IDs,
    and the impact A values from the product's impact info.
    """
    base_info = {
        "product_names": product.get("product_names", []),
        "product_description": product.get("product_description", []),
        "product_ids": product.get("product_ids", [])
    }
    impact = get_impact_info(product, "global warming")
    if impact:
        a_values = {key: impact.get(key, None) for key in A_KEYS}
    else:
        a_values = {key: None for key in A_KEYS}
    base_info["A_values"] = fill_missing_a_values(a_values)
    return base_info


@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    # Render the Jinja2 template located at templates/index.html
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/search")
def search_endpoint(request: QueryRequest):
    query = request.query

    if not query:
        return {
            "message": "please enter a query",
            "status": 400
        }
    print(f"query: {query}")
    # First, check for an exact match by key lookup.
    exact_product = find_exact_match(query)
    if exact_product is not None:
        product_info = extract_product_info(exact_product)
        impact = get_impact_info(exact_product, "global warming")
        if impact is None:
            impact_response = {"unit": None, "A_values": {key: None for key in A_KEYS}}
        else:
            a_values = {key: impact.get(key, None) for key in A_KEYS}
            a_values = fill_missing_a_values(a_values)
            impact_response = {"unit": impact.get("unit"), "A_values": a_values}
        matched_products = [{"product_info": product_info, "similarity": 1.0}]
        return {
            "message": "Exact match found by key lookup.",
            "score_type": "top_match",
            "similarity_scores": [1.0],
            "impact": impact_response,
            "matched_products": matched_products,
            "status": 200
        }

    # No exact match found, so perform similarity search.
    results = search_products(query, k=3)
    matched_products = [
        {"product_info": extract_product_info(product), "similarity": sim}
        for product, sim in results
    ]
    similarity_scores = [sim for _, sim in results]

    if not results or results[0][1] < 0.70:
        return {
            "message": "Exact match not found.",
            "score_type": None,
            "similarity_scores": similarity_scores,
            "matched_products": matched_products
        }

    top_similarity = results[0][1]
    # Use a threshold of 0.85 for the top match branch.
    if top_similarity >= 0.85:
        impact = get_impact_info(results[0][0], impact_filter="global warming")
        if impact is None:
            impact_response = {"unit": None, "A_values": {key: None for key in A_KEYS}}
        else:
            a_values = {key: impact.get(key, None) for key in A_KEYS}
            a_values = fill_missing_a_values(a_values)
            impact_response = {"unit": impact.get("unit"), "A_values": a_values}
        return {
            "message": "Exact match found based on similarity.",
            "score_type": "top_match",
            "similarity_scores": [top_similarity],
            "impact": impact_response,
            "matched_products": matched_products
        }
    else:
        impact_data_list = []
        weights = []
        for matched_product, sim in results:
            impact = get_impact_info(matched_product, impact_filter="global warming")
            if impact is not None:
                impact_data_list.append(impact)
                weights.append(sim)
        if not impact_data_list:
            return {
                "message": "Impact data not found in top matches.",
                "score_type": "weighted_average",
                "similarity_scores": similarity_scores,
                "matched_products": matched_products,
                "impact": {"unit": None, "A_values": {key: None for key in A_KEYS}}
            }
        weighted_avg, unit = compute_weighted_average(impact_data_list, weights)
        weighted_avg = fill_missing_a_values(weighted_avg)
        return {
            "message": "Exact match not found, using weighted average.",
            "score_type": "weighted_average",
            "similarity_scores": similarity_scores,
            "impact": {
                "unit": unit,
                "A_values": weighted_avg
            },
            "matched_products": matched_products
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
