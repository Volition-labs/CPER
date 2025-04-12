import os
import json
import pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import faiss

# ---------- Load Processed Data ----------
with open('data/revised_processed_json_data_v01.json', 'r', encoding='utf-8') as f:
    processed_data = json.load(f)

# ---------- Create Text Representations ----------
texts = []
for item in tqdm(processed_data, desc="Generating text inputs"):
    product_names = " ".join(item.get("product_names", []))
    product_desc = " ".join(item.get("product_description", []))
    try:
        product_ids = " ".join(item.get("product_ids", []))
    except TypeError:
        product_ids = ''
    combined_text = f"{product_names}. {product_desc}. {product_ids}"
    texts.append(combined_text)

# ---------- Load Sentence Transformer ----------
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embedding(text):
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    return F.normalize(embedding, p=2, dim=1)

# ---------- Generate Embeddings ----------
embeddings_list = []
for text in tqdm(texts, desc="Computing embeddings"):
    emb = get_embedding(text)
    embeddings_list.append(emb.cpu().numpy())

all_embeddings = np.vstack(embeddings_list).astype('float32')

# ---------- Build and Save FAISS Index ----------
embedding_dim = all_embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(embedding_dim)
faiss_index.add(all_embeddings)
print("FAISS index built with", faiss_index.ntotal, "vectors.")

# ---------- Save FAISS Index and Mapping ----------
os.makedirs("model", exist_ok=True)
index_filename = "model/revised_faiss_index_v01.index"
mapping_filename = "model/revised_json_mapping_v01.pkl"

faiss.write_index(faiss_index, index_filename)
print(f"FAISS index saved to {index_filename}")

with open(mapping_filename, "wb") as f:
    pickle.dump(processed_data, f)
print(f"JSON mapping saved to {mapping_filename}")

