
CPER EPD Semantic Search
=================================

A hybrid semantic search engine for retrieving Environmental Product Declarations (EPDs)
based on Global Warming Potential impact categories (A1–A5). Built using LLMs/transformers,
FAISS, a cross-encoder re-ranker, and deployed with FastAPI.

----------------------
Overview
----------------------

This system lets you:

- Parse and clean EPD JSON files
- Embed product info with a bi-encoder
- Build a FAISS similarity index
- Re-rank top results using a cross-encoder
- Query for environmental impact values using natural language

----------------------
Theoretical Model
----------------------

We combine:

1. Bi-Encoder Search (FAISS): Fast approximate nearest neighbor search
2. Cross-Encoder Re-Ranking: Accurate similarity scoring
3. Threshold Logic:

If similarity == 1.0 => Return top match
If 0.9 <= similarity < 1.0 => Return weighted average
If similarity < 0.9 => No match found

----------------------
Setup
----------------------

1. Install Dependencies
```
pip install -r requirements.txt
```
2. (Optional) Pre-Download Models
```
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
```
3. Build the Index

03_indexing_jsons.py

----------------------
Run the Search Server
----------------------
```
uvicorn main:app --reload --host 0.0.0.0 --port 8100
```
Then visit:
```
http://localhost:8100
```
----------------------
API Usage
----------------------

Endpoint: POST /search
```
Payload:
{
  "query": "Framery Q"
}
```
Returns:
- Top matching products
- Impact values (A1–A5)
- Match score and method

----------------------
A1–A5 Impact Calculation
----------------------

If using weighted average:
```
A_i = sum(sim_k * A_i_k) / sum(sim_k)
```
Where:
- sim_k: similarity of the k-th match
- A_i_k: A1–A5 values per match

----------------------
Example Queries
----------------------

- "Reinforcing steel bars"
- "Framery Q"
- "Aluminium window frames"
- "Pre-mix concrete"

----------------------
Notes
----------------------

- All models are cached locally after first download (~/.cache/huggingface)
- Works offline after model caching
- Uses cosine similarity via normalized embeddings in FAISS

----------------------
License
----------------------

MIT License — free to use and modify with attribution.

----------------------
Acknowledgements
----------------------

Thanks to Hugging Face, Sentence Transformers, and the EPD data providers.
