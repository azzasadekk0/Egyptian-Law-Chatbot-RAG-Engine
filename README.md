# Egyptian Law Chatbot — RAG Engine

A **Retrieval-Augmented Generation (RAG)** system that answers Arabic legal questions about Egyptian law with grounded citations covering **Civil**, **Penal**, **Commercial**, and **Personal Status** law.

---

## Features

- **Domain-Aware Routing:** Automatically classifies queries into 4 legal domains using a custom KNN classifier.
- **Adaptive Hybrid Retrieval:** Combines FAISS (dense) and BM25 (sparse) with Reciprocal Rank Fusion (RRF) for high-precision semantic and keyword matching.
- **Conversational Memory:** Seamlessly handles ambiguous follow-up questions (e.g., "ولو رفض؟") using lightweight heuristic detection and LLM resolution.
- **Strict Grounding:** Strict evidence-based answering with fallback refusal when supporting legal evidence is not found.
- **Cross-Encoder Reranking:** Re-scores the top 15 hybrid results down to the top 3 most relevant chunks using a multilingual cross-encoder.

---


## How It Works

Every query goes through 7 stages:

1. **Memory** — detect ambiguous follow-ups and resolve them using conversation history
2. **Classify** — route the query to a legal domain (civil / penal / commercial / personal_status)
3. **Route** — search that domain's chunks; fall back to full corpus if classifier confidence is low
4. **Rewrite** — expand short/ambiguous phrasing into formal legal terminology (LLM-based query rewriting)
5. **Retrieve** — top-15 results from FAISS (semantic) + BM25 (keyword), merged with RRF
6. **Rerank** — multilingual cross-encoder narrows 15 → 3 highest-relevance chunks
7. **Generate** — LLM-based grounded answer generation producing an Arabic answer with citations (temperature 0)

---

## Project Structure

```text
├── RAG_Engine/
│   ├── config.py        ← all settings and paths
│   ├── classifier.py    ← domain classifier (KNN + keyword boost)
│   ├── retriever.py     ← FAISS + BM25 hybrid retrieval with Arabic normalization
│   ├── reranker.py      ← cross-encoder reranker
│   ├── generator.py     ← follow-up resolver + query rewriter + answer generator
│   ├── citations.py     ← citation formatter
│   ├── memory.py        ← conversational memory (per-session, last 10 turns)
│   ├── pipeline.py      ← orchestrates all 7 stages
│   └── run.py           ← CLI entrypoint
│
├── NLP_ML_Pipeline-main/
│   ├── chunks_with_embeddings.jsonl   ← legal chunks + embeddings
│   ├── embeddings_matrix.npy          ← dense embeddings matrix
│   ├── knn_classifier_final.pkl       ← trained domain classifier
│   └── chunks_cleaned.jsonl           ← text + metadata only
│
├── tests/
│   ├── healthcheck.py   ← fast component check, no Groq API call required
│   ├── evaluation.py    ← 25-question evaluation (Recall@K, MRR, Hit Rate)
│   └── system.py        ← full system test covering all pipeline stages
└── requirements.txt
```

---

## Setup

**Requirements:** Python 3.11+, a Groq API key with access to llama-3.3-70b-versatile.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
echo GROQ_API_KEY=gsk_...your-key... > .env
```

---

## Usage

```bash
# Ask a single question
python RAG_Engine/run.py --question "ما عقوبة السرقة في مصر؟"

# Interactive mode with conversational memory
python RAG_Engine/run.py --interactive

# Built-in quick benchmark (5 questions)
python RAG_Engine/run.py --benchmark

# Skip query rewriting (faster, lower cost)
python RAG_Engine/run.py --interactive --no-rewrite
```

---

## Running Tests

### Step-by-step from your terminal

```bash
# Make sure your venv is active first
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# Test 1: Smoke test (no API call)
python tests/healthcheck.py
# Validates: embedding model, classifier, retriever, reranker
# No Groq API call required. Local models must be available or already cached.

# Test 2: Full system test (requires API key)
python tests/system.py
# Runs automated end-to-end checks across all pipeline stages including conversational memory
# Prints a scorecard at the end

# Test 3: Benchmark evaluation (requires API key)
python tests/evaluation.py
# Evaluates 25 legal questions with Recall@K, MRR, Hit Rate, groundedness
# Prints an ablation table: WITH vs WITHOUT query rewriting
```

## Evaluation

The project includes testing and evaluation scripts to validate the main RAG components:

- `tests/healthcheck.py` — checks core local components without requiring a Groq API call.
- `tests/system.py` — runs end-to-end system checks across the full pipeline.
- `tests/evaluation.py` — evaluates retrieval quality using Recall@K, MRR and Hit Rate.

Latency depends on local model cache status, hardware and Groq API response time.

---


## Key Settings (`RAG_Engine/config.py`)

| Setting | Default | What it controls |
|---|---|---|
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq LLM model |
| `CONFIDENCE_THRESHOLD` | `0.35` | Below this → full-corpus search |
| `DENSE_TOP_K` | `15` | FAISS candidates per query |
| `SPARSE_TOP_K` | `15` | BM25 candidates per query |
| `RERANK_TOP_N` | `3` | Chunks passed to the LLM |
| `REWRITE_MIN_TOKENS` | `3` | Minimum query length to trigger rewriting |
| `REWRITE_MAX_TOKENS` | `60` | Maximum (already detailed, skip rewriting) |
