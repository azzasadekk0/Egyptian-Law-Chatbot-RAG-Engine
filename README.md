# Egyptian Law Chatbot_RAG Engine

A **Retrieval-Augmented Generation (RAG)** system that answers Arabic legal questions about Egyptian law with grounded citations — covering **Civil**, **Penal**, **Commercial**, and **Personal Status** law.

---

## Features

- **Domain-Aware Routing:** Automatically classifies queries into 4 legal domains using a custom KNN classifier.
- **Adaptive Hybrid Retrieval:** Combines FAISS (dense) and BM25 (sparse) with Reciprocal Rank Fusion (RRF) for high-precision semantic and keyword matching.
- **Conversational Memory:** Seamlessly handles ambiguous follow-up questions (e.g., "ولو رفض؟") using zero-latency heuristic detection and LLM resolution.
- **Strict Grounding:** Zero hallucination policy. The system cites exact articles and falls back to a standard refusal if evidence is missing.
- **Cross-Encoder Reranking:** Re-scores the top 15 hybrid results down to the top 3 most relevant chunks using a multilingual cross-encoder.

---


## How It Works

Every query goes through 7 stages:

1. **Memory** — detect ambiguous follow-ups and resolve them using conversation history
2. **Classify** — route the query to a legal domain (civil / penal / commercial / personal_status)
3. **Route** — search that domain's chunks; fall back to full corpus if classifier confidence is low
4. **Rewrite** — expand short/ambiguous phrasing into formal legal terminology (GPT-4o)
5. **Retrieve** — top-15 results from FAISS (semantic) + BM25 (keyword), merged with RRF
6. **Rerank** — multilingual cross-encoder narrows 15 → 3 highest-relevance chunks
7. **Generate** — GPT-4o produces a grounded Arabic answer with citations (temperature 0)

---

## Project Structure

```text
├── RAG_Engine/
│   ├── config.py        ← all settings and paths
│   ├── classifier.py    ← domain classifier (KNN + keyword boost)
│   ├── retriever.py     ← FAISS + BM25 hybrid retrieval with Arabic normalisation
│   ├── reranker.py      ← cross-encoder reranker
│   ├── generator.py     ← follow-up resolver + query rewriter + answer generator
│   ├── citations.py     ← citation formatter
│   ├── memory.py        ← conversational memory (per-session, last 10 turns)
│   ├── pipeline.py      ← orchestrates all 7 stages
│   └── run.py           ← CLI entrypoint
│
├── NLP_ML_Pipeline-main/
│   ├── chunks_with_embeddings.jsonl   ← 742 legal chunks + embeddings
│   ├── embeddings_matrix.npy          ← (742 × 768) float32 matrix
│   ├── knn_classifier_final.pkl       ← trained classifier (sklearn 1.6.1)
│   └── chunks_cleaned.jsonl           ← text + metadata only
│
├── tests/
│   ├── healthcheck.py   ← fast component check, no API call needed
│   ├── evaluation.py    ← 25-question evaluation (Recall@K, MRR, Hit Rate)
│   └── system.py        ← full 66-check system test covering all 7 stages
└── requirements.txt
```

---

## Setup

**Requirements:** Python 3.12+, an OpenAI API key with GPT-4o access.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
echo OPENAI_API_KEY=sk-...your-key... > .env
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

# Test 1: Smoke test (no API call, ~30s on first run)
python Tests/healthcheck.py
# Validates: embedding model, classifier, retriever, reranker
# No OpenAI API call — safe to run offline

# Test 2: Full system test (requires API key, ~3-5 min) 
python Tests/system.py
# Runs 66 checks across all 7 stages including conversational memory
# Prints a scorecard at the end — look for "ALL CHECKS PASSED"

# Test 3: Benchmark evaluation (requires API key, ~10-15 min) 
python Tests/evaluation.py
# Evaluates 25 legal questions with Recall@K, MRR, Hit Rate, groundedness
# Prints an ablation table: WITH vs WITHOUT query rewriting
```

## Performance / Evaluation

The system is continuously evaluated using our testing suites (`Tests/evaluation.py` and `Tests/system.py`):

- **System Reliability:** **66/67 automated checks passed (99% pass rate)** across end-to-end system validation covering retrieval, memory, reranking, and generation.
- **Retrieval Metrics:** Evaluated using **Recall@K**, **MRR**, and **Hit Rate** with comparative testing **with vs without query rewriting** to measure retrieval effectiveness.
- **Latency:** **Sub-second local retrieval** after cache warm-up. Total end-to-end response time varies depending on OpenAI API latency and query complexity.
---

## Python API

```python
from RAG_Engine import EgyptianLegalRAG

rag = EgyptianLegalRAG()   # loads all models once

# Stateless query
result = rag.query("ما عقوبة السرقة في القانون المصري؟")
print(result.answer)
print(result.citations)

# Multi-turn conversation with memory
r1 = rag.query("ما شروط الحضانة بعد الطلاق؟",  session_id="user_1")
r2 = rag.query("ولو رفض الأب؟",                session_id="user_1")  # auto-resolved
r3 = rag.query("ولو الأب سافر للخارج؟",         session_id="user_1")  # auto-resolved

# Clear session history
rag.clear_session("user_1")
```

---


## Key Settings (`RAG_Engine/config.py`)

| Setting | Default | What it controls |
|---|---|---|
| `LLM_MODEL` | `gpt-4o` | OpenAI model |
| `CONFIDENCE_THRESHOLD` | `0.35` | Below this → full-corpus search |
| `DENSE_TOP_K` | `15` | FAISS candidates per query |
| `SPARSE_TOP_K` | `15` | BM25 candidates per query |
| `RERANK_TOP_N` | `3` | Chunks passed to the LLM |
| `REWRITE_MIN_TOKENS` | `3` | Minimum query length to trigger rewriting |
| `REWRITE_MAX_TOKENS` | `60` | Maximum (already detailed, skip rewriting) |
