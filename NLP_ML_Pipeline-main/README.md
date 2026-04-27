# 🤖 AI Legal Chatbot — NLP/ML Engineer Deliverables

**Project:** Egyptian Law Legal Chatbot (RAG-Based)
**Role:** NLP / ML Engineer

---

## 📁 Output Files

| File | Description |
|------|-------------|
| `chunks_cleaned.jsonl` | Cleaned legal chunks — OCR-fixed, metadata-validated (742 records) |
| `chunks_with_embeddings.jsonl` | Chunks + 768-dim embedding vectors (742 records) |
| `embeddings_matrix.npy` | Embeddings-only matrix for fast loading — shape (742 × 768) |
| `knn_classifier_final.pkl` | Trained hybrid KNN + keyword classifier |

---

## ⚙️ Pipeline Overview

```
chunks_improved.jsonl         (from Data Engineer)
        │
        ▼
  [Step 1] Clean & Validate
        │  • Fix OCR noise in Arabic text
        │  • Fix civil law filename (YYYYYYY → Civil-Law.pdf)
        │  • Validate all metadata fields
        ▼
  chunks_cleaned.jsonl
        │
        ▼
  [Step 2] Model Selection
        │  • Benchmarked 3 Arabic/multilingual models
        │  • Selected: paraphrase-multilingual-mpnet-base-v2
        │  • Best gap score: 0.41 (relevant vs irrelevant)
        ▼
  [Step 3] Build Embeddings
        │  • Batch size: 32 (T4-safe)
        │  • Normalized vectors (L2 norm = 1.0)
        │  • Embedding dim: 768
        ▼
  chunks_with_embeddings.jsonl + embeddings_matrix.npy
        │
        ▼
  [Step 4] Classification Model
        │  • Hybrid: KNN (cosine) + Keyword Boost
        │  • K=7, boost_weight=0.6
        │  • Accuracy: 90%
        ▼
  knn_classifier_final.pkl
```

---

## 🧠 Embedding Model

| Property | Value |
|----------|-------|
| Model | `paraphrase-multilingual-mpnet-base-v2` |
| Dimension | 768 |
| Normalization | L2 normalized |
| Language | Multilingual (Arabic-capable) |
| Device | CUDA (T4) |

**Why this model?** Benchmarked against `CAMeL-Lab/bert-base-arabic-camelbert-mix` and `intfloat/multilingual-e5-base`. Selected based on highest discrimination gap (0.41) between relevant and irrelevant legal passages.

---

## 🎯 Classification Model

### Architecture: Hybrid KNN + Keyword Boost

```
classify_query(question)
        │
        ├─► KNN on embeddings  →  probability vector (4 classes)
        │
        ├─► Keyword matching   →  boost vector (4 classes)
        │       • Multi-word keywords weighted ×2
        │       • Adaptive: boost=0 if no keywords found
        │
        └─► Weighted combine   →  final scores → top domain
```

### Performance

| Domain | Precision | Recall | F1 |
|--------|-----------|--------|----|
| civil | 0.83 | 1.00 | 0.91 |
| penal | 1.00 | 1.00 | 1.00 |
| commercial | 1.00 | 0.60 | 0.75 |
| personal_status | 0.83 | 1.00 | 0.91 |
| **Overall** | **0.92** | **0.90** | **0.89** |

**Overall Accuracy: 90%**

---

## 🔌 How to Use (for RAG Engineer)

### Load Embeddings
```python
import numpy as np, json

# Fast matrix load
embeddings = np.load("embeddings_matrix.npy")       # (742, 768)

# Full records with metadata + embeddings
records = []
with open("chunks_with_embeddings.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))
```

### Classify a Query
```python
import pickle
from sentence_transformers import SentenceTransformer

with open("knn_classifier_final.pkl", "rb") as f:
    clf_data = pickle.load(f)

# Then call classify_query() from Step 4 notebook
result = classify_query("ما عقوبة السرقة في مصر؟")
# → {"law_type": "penal", "confidence": 1.0}
```

### Metadata Schema (per chunk)
```json
{
  "id":             "uuid:chunk_index",
  "text":           "نص المادة القانونية...",
  "doc_id":         "uuid",
  "document_name":  "Penal-Code.pdf",
  "law_type":       "penal",
  "law_name":       "قانون العقوبات المصري",
  "year":           "1937",
  "article_number": "230",
  "page_start":     45,
  "page_end":       45,
  "chunk_index":    12,
  "char_len":       1150,
  "embedding":      [0.023, -0.041, "..."]
}
```

---

## ⚠️ Known Limitations

1. **Commercial recall = 0.60** — 2 questions about company structure misclassify. Can be improved with more domain-specific keywords.
2. **OCR noise** — ~12% of chunks have residual Arabic OCR artifacts. Partially cleaned.
3. **article_number null** — 36% of chunks (mostly commercial) have no article number, which limits metadata filtering.

---

## 🛠️ Requirements

```
sentence-transformers
scikit-learn
numpy
torch
```

---
*NLP/ML Engineer — Egyptian Law AI Chatbot Project*
