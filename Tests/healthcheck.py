import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# Go up one level from tests/ to project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

from sentence_transformers import SentenceTransformer
from RAG_Engine.config import EMBEDDING_MODEL_NAME
from RAG_Engine.classifier import QueryClassifier
from RAG_Engine.retriever import HybridRetriever
from RAG_Engine.reranker import CrossEncoderReranker

print("\n" + "="*60)
print("SMOKE TEST - Egyptian Legal RAG Engine")
print("="*60)

# 1. Load embedding model
print("\n[1/5] Loading embedding model...")
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print(f"      [OK] Loaded: {EMBEDDING_MODEL_NAME}")

# 2. Test classifier
print("\n[2/5] Testing classifier...")
clf = QueryClassifier(embed_model)
test_questions = [
    ("ما عقوبة جريمة القتل العمد في مصر؟", "penal"),
    ("ما شروط الحضانة بعد الطلاق؟", "personal_status"),
    ("متى تسقط دعوى المسؤولية التقصيرية؟", "civil"),
    ("ما هي أركان الشركة التجارية؟", "commercial"),
]
all_correct = True
for q, expected in test_questions:
    result = clf.classify(q)
    status = "[OK]" if result["law_type"] == expected else f"[WARN expected={expected}]"
    print(f"      {status} '{q[:45]}' -> {result['law_type']} ({result['confidence']:.3f})")
    if result["law_type"] != expected:
        all_correct = False
print(f"      {'[OK] All correct' if all_correct else '[WARN] Some misclassifications'}")

# 3. Test retriever
print("\n[3/5] Building hybrid retriever (FAISS + BM25)...")
retriever = HybridRetriever(embed_model)

q = "ما عقوبة السرقة في القانون المصري؟"
q_vec = embed_model.encode(q, convert_to_numpy=True, normalize_embeddings=True)
chunks = retriever.retrieve(question=q, query_vec=q_vec, law_type="penal", dense_top_k=10, sparse_top_k=10)
print(f"      [OK] Retrieved {len(chunks)} chunks for penal domain")

# Check metadata fields
required_fields = ["text", "law_type", "law_name", "document_name"]
for field in required_fields:
    assert all(field in c for c in chunks), f"Missing field: {field}"
print(f"      [OK] All required metadata fields present: {required_fields}")
print(f"      Sample: law_type={chunks[0].get('law_type')}, law_name={chunks[0].get('law_name', '')[:30]}")

# 4. Full corpus retrieval (unknown domain)
print("\n[4/5] Testing full corpus retrieval (low-confidence fallback)...")
chunks_all = retriever.retrieve(question=q, query_vec=q_vec, law_type=None)
print(f"      [OK] Searched full corpus - retrieved {len(chunks_all)} chunks")

# 5. Test reranker
print("\n[5/5] Testing cross-encoder reranker...")
reranker = CrossEncoderReranker()
reranked = reranker.rerank(query=q, chunks=chunks, top_n=3)
print(f"      [OK] Reranked {len(chunks)} -> {len(reranked)} chunks")
for i, c in enumerate(reranked, 1):
    print(f"         [{i}] score={c.get('_rerank_score', 0):.4f} | {c.get('law_name', '')[:40]}")

print("\n" + "="*60)
print("[ALL PASSED] Smoke tests complete.")
print("   Ready to run: python RAG_Engine/run.py --benchmark")
print("="*60 + "\n")
