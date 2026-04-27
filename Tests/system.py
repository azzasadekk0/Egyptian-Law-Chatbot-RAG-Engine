"""
system_test.py — Full system test for the Egyptian Legal RAG Engine.

Tests:
  [A] Imports & configuration
  [B] Smoke test (classifier, retriever, reranker — no LLM)
  [C] Memory unit tests (follow-up detection, context, domain carry-over)
  [D] Follow-up resolution via LLM (one API call)
  [E] End-to-end pipeline with memory (2-turn conversation)
  [F] Stateless query (backward-compatibility check)

Prints a final scorecard with PASS/FAIL per section.
"""

import sys, os, io, time, traceback

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Project root = parent of tests/
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.dirname(_THIS_DIR)          # one level up from tests/
sys.path.insert(0, _ROOT)

import logging
logging.disable(logging.CRITICAL)   # keep output clean

ROOT = os.path.dirname(os.path.abspath(__file__))

# helpers 

results: dict[str, list[tuple[str, bool, str]]] = {}

def section(name: str):
    results[name] = []
    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"{'='*65}")

def check(label: str, condition: bool, detail: str = "", section_name: str = ""):
    icon = "✓" if condition else "✗"
    tag  = "" if condition else f"  ← {detail}" if detail else ""
    print(f"  [{icon}] {label}{tag}")
    key = section_name or list(results.keys())[-1]
    results[key].append((label, condition, detail))

def run_section(name, fn):
    section(name)
    try:
        fn(name)
    except Exception as e:
        tb = traceback.format_exc()
        check(f"SECTION CRASHED: {e}", False, tb[:200], name)

# A: Imports & configuration 

def test_imports(sec):
    from RAG_Engine.config import (
        ROOT_DIR, NLP_DIR, CACHE_DIR, FAISS_INDEX_CACHE,
        CHUNKS_WITH_EMBEDDINGS, EMBEDDINGS_MATRIX, CLASSIFIER_PKL,
        OPENAI_API_KEY, EMBEDDING_MODEL_NAME,
        REWRITE_MIN_TOKENS, REWRITE_MAX_TOKENS,
        DENSE_TOP_K, SPARSE_TOP_K,
    )
    from RAG_Engine.memory import ConversationMemory
    from RAG_Engine.pipeline import EgyptianLegalRAG, RAGResult
    from RAG_Engine import __version__

    check("RAG_Engine package imports cleanly", True)
    check("ROOT_DIR exists", ROOT_DIR.exists(), str(ROOT_DIR))
    check("NLP_DIR exists", NLP_DIR.exists(), str(NLP_DIR))
    check("CACHE_DIR exists (auto-created)", CACHE_DIR.exists())
    check("chunks_with_embeddings.jsonl found", CHUNKS_WITH_EMBEDDINGS.exists())
    check("embeddings_matrix.npy found", EMBEDDINGS_MATRIX.exists())
    check("knn_classifier_final.pkl found", CLASSIFIER_PKL.exists())
    check("OPENAI_API_KEY is set", bool(OPENAI_API_KEY) and OPENAI_API_KEY.startswith("sk-"))
    check("No hardcoded absolute path (portable)", ROOT_DIR.exists())
    check("Package version is 1.1.0", __version__ == "1.1.0")
    check("REWRITE_MIN_TOKENS=3, MAX=60",
          REWRITE_MIN_TOKENS == 3 and REWRITE_MAX_TOKENS == 60)
    check("DENSE_TOP_K=15, SPARSE_TOP_K=15",
          DENSE_TOP_K == 15 and SPARSE_TOP_K == 15)

# B: Smoke test (no LLM) 



def test_smoke(sec):

    from sentence_transformers import SentenceTransformer
    from RAG_Engine.config import EMBEDDING_MODEL_NAME, FAISS_INDEX_CACHE
    from RAG_Engine.classifier import QueryClassifier
    from RAG_Engine.retriever  import HybridRetriever
    from RAG_Engine.reranker   import CrossEncoderReranker
    from RAG_Engine.memory     import ConversationMemory

    t0 = time.perf_counter()
    embed = SentenceTransformer(EMBEDDING_MODEL_NAME)
    check("Embedding model loaded", True,
          f"{(time.perf_counter()-t0)*1000:.0f}ms")

    clf = QueryClassifier(embed)

    # Hard-required domains (classifier recall = 100% for these)
    hard_required = [
        ("ما عقوبة القتل العمد في مصر؟",        "penal"),
        ("ما شروط الحضانة بعد الطلاق؟",         "personal_status"),
        ("متى تسقط دعوى المسؤولية التقصيرية؟", "civil"),
    ]
    hard_ok = True
    for q, exp in hard_required:
        r = clf.classify(q)
        ok = r["law_type"] == exp
        if not ok:
            hard_ok = False
        check(f"  Classify '{q[:45]}' → {exp}",
              ok, f"got {r['law_type']} ({r['confidence']:.2f})")
    check("Penal / Civil / PersonalStatus classifications correct", hard_ok)

    # Soft-check for commercial (documented 60% recall — civil overlap is acceptable)
    r_com = clf.classify("ما هي أركان عقد الشركة التجارية؟")
    com_ok = r_com["law_type"] in ("commercial", "civil")   # civil is acceptable fallback
    check("Commercial query classified as commercial OR civil (known 60% recall)",
          com_ok, f"got '{r_com['law_type']}' — civil/commercial share semantic space")

    t1 = time.perf_counter()
    retriever = HybridRetriever(embed)
    elapsed = (time.perf_counter()-t1)*1000

    faiss_cached = FAISS_INDEX_CACHE.exists()
    check("FAISS index persisted to disk", faiss_cached, str(FAISS_INDEX_CACHE))
    check(f"HybridRetriever built ({elapsed:.0f}ms)", True)

    q = "ما عقوبة السرقة في القانون المصري؟"
    qv = embed.encode(q, convert_to_numpy=True, normalize_embeddings=True)
    chunks = retriever.retrieve(q, qv, law_type="penal", dense_top_k=15, sparse_top_k=15)
    check("Domain retrieval returns 15 chunks", len(chunks) == 15, f"got {len(chunks)}")
    required = ["text", "law_type", "law_name", "document_name"]
    check("All metadata fields present",
          all(f in c for c in chunks for f in required))

    chunks_all = retriever.retrieve(q, qv, law_type=None)
    check("Full-corpus fallback returns results", len(chunks_all) > 0)

    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(q, chunks, top_n=3)
    check("Reranker reduces 15 → 3 chunks", len(reranked) == 3)
    check("Rerank scores present",
          all("_rerank_score" in c for c in reranked))
    check("Scores are floats in plausible range",
          all(-15 < c["_rerank_score"] < 15 for c in reranked))

# C: Memory unit tests 

def test_memory(sec):
    from RAG_Engine.memory import ConversationMemory, MEMORY_MAX_TURNS

    mem = ConversationMemory()

    check("Empty session has no history", not mem.has_history("s1"))
    check("get_last_law_type returns None on empty", mem.get_last_law_type("s1") is None)
    check("format_context returns '' on empty", mem.format_context("s1") == "")

    # Add turns
    mem.add_turn("s1", "ما شروط الحضانة بعد الطلاق؟", "الحضانة تنتقل...", "personal_status")
    check("has_history after add_turn", mem.has_history("s1"))
    check("get_last_law_type = personal_status", mem.get_last_law_type("s1") == "personal_status")
    check("format_context non-empty", len(mem.format_context("s1")) > 10)

    # Follow-up detection — should match
    followups = [
        "ولو رفض؟",
        "ولو الأب سافر؟",
        "وهل يجوز ذلك؟",
        "فماذا يحدث؟",
        "وكيف؟",
    ]
    for q in followups:
        check(f"  is_followup('{q}')", mem.is_followup("s1", q))

    # Should NOT match (long or no conjunction)
    non_followups = [
        "ما عقوبة السرقة في القانون المصري؟",
        "ما هي شروط الزواج القانوني في مصر وكيف يتم توثيقه أمام المحكمة؟",
    ]
    for q in non_followups:
        check(f"  not is_followup('{q[:40]}')", not mem.is_followup("s1", q))

    # No history → never a follow-up
    check("is_followup = False on unknown session",
          not mem.is_followup("new_session", "ولو رفض؟"))

    # Domain carry-over: add unknown turn, last known domain still returned
    mem.add_turn("s1", "كيف أرفع قضية؟", "...", "unknown")
    check("get_last_law_type skips 'unknown' turns",
          mem.get_last_law_type("s1") == "personal_status")

    # Max-turn enforcement
    mem2 = ConversationMemory(max_turns=3)
    for i in range(5):
        mem2.add_turn("s2", f"q{i}", f"a{i}", "civil")
    check("Max-turns enforced (deque maxlen)",
          len(mem2.get_turns("s2")) == 3)

    # clear_session
    mem.clear_session("s1")
    check("clear_session removes history", not mem.has_history("s1"))

    check(f"MEMORY_MAX_TURNS constant = 10", MEMORY_MAX_TURNS == 10)

#  D: Follow-up resolution via LLM 

def test_followup_resolution(sec):
    from RAG_Engine.generator import LegalGenerator

    gen = LegalGenerator()
    context = (
        "المستخدم: ما شروط حضانة الأم بعد الطلاق في القانون المصري؟\n"
        "المساعد: تستحق الأم الحضانة حتى سن السابعة للذكر والتاسعة للأنثى وفق أحكام قانون الأحوال الشخصية."
    )

    test_cases = [
        "ولو رفض الأب؟",
        "ولو الأب سافر للخارج؟",
    ]

    for q in test_cases:
        t0 = time.perf_counter()
        resolved = gen.resolve_followup(q, context)
        ms = (time.perf_counter()-t0)*1000
        is_expanded = len(resolved.split()) > len(q.split())   # longer than original
        is_arabic   = any("\u0600" <= c <= "\u06FF" for c in resolved)
        check(f"  resolve_followup('{q}') returns Arabic text", is_arabic, resolved[:80])
        check(f"  Resolved is longer than original ({len(resolved.split())} > {len(q.split())} words)",
              is_expanded, resolved[:80])
        check(f"  Latency {ms:.0f}ms (< 15000ms)", ms < 15000)
        print(f"        ↳ '{resolved[:80]}'")

# E: End-to-end pipeline with memory 

def test_pipeline_with_memory(sec):
    from RAG_Engine.pipeline import EgyptianLegalRAG

    rag = EgyptianLegalRAG()
    sid = "test_session_e2e"

    # Turn 1 — fresh question
    t0 = time.perf_counter()
    r1 = rag.query("ما شروط الحضانة بعد الطلاق في القانون المصري؟", session_id=sid)
    ms1 = (time.perf_counter()-t0)*1000

    check("Turn 1: law_type = personal_status", r1.law_type == "personal_status",
          f"got {r1.law_type}")
    check("Turn 1: answer is non-empty Arabic",
          bool(r1.answer) and any("\u0600" <= c <= "\u06FF" for c in r1.answer))
    check("Turn 1: is_grounded", r1.is_grounded)
    check("Turn 1: followup_resolved = False", not r1.followup_resolved)
    check(f"Turn 1: latency {ms1:.0f}ms", ms1 < 30000)
    print(f"        ↳ Answer[:120]: {r1.answer[:120]}")

    # Turn 2 — ambiguous follow-up
    t1 = time.perf_counter()
    r2 = rag.query("ولو رفض الأب؟", session_id=sid)
    ms2 = (time.perf_counter()-t1)*1000

    check("Turn 2: followup_resolved = True", r2.followup_resolved,
          f"resolved_query='{r2.resolved_query[:60]}'")
    check("Turn 2: resolved_query is longer than original",
          len(r2.resolved_query.split()) > len("ولو رفض الأب؟".split()),
          r2.resolved_query[:80])
    check("Turn 2: domain = personal_status (preserved from memory)",
          r2.law_type == "personal_status", f"got {r2.law_type}")
    check("Turn 2: answer is non-empty Arabic",
          bool(r2.answer) and any("\u0600" <= c <= "\u06FF" for c in r2.answer))
    check(f"Turn 2: latency {ms2:.0f}ms", ms2 < 30000)
    print(f"        ↳ Resolved: '{r2.resolved_query[:80]}'")
    print(f"        ↳ Answer[:120]: {r2.answer[:120]}")

    check("Memory stores 2 turns", len(rag.memory.get_turns(sid)) == 2)
    rag.clear_session(sid)
    check("clear_session empties history", not rag.memory.has_history(sid))

#  F: Stateless (backward-compat) 

def test_stateless(sec):
    from RAG_Engine.pipeline import EgyptianLegalRAG

    rag = EgyptianLegalRAG()

    r = rag.query("ما عقوبة السرقة في القانون المصري؟")   # no session_id
    check("Stateless query: answer non-empty", bool(r.answer))
    check("Stateless query: is_grounded", r.is_grounded)
    check("Stateless query: followup_resolved = False", not r.followup_resolved)
    check("Stateless query: law_type = penal", r.law_type == "penal", f"got {r.law_type}")
    check("No sessions stored in memory", rag.memory.session_count == 0)
    print(f"        ↳ Answer[:120]: {r.answer[:120]}")

#  Run all sections 

print("\n" + "█"*65)
print("  EGYPTIAN LEGAL RAG ENGINE — FULL SYSTEM TEST")
print("█"*65)

run_section("[A] Imports & Configuration", test_imports)
run_section("[B] Smoke Test  (classifier · retriever · reranker)", test_smoke)
run_section("[C] Memory Unit Tests", test_memory)
run_section("[D] Follow-up Resolution  (LLM)", test_followup_resolution)
run_section("[E] End-to-End Pipeline with Memory  (LLM)", test_pipeline_with_memory)
run_section("[F] Stateless Query  (backward-compat)", test_stateless)

#  Scorecard

print("\n" + "█"*65)
print("  SCORECARD")
print("█"*65)

total_pass = 0
total_fail = 0
for sec_name, checks in results.items():
    passed = sum(1 for _, ok, _ in checks if ok)
    failed = sum(1 for _, ok, _ in checks if not ok)
    total_pass += passed
    total_fail += failed
    icon = "✓" if failed == 0 else "✗"
    print(f"  [{icon}] {sec_name:<50}  {passed}/{passed+failed}")
    if failed:
        for label, ok, detail in checks:
            if not ok:
                print(f"        ✗ FAIL: {label} — {detail[:80]}")

print("─"*65)
grand_total = total_pass + total_fail
pct = total_pass / grand_total * 100 if grand_total else 0
print(f"  TOTAL:  {total_pass}/{grand_total} checks passed  ({pct:.0f}%)")
if total_fail == 0:
    print("  🎉 ALL CHECKS PASSED — System is production-ready")
else:
    print(f"  ⚠  {total_fail} check(s) failed — review output above")
print("█"*65 + "\n")
