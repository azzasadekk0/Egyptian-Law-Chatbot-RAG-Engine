"""
evaluation.py — Comprehensive evaluation harness for the Egyptian Legal RAG Engine.

Metrics computed per question and aggregated:
    - Retrieval Recall@K : fraction of expected law_type found in top-K chunks
    - Hit Rate           : 1 if at least 1 relevant chunk retrieved, else 0
    - MRR                : Mean Reciprocal Rank (rank of first relevant chunk)
    - Groundedness       : LLM produced a grounded answer (not fallback)
    - Latency (ms)       : end-to-end wall-clock time
    - Rewrite used       : whether adaptive gate triggered rewriting

Two ablation modes are compared automatically:
    A) With query rewriting
    B) Without query rewriting
This directly answers "does rewriting add measurable value?"

Test set covers all 4 legal domains + ambiguous / edge-case queries.
"""

import sys
import os
import io

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.disable(logging.CRITICAL)   # suppress model loading noise

from RAG_Engine.pipeline import EgyptianLegalRAG, RAGResult

# ── Benchmark dataset ─────────────────────────────────────────────────────────
# Format: (question, expected_law_type)
# expected_law_type is used to compute recall / hit-rate metrics.
# Use None for truly ambiguous queries where any domain is acceptable.

BENCHMARK_CASES: list[tuple[str, str | None]] = [
    # ── Penal law ──────────────────────────────────────────────────────────────
    ("ما عقوبة جريمة القتل العمد في القانون المصري؟",          "penal"),
    ("ما عقوبة السرقة بالإكراه في مصر؟",                      "penal"),
    ("ما هو حكم جريمة الاختلاس في مصر؟",                      "penal"),
    ("ما عقوبة تزوير المحررات الرسمية؟",                       "penal"),
    ("ما الفرق بين الجنحة والجناية في القانون المصري؟",        "penal"),

    # ── Civil law ─────────────────────────────────────────────────────────────
    ("متى تسقط دعوى المسؤولية التقصيرية بالتقادم؟",           "civil"),
    ("ما هي شروط صحة العقد في القانون المدني المصري؟",        "civil"),
    ("ما أثر الغلط والتدليس على العقود في القانون المدني؟",   "civil"),
    ("كيف يُحسب التعويض عن الضرر المادي والأدبي؟",            "civil"),
    ("ما حق المشتري عند ظهور عيب خفي في المبيع؟",             "civil"),

    # ── Commercial law ───────────────────────────────────────────────────────
    ("ما هي أركان عقد الشركة التجارية وشروط تأسيسها؟",        "commercial"),
    ("ما التزامات التاجر بموجب القانون التجاري المصري؟",       "commercial"),
    ("ما أحكام الإفلاس والتسوية الواقية في مصر؟",             "commercial"),
    ("ما هي الأوراق التجارية وشروط صحتها؟",                   "commercial"),
    ("ما شروط تسجيل العلامة التجارية في مصر؟",                "commercial"),

    # ── Personal status law ──────────────────────────────────────────────────
    ("ما شروط الحضانة بعد الطلاق في القانون المصري؟",         "personal_status"),
    ("ما حقوق المرأة في الميراث بالقانون المصري؟",             "personal_status"),
    ("ما شروط الزواج القانوني وإجراءاته في مصر؟",             "personal_status"),
    ("ما أحكام النفقة للزوجة والأولاد بعد الانفصال؟",         "personal_status"),
    ("هل يجوز الطلاق بالتراضي أمام المحكمة في مصر؟",         "personal_status"),

    # ── Ambiguous / edge-case queries ────────────────────────────────────────
    ("ما هو الدستور المصري؟",                                  None),   # out of scope
    ("ما حقوق العامل في حالة الفصل التعسفي؟",                 None),   # labour law
    ("كيف أرفع دعوى قضائية في مصر؟",                          None),   # procedural
    ("ما الفرق بين العقد والوعد بالتعاقد؟",                   "civil"), # conceptual
    ("هل يُعاقب على الشروع في الجريمة كالجريمة التامة؟",     "penal"), # nuanced penal
]

K_VALUES = [1, 3, 5, 10]   # Recall@K evaluated at these cut-offs


# ── Metric helpers ────────────────────────────────────────────────────────────

def _extract_law_types(result: RAGResult) -> list[str]:
    """Extract law_type values from all reranked chunks."""
    return [c.get("law_type", "") for c in result.reranked_chunks]


def _recall_at_k(retrieved_types: list[str], expected: str | None, k: int) -> float:
    """1.0 if expected law_type appears in top-k retrieved chunks, else 0.0."""
    if expected is None:
        return float("nan")   # skip for ambiguous queries
    return 1.0 if expected in retrieved_types[:k] else 0.0


def _hit_rate(retrieved_types: list[str], expected: str | None) -> float:
    """1.0 if expected law_type appears in ANY retrieved chunk."""
    if expected is None:
        return float("nan")
    return 1.0 if expected in retrieved_types else 0.0


def _mrr(retrieved_types: list[str], expected: str | None) -> float:
    """Mean Reciprocal Rank — 1/rank of first relevant chunk (0 if not found)."""
    if expected is None:
        return float("nan")
    for rank, lt in enumerate(retrieved_types, start=1):
        if lt == expected:
            return 1.0 / rank
    return 0.0


def _safe_mean(values: list[float]) -> float:
    """Mean of non-NaN values."""
    valid = [v for v in values if v == v]   # NaN != NaN
    return sum(valid) / len(valid) if valid else float("nan")


# ── Run one ablation pass ─────────────────────────────────────────────────────

def run_pass(rag: EgyptianLegalRAG, rewrite: bool, label: str) -> dict:
    """
    Run the full benchmark with rewrite=True or rewrite=False.
    Returns aggregated metrics dict.
    """
    print(f"\n{'='*70}")
    print(f"  ABLATION PASS: {label}")
    print(f"{'='*70}")

    rows: list[dict] = []

    for i, (question, expected_domain) in enumerate(BENCHMARK_CASES, 1):
        result = rag.query(question, rewrite=rewrite)
        retrieved_types = _extract_law_types(result)

        row = {
            "q": question,
            "expected": expected_domain,
            "predicted": result.law_type,
            "confidence": result.confidence,
            "rewrite_used": result.rewrite_used,
            "retrieved_count": result.retrieved_count,
            "grounded": result.is_grounded,
            "latency_ms": result.latency_ms,
            "retrieved_types": retrieved_types,
            "hit": _hit_rate(retrieved_types, expected_domain),
            "mrr": _mrr(retrieved_types, expected_domain),
        }
        for k in K_VALUES:
            row[f"recall@{k}"] = _recall_at_k(retrieved_types, expected_domain, k)

        rows.append(row)

        # Per-question summary
        domain_ok = (
            "✓" if expected_domain and result.law_type == expected_domain
            else ("?" if expected_domain is None else "✗")
        )
        rw_tag = "RW" if result.rewrite_used else "  "
        grounded_tag = "✓" if result.is_grounded else "✗"
        recall3 = row["recall@3"]
        recall3_str = f"{recall3:.0f}" if recall3 == recall3 else "?"
        print(
            f"[{i:2d}/{len(BENCHMARK_CASES)}] [{rw_tag}] domain:{domain_ok} "
            f"recall@3:{recall3_str} grounded:{grounded_tag} "
            f"{result.latency_ms:5.0f}ms | {question[:45]}"
        )

    # ── Aggregate metrics ──────────────────────────────────────────────────────
    agg = {
        "label":       label,
        "n_questions": len(rows),
        "hit_rate":    _safe_mean([r["hit"]  for r in rows]),
        "mrr":         _safe_mean([r["mrr"]  for r in rows]),
        "groundedness":_safe_mean([float(r["grounded"]) for r in rows]),
        "latency_avg": sum(r["latency_ms"] for r in rows) / len(rows),
        "rewrite_pct": sum(r["rewrite_used"] for r in rows) / len(rows) * 100,
    }
    for k in K_VALUES:
        agg[f"recall@{k}"] = _safe_mean([r[f"recall@{k}"] for r in rows])

    return agg


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("  EGYPTIAN LEGAL RAG ENGINE — EVALUATION BENCHMARK")
    print(f"  {len(BENCHMARK_CASES)} questions | K={K_VALUES}")
    print("=" * 70)

    rag = EgyptianLegalRAG()

    # Run both ablation passes
    pass_with    = run_pass(rag, rewrite=True,  label="WITH query rewriting")
    pass_without = run_pass(rag, rewrite=False, label="WITHOUT query rewriting")

    # ── Print comparison table ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  COMPARISON TABLE")
    print(f"{'='*70}")
    header = f"{'Metric':<22} {'WITH rewrite':>14} {'WITHOUT rewrite':>16} {'Δ':>8}"
    print(header)
    print("-" * 70)

    metrics_to_compare = (
        [f"recall@{k}" for k in K_VALUES]
        + ["hit_rate", "mrr", "groundedness", "latency_avg", "rewrite_pct"]
    )

    for m in metrics_to_compare:
        w   = pass_with.get(m, float("nan"))
        wo  = pass_without.get(m, float("nan"))
        if w == w and wo == wo:
            delta = w - wo
            delta_str = f"{delta:+.3f}"
        else:
            delta_str = "N/A"
        w_str  = f"{w:.3f}"  if w  == w  else "N/A"
        wo_str = f"{wo:.3f}" if wo == wo else "N/A"
        print(f"  {m:<20} {w_str:>14} {wo_str:>16} {delta_str:>8}")

    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
