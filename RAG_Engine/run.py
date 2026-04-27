import argparse
import logging
import sys
import os
import uuid

# Force UTF-8 output on Windows (prevents cp1252 encoding errors with Arabic)
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Use cached models only — skip HuggingFace update checks for faster startup
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RAG_Engine.pipeline import EgyptianLegalRAG

#  Logging setup 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run")

#  Benchmark questions (one per domain) 
BENCHMARK_QUESTIONS = [
    # Penal
    "ما عقوبة جريمة القتل العمد في القانون المصري؟",
    # Civil
    "متى تسقط دعوى المسؤولية التقصيرية بالتقادم؟",
    # Commercial
    "ما هي أركان عقد الشركة التجارية وشروط تأسيسها؟",
    # Personal status
    "ما شروط الحضانة وحق الزيارة بعد الطلاق في مصر؟",
    # Edge case: ambiguous / out-of-scope
    "ما هو الدستور المصري؟",
]


def run_single(rag: EgyptianLegalRAG, question: str, rewrite: bool = True) -> None:
    """Run one question and print the full result."""
    print(f"\n>> {question}\n")
    result = rag.query(question, rewrite=rewrite)
    print(result)


def run_benchmark(rag: EgyptianLegalRAG, rewrite: bool = True) -> None:
    """Run all benchmark questions and print results."""
    print("\n" + "═" * 65)
    print("  BENCHMARK — Egyptian Legal RAG Engine")
    print("=" * 65)
    for i, question in enumerate(BENCHMARK_QUESTIONS, start=1):
        print(f"\n[{i}/{len(BENCHMARK_QUESTIONS)}]")
        run_single(rag, question, rewrite=rewrite)


def run_interactive(rag: EgyptianLegalRAG, rewrite: bool = True) -> None:
    """REPL loop with conversational memory — type a question, get an answer."""
    session_id = str(uuid.uuid4())
    print("\n" + "═" * 65)
    print("  Egyptian Legal RAG — Interactive Mode")
    print("  اكتب سؤالك القانوني بالعربية | 'quit' للخروج")
    print(f"  Session: {session_id[:8]}…")
    print("=" * 65)
    while True:
        try:
            question = input("\nالسؤال: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nخروج...")
            break
        if question.lower() in ("quit", "exit", "q", "خروج"):
            print("تم الخروج.")
            break
        if not question:
            continue
        result = rag.query(question, session_id=session_id, rewrite=rewrite)
        print(result)


#  Main 

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Egyptian Legal RAG Engine — CLI"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--question",    type=str, help="Single Arabic legal question")
    group.add_argument("--benchmark",   action="store_true", help="Run benchmark questions")
    group.add_argument("--interactive", action="store_true", help="Interactive REPL mode")
    parser.add_argument("--no-rewrite", action="store_true",
                        help="Skip query rewriting step (faster)")
    args = parser.parse_args()

    rewrite = not args.no_rewrite

    # Load the pipeline (expensive — models load here)
    logger.info("Loading RAG Engine...")
    rag = EgyptianLegalRAG()
    logger.info("RAG Engine ready.\n")

    if args.question:
        run_single(rag, args.question, rewrite=rewrite)
    elif args.benchmark:
        run_benchmark(rag, rewrite=rewrite)
    elif args.interactive:
        run_interactive(rag, rewrite=rewrite)
    else:
        # Default: run benchmark
        run_benchmark(rag, rewrite=rewrite)


if __name__ == "__main__":
    main()
