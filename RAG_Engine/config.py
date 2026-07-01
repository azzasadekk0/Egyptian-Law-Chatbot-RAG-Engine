import os
from pathlib import Path
from dotenv import load_dotenv

# Always resolve from this file's location. RAG_ROOT env-var overrides for Docker/CI.
_env_root = os.environ.get("RAG_ROOT", "")
ROOT_DIR: Path = Path(_env_root).resolve() if _env_root else Path(__file__).parent.parent.resolve()

NLP_DIR = ROOT_DIR / "NLP_ML_Pipeline-main"
RAG_DIR = ROOT_DIR / "RAG_Engine"

CACHE_DIR = ROOT_DIR / ".cache_rag"
CACHE_DIR.mkdir(exist_ok=True)
FAISS_INDEX_CACHE = CACHE_DIR / "faiss_index.bin"   # persisted binary index

# Read-only artifacts produced by the NLP engineer
CHUNKS_WITH_EMBEDDINGS = NLP_DIR / "chunks_with_embeddings.jsonl"
EMBEDDINGS_MATRIX      = NLP_DIR / "embeddings_matrix.npy"
CLASSIFIER_PKL         = NLP_DIR / "knn_classifier_final.pkl"
CHUNKS_CLEANED         = NLP_DIR / "chunks_cleaned.jsonl"

# Must match the model used during training of the NLP artifacts
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_DIM        = 768

CONFIDENCE_THRESHOLD = 0.35   # below this → search full corpus (no domain filter)
KNOWN_LAW_TYPES      = ["civil", "commercial", "penal", "personal_status"]

DENSE_TOP_K   = 15    # FAISS dense retrieval top-K
SPARSE_TOP_K  = 15    # BM25 sparse retrieval top-K
RRF_K         = 60    # Reciprocal Rank Fusion constant (standard value)
RERANK_TOP_N  = 3     # Final chunks passed to LLM after reranking

REWRITE_MIN_TOKENS = 3    # queries shorter than this get rewritten
REWRITE_MAX_TOKENS = 60   # very long queries already carry enough context

RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

LLM_MODEL       = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.0    # deterministic — legal answers must be grounded
LLM_MAX_TOKENS  = 1024

load_dotenv(ROOT_DIR / ".env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY not found. "
        "Create a .env file in the project root with: GROQ_API_KEY=gsk_..."
    )

WEAK_EVIDENCE_MSG = (
    "لم يتم العثور على نص قانوني صريح ضمن البيانات المتاحة."
)
