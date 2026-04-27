import logging
import os
import warnings
from unittest.mock import patch

from RAG_Engine.config import RERANKER_MODEL, RERANK_TOP_N

logger = logging.getLogger(__name__)


def _safe_auto_processor_from_pretrained(original_fn):
    """
    Wrap AutoProcessor.from_pretrained so it returns None instead of
    raising ValueError on models that don't ship a processor config.
    sentence-transformers >= 5.x calls AutoProcessor during CrossEncoder
    init; mmarco cross-encoders don't have one, causing a hard crash.
    """
    def wrapper(*args, **kwargs):
        try:
            return original_fn(*args, **kwargs)
        except (ValueError, OSError):
            return None
    return wrapper


class CrossEncoderReranker:
    """Reranks retrieved chunks using a multilingual cross-encoder model."""

    def __init__(self) -> None:
        logger.info(f"Loading cross-encoder: {RERANKER_MODEL}")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

        # sentence-transformers 5.x tries AutoProcessor.from_pretrained()
        # which raises ValueError for cross-encoder models without a
        # processor config. We patch it to return None instead.
        from transformers.models.auto.processing_auto import AutoProcessor
        original = AutoProcessor.from_pretrained

        with warnings.catch_warnings(), \
             patch.object(AutoProcessor, "from_pretrained",
                          _safe_auto_processor_from_pretrained(original)):
            warnings.filterwarnings("ignore", message="Unrecognized processing class")
            warnings.filterwarnings("ignore", message="Can't instantiate")
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(
                RERANKER_MODEL,
                max_length=512,
            )
        logger.info("Cross-encoder loaded")

    def rerank(self, query: str, chunks: list[dict], top_n: int = RERANK_TOP_N) -> list[dict]:
        """
        Score (query, chunk_text) pairs and return the top_n highest-scoring chunks.
        Each returned chunk dict gains a '_rerank_score' field.
        """
        if not chunks:
            return []

        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = self.model.predict(pairs, show_progress_bar=False)

        scored_chunks = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)

        top_chunks = []
        for score, chunk in scored_chunks[:top_n]:
            chunk_with_score = dict(chunk)
            chunk_with_score["_rerank_score"] = round(float(score), 4)
            top_chunks.append(chunk_with_score)

        logger.debug(
            f"Reranked {len(chunks)} → {len(top_chunks)} chunks. "
            f"Top score: {top_chunks[0]['_rerank_score'] if top_chunks else 'N/A'}"
        )
        return top_chunks
