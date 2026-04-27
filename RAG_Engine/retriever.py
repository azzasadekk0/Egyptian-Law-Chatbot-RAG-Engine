import json
import logging
import re
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from RAG_Engine.config import (
    CHUNKS_WITH_EMBEDDINGS,
    EMBEDDINGS_MATRIX,
    FAISS_INDEX_CACHE,
    DENSE_TOP_K,
    SPARSE_TOP_K,
    RRF_K,
)

logger = logging.getLogger(__name__)


#  Arabic normalisation & tokenisation 

# Characters to strip completely (diacritics + tatweel)
_DIACRITICS = re.compile(
    r"[\u064B-\u065F"   # fathatan … sukun (all tashkeel)
    r"\u0670"           # alef wasla superscript
    r"\u0640]"          # tatweel (kashida)
)

# Alef variants → bare alef
_ALEF_VARIANTS = re.compile(r"[إأآٱ]")

# Teh marbuta → heh  (so حضانة == حضانه in keyword matching)
_TEH_MARBUTA = re.compile(r"ة")

# Waw with hamza above → plain waw
_WAW_HAMZA = re.compile(r"ؤ")

# Yeh variants → bare yeh (alef maqsoura → yeh)
_YEH_VARIANTS = re.compile(r"[ىئ]")

# Keep only Arabic letters and digits; split on everything else
_ARABIC_TOKEN = re.compile(r"[\u0600-\u06FF\d]+")

# Very short tokens add noise to BM25 (single letters are often particles)
_MIN_TOKEN_LEN = 2

# Common Arabic stop words (light set — keeps meaningful legal particles)
_STOP_WORDS = {
    "في", "من", "إلى", "على", "عن", "مع", "هذا", "هذه", "ذلك",
    "التي", "الذي", "وهو", "وهي", "أو", "ثم", "قد", "لم", "لا",
    "ما", "هل", "كان", "كانت", "يكون",
}


def _normalise_arabic(text: str) -> str:
    """
    Apply a consistent normalisation pipeline to Arabic text before tokenisation.
    This greatly improves BM25 recall by collapsing orthographic variants:
        - Strip diacritics (tashkeel) and tatweel
        - Unify alef variants (أإآٱ → ا)
        - Unify teh marbuta (ة → ه)
        - Unify waw with hamza (ؤ → و)
        - Unify yeh/alef maqsoura variants (ى/ئ → ي)
    """
    text = _DIACRITICS.sub("", text)
    text = _ALEF_VARIANTS.sub("ا", text)
    text = _TEH_MARBUTA.sub("ه", text)
    text = _WAW_HAMZA.sub("و", text)
    text = _YEH_VARIANTS.sub("ي", text)
    return text


def _tokenise_arabic(text: str) -> list[str]:
    """
    Full Arabic tokenisation pipeline for BM25:
        1. Normalise orthography
        2. Extract Arabic/digit tokens
        3. Remove stop words and very short tokens
    Returns a non-empty token list (falls back to whitespace split if needed).
    """
    normalised = _normalise_arabic(text)
    tokens = _ARABIC_TOKEN.findall(normalised)
    tokens = [t for t in tokens if len(t) >= _MIN_TOKEN_LEN and t not in _STOP_WORDS]
    return tokens if tokens else normalised.split()


#  HybridRetriever 

class HybridRetriever:
    """
    Loads the NLP engineer's artifacts and exposes a single retrieve() method.

    Persistence
    -----------
    The FAISS index is expensive to rebuild (allocates + normalises all vectors).
    After the first build it is serialised to FAISS_INDEX_CACHE so subsequent
    process starts load it in < 1 s.
    """

    def __init__(self, embed_model: SentenceTransformer) -> None:
        self.embed_model = embed_model
        self._load_chunks()
        self._load_or_build_faiss_index()
        self._build_bm25_index()

    #  Initialization 

    def _load_chunks(self) -> None:
        """Load all records from chunks_with_embeddings.jsonl."""
        logger.info(f"Loading chunks from: {CHUNKS_WITH_EMBEDDINGS}")
        self.records: list[dict] = []
        with open(CHUNKS_WITH_EMBEDDINGS, "r", encoding="utf-8") as f:
            for line in f:
                self.records.append(json.loads(line))

        logger.info(f"Loaded {len(self.records)} chunks")

        # Load the pre-computed matrix for FAISS (only needed if rebuilding)
        self.embeddings_matrix = np.load(EMBEDDINGS_MATRIX).astype(np.float32)
        logger.info(f"Embeddings matrix shape: {self.embeddings_matrix.shape}")

    def _load_or_build_faiss_index(self) -> None:
        """
        Load a persisted FAISS index from disk, or build + save one if absent.
        Using IndexFlatIP on L2-normalised vectors → cosine similarity.
        IndexIDMap allows mapping result IDs back to chunk indices.
        """
        cache_path: Path = FAISS_INDEX_CACHE

        if cache_path.exists():
            logger.info(f"Loading persisted FAISS index from: {cache_path}")
            self.faiss_index = faiss.read_index(str(cache_path))
            logger.info(
                f"FAISS index loaded — {self.faiss_index.ntotal} vectors, "
                f"dim={self.faiss_index.d}"
            )
        else:
            logger.info("FAISS index not found — building from embeddings matrix …")
            dim = self.embeddings_matrix.shape[1]   # 768
            base_index = faiss.IndexFlatIP(dim)
            self.faiss_index = faiss.IndexIDMap(base_index)

            ids = np.arange(len(self.records), dtype=np.int64)
            self.faiss_index.add_with_ids(self.embeddings_matrix, ids)
            logger.info(
                f"FAISS index built — {self.faiss_index.ntotal} vectors, dim={dim}"
            )

            # Persist for future runs
            faiss.write_index(self.faiss_index, str(cache_path))
            logger.info(f"FAISS index saved to: {cache_path}")

    def _build_bm25_index(self) -> None:
        """
        Build one BM25 index over ALL normalised chunk texts.
        Domain filtering is done by masking at query time.
        """
        corpus_tokens = [_tokenise_arabic(r["text"]) for r in self.records]
        self.bm25 = BM25Okapi(corpus_tokens)
        logger.info("BM25 index built over full corpus (normalised Arabic)")

    #  Domain helpers 

    def _get_domain_indices(self, law_type: Optional[str]) -> list[int]:
        """
        Return chunk indices belonging to the specified domain.
        Returns all indices when law_type is None or 'unknown'.
        """
        if not law_type or law_type == "unknown":
            return list(range(len(self.records)))
        indices = [
            i for i, r in enumerate(self.records)
            if r.get("law_type") == law_type
        ]
        logger.debug(f"Domain filter '{law_type}' → {len(indices)} chunks")
        return indices

    #  Individual retrievers 

    def _dense_retrieve(
        self, query_vec: np.ndarray, domain_indices: list[int], top_k: int
    ) -> list[tuple[int, float]]:
        """
        FAISS search restricted to domain_indices.
        Returns [(chunk_idx, score), ...] sorted by score DESC.
        """
        # Over-fetch then filter (FAISS doesn't natively support ID filtering)
        fetch_k = min(len(self.records), top_k * 10)
        q = query_vec.reshape(1, -1).astype(np.float32)
        scores_arr, idx_arr = self.faiss_index.search(q, fetch_k)

        domain_set = set(domain_indices)
        results = []
        for score, idx in zip(scores_arr[0], idx_arr[0]):
            if idx in domain_set:
                results.append((int(idx), float(score)))
            if len(results) >= top_k:
                break

        return results  # already sorted by FAISS score DESC

    def _sparse_retrieve(
        self, question: str, domain_indices: list[int], top_k: int
    ) -> list[tuple[int, float]]:
        """
        BM25 search restricted to domain_indices.
        Query is normalised using the same pipeline as corpus tokens.
        Returns [(chunk_idx, score), ...] sorted by score DESC.
        """
        tokens = _tokenise_arabic(question)
        all_scores = self.bm25.get_scores(tokens)      # shape (N,)

        # Mask to domain
        domain_scores = [
            (idx, all_scores[idx]) for idx in domain_indices
        ]
        domain_scores.sort(key=lambda x: x[1], reverse=True)
        return domain_scores[:top_k]

    #  RRF Fusion 

    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: list[list[tuple[int, float]]],
        k: int = RRF_K,
    ) -> list[tuple[int, float]]:
        """
        Merge multiple ranked lists using Reciprocal Rank Fusion.
        Formula: RRF(d) = Σ 1 / (k + rank(d))

        Args:
            ranked_lists: Each list is [(chunk_idx, score), ...] sorted DESC.
            k:            RRF constant (default 60 from original RRF paper).

        Returns:
            [(chunk_idx, rrf_score), ...] sorted DESC by rrf_score.
        """
        rrf_scores: dict[int, float] = {}
        for ranked_list in ranked_lists:
            for rank, (chunk_idx, _) in enumerate(ranked_list, start=1):
                rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + 1.0 / (k + rank)

        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    #  Public API 

    def retrieve(
        self,
        question: str,
        query_vec: np.ndarray,
        law_type: Optional[str] = None,
        dense_top_k: int = DENSE_TOP_K,
        sparse_top_k: int = SPARSE_TOP_K,
    ) -> list[dict]:
        """
        Hybrid retrieval: FAISS dense + BM25 sparse → RRF fusion.

        Args:
            question:    Original (or rewritten) Arabic question text.
            query_vec:   L2-normalized embedding vector (768,).
            law_type:    Domain from classifier. None/\"unknown\" = no filter.
            dense_top_k: How many FAISS results to fetch.
            sparse_top_k:How many BM25 results to fetch.

        Returns:
            List of chunk dicts (from records), sorted by RRF score DESC.
            Each dict has: id, text, law_type, law_name, article_number,
                           document_name, year, page_start, page_end, chunk_index.
            The embedding field is stripped for efficiency.
        """
        domain_indices = self._get_domain_indices(law_type)

        # Dense retrieval
        dense_results  = self._dense_retrieve(query_vec, domain_indices, dense_top_k)
        # Sparse retrieval (normalised query)
        sparse_results = self._sparse_retrieve(question, domain_indices, sparse_top_k)

        # RRF fusion
        fused = self._reciprocal_rank_fusion([dense_results, sparse_results])

        # Fetch top chunks (up to max(dense_top_k, sparse_top_k), de-duped)
        max_results = max(dense_top_k, sparse_top_k)
        top_indices = [idx for idx, _ in fused[:max_results]]

        # Build result dicts (strip embedding to save memory)
        results = []
        for idx in top_indices:
            chunk = {k: v for k, v in self.records[idx].items() if k != "embedding"}
            results.append(chunk)

        logger.debug(
            f"Retrieved {len(results)} chunks for law_type='{law_type}' "
            f"(dense={len(dense_results)}, sparse={len(sparse_results)})"
        )
        return results
