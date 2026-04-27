import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from RAG_Engine.config import (
    EMBEDDING_MODEL_NAME,
    CONFIDENCE_THRESHOLD,
    DENSE_TOP_K,
    SPARSE_TOP_K,
    RERANK_TOP_N,
    REWRITE_MIN_TOKENS,
    REWRITE_MAX_TOKENS,
    WEAK_EVIDENCE_MSG,
)
from RAG_Engine.classifier import QueryClassifier
from RAG_Engine.retriever  import HybridRetriever
from RAG_Engine.reranker   import CrossEncoderReranker
from RAG_Engine.generator  import LegalGenerator
from RAG_Engine.memory     import ConversationMemory

logger = logging.getLogger(__name__)


# Result dataclass 

@dataclass
class RAGResult:
    """Full result object from a single pipeline run."""

    # Input
    original_query:   str  = ""
    resolved_query:   str  = ""   # after follow-up resolution (Stage 0)
    rewritten_query:  str  = ""   # after query rewriting     (Stage 3)
    rewrite_used:     bool = False
    followup_resolved: bool = False

    # Classification
    law_type:         str   = "unknown"
    confidence:       float = 0.0
    domain_filtered:  bool  = False
    domain_from_memory: bool = False   # True when domain was carried from prior turn

    # Retrieval
    retrieved_count:  int  = 0
    reranked_chunks:  list = field(default_factory=list)

    # Output
    answer:           str  = ""
    citations:        str  = ""
    is_grounded:      bool = False

    # Diagnostics
    latency_ms:       float = 0.0

    def __str__(self) -> str:
        rewrite_tag = "نعم" if self.rewrite_used else "لا"
        followup_tag = "نعم" if self.followup_resolved else "لا"
        domain_src   = " (من الذاكرة)" if self.domain_from_memory else ""
        lines = [
            "═" * 65,
            f"السؤال الأصلي      : {self.original_query}",
            f"بعد تفسير السياق   : {self.resolved_query}",
            f"بعد إعادة الصياغة  : {self.rewritten_query}",
            f"تفسير متابعة       : {followup_tag}",
            f"إعادة الصياغة      : {rewrite_tag}",
            f"النطاق القانوني    : {self.law_type}{domain_src} (ثقة: {self.confidence:.2f})",
            f"تصفية النطاق       : {'نعم' if self.domain_filtered else 'لا (بحث كامل)'}",
            f"المقاطع المسترجعة  : {self.retrieved_count} → {len(self.reranked_chunks)} بعد إعادة الترتيب",
            "─" * 65,
            self.answer,
        ]
        if self.citations:
            lines += ["", self.citations]
        lines += [
            "─" * 65,
            f"الوقت المستغرق     : {self.latency_ms:.0f} ms",
            "═" * 65,
        ]
        return "\n".join(lines)


#  Adaptive rewrite gate 

def _should_rewrite(question: str) -> bool:
    """
    Return True when query rewriting is likely to add value.
    Operates on the resolved query (after follow-up expansion).
    """
    n_tokens = len(question.split())
    return REWRITE_MIN_TOKENS <= n_tokens <= REWRITE_MAX_TOKENS


#  Main pipeline 

class EgyptianLegalRAG:
    """
    Production-grade RAG pipeline for Egyptian law Q&A with conversational memory.

    Initialization loads all models into memory once. Each call to query() is
    independent except for the shared ConversationMemory which persists across
    calls within the same process.

    Memory usage
    ------------
    # Stateless (no memory):
    result = rag.query("ما عقوبة السرقة؟")

    # Stateful (with memory):
    result = rag.query("ما عقوبة السرقة؟",     session_id="alice")
    result = rag.query("ولو كانت بالإكراه؟",  session_id="alice")  # follow-up resolved

    # Clear a session:
    rag.clear_session("alice")
    """

    def __init__(self) -> None:
        logger.info("Initializing Egyptian Legal RAG Engine...")
        t0 = time.perf_counter()

        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        self.classifier = QueryClassifier(self.embed_model)
        self.retriever  = HybridRetriever(self.embed_model)
        self.reranker   = CrossEncoderReranker()
        self.generator  = LegalGenerator()
        self.memory     = ConversationMemory()   # shared across all sessions

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(f"RAG Engine ready — init took {elapsed:.0f} ms")

    #  Public helpers 

    def clear_session(self, session_id: str) -> None:
        """Discard all conversation history for a session (e.g. on logout)."""
        self.memory.clear_session(session_id)
        logger.info(f"[Memory] Session '{session_id}' cleared")

    @property
    def active_sessions(self) -> int:
        """Number of sessions currently held in memory."""
        return self.memory.session_count

    #  Main entry point 

    def query(
        self,
        question: str,
        session_id: Optional[str]  = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        dense_top_k: int  = DENSE_TOP_K,
        sparse_top_k: int = SPARSE_TOP_K,
        rerank_top_n: int = RERANK_TOP_N,
        rewrite: bool     = True,
    ) -> RAGResult:
        """
        Run the full 7-stage pipeline on an Arabic legal question.

        Args:
            question:             Arabic legal question (any length / register).
            session_id:           Conversation session identifier. Pass the same
                                  string across turns to enable memory. None = stateless.
            confidence_threshold: Classifier confidence cutoff.
            dense_top_k:         FAISS candidates per query.
            sparse_top_k:        BM25 candidates per query.
            rerank_top_n:        Final chunks passed to LLM.
            rewrite:             Master switch for query rewriting.

        Returns:
            RAGResult dataclass with all fields populated.
        """
        t_start = time.perf_counter()
        result  = RAGResult(original_query=question)

        # Stage 0: Conversational memory 
        working_question = question   # this evolves through stages 0 → 3

        if session_id and self.memory.has_history(session_id):
            context = self.memory.format_context(session_id)

            if self.memory.is_followup(session_id, question):
                logger.info(
                    f"[Stage 0] Follow-up detected: '{question[:50]}' — resolving…"
                )
                working_question = self.generator.resolve_followup(question, context)
                result.followup_resolved = True
                logger.info(f"[Stage 0] Resolved → '{working_question[:70]}'")
            else:
                logger.info("[Stage 0] New question in existing session (no follow-up)")
        else:
            logger.info(
                f"[Stage 0] {'No session' if not session_id else 'First turn in session'}"
            )

        result.resolved_query = working_question

        # Stage 1: Classify query
        logger.info(f"[Stage 1] Classifying: '{working_question[:60]}'")
        clf_result = self.classifier.classify(working_question, confidence_threshold)
        result.law_type   = clf_result["law_type"]
        result.confidence = clf_result["confidence"]

        # Domain carry-over: if classifier is uncertain and memory has a domain, reuse it
        if (
            session_id
            and result.law_type == "unknown"
            and (mem_domain := self.memory.get_last_law_type(session_id))
        ):
            result.law_type = mem_domain
            result.confidence = confidence_threshold   # mark as minimum-threshold
            result.domain_from_memory = True
            logger.info(
                f"[Stage 1] Low confidence — carrying domain from memory: '{mem_domain}'"
            )
        else:
            logger.info(
                f"[Stage 1] → {result.law_type} (confidence={result.confidence:.3f})"
            )

        # Stage 2: Adaptive routing 
        domain_filter = None if result.law_type == "unknown" else result.law_type
        result.domain_filtered = domain_filter is not None
        logger.info(
            f"[Stage 2] Domain filter: {domain_filter or 'FULL CORPUS'}"
        )

        #  Stage 3: Query rewriting (adaptive gate) 
        apply_rewrite = rewrite and _should_rewrite(working_question)
        if apply_rewrite:
            logger.info("[Stage 3] Rewriting query…")
            working_question = self.generator.rewrite_query(working_question)
            result.rewrite_used = True
        else:
            reason = "rewrite=False" if not rewrite else "query already detailed"
            logger.info(f"[Stage 3] Skipping rewrite ({reason})")
            result.rewrite_used = False

        result.rewritten_query = working_question
        logger.info(f"[Stage 3] → '{working_question[:70]}'")

        #  Embed the final working question 
        query_vec = self.embed_model.encode(
            working_question,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        #  Stage 4: Hybrid retrieval (FAISS + BM25 → RRF) 
        logger.info("[Stage 4] Hybrid retrieval…")
        retrieved_chunks = self.retriever.retrieve(
            question     = working_question,
            query_vec    = query_vec,
            law_type     = domain_filter,
            dense_top_k  = dense_top_k,
            sparse_top_k = sparse_top_k,
        )
        result.retrieved_count = len(retrieved_chunks)
        logger.info(f"[Stage 4] → {result.retrieved_count} chunks")

        #  Stage 5: Cross-encoder reranking 
        logger.info("[Stage 5] Reranking…")
        result.reranked_chunks = self.reranker.rerank(
            query  = result.resolved_query,   # resolved (not rewritten) for fidelity
            chunks = retrieved_chunks,
            top_n  = rerank_top_n,
        )
        logger.info(f"[Stage 5] → {len(result.reranked_chunks)} chunks selected")

        #  Stage 6: Generate grounded Arabic answer 
        logger.info("[Stage 6] Generating answer…")
        gen_result = self.generator.generate_answer(
            question = result.resolved_query,   # original intent (not expanded)
            chunks   = result.reranked_chunks,
        )
        result.answer      = gen_result["answer"]
        result.citations   = gen_result["citations"]
        result.is_grounded = gen_result["is_grounded"]

        #  Stage 7: Save turn to memory 
        if session_id:
            self.memory.add_turn(
                session_id = session_id,
                question   = question,          # store original (not resolved/rewritten)
                answer     = result.answer,
                law_type   = result.law_type,
            )
            logger.debug(f"[Stage 7] Turn saved for session '{session_id}'")

        result.latency_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            f"[Done] grounded={result.is_grounded}, "
            f"followup={result.followup_resolved}, "
            f"rewrite={result.rewrite_used}, "
            f"latency={result.latency_ms:.0f}ms"
        )
        return result
