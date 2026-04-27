import pickle
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

from RAG_Engine.config import (
    CLASSIFIER_PKL,
    EMBEDDING_MODEL_NAME,
    CONFIDENCE_THRESHOLD,
)

logger = logging.getLogger(__name__)


class QueryClassifier:
    """
    Hybrid KNN + Keyword classifier for Egyptian law domain routing.
    Reuses the knn_classifier_final.pkl produced by the NLP engineer.
    """

    def __init__(self, embed_model: SentenceTransformer) -> None:
        """
        Args:
            embed_model: Already-loaded SentenceTransformer instance.
                         Must be paraphrase-multilingual-mpnet-base-v2
                         to match the training embeddings.
        """
        self.embed_model = embed_model
        self._load_classifier()

    def _load_classifier(self) -> None:
        """Load the pkl and unpack all components."""
        logger.info(f"Loading classifier from: {CLASSIFIER_PKL}")
        with open(CLASSIFIER_PKL, "rb") as f:
            clf_data = pickle.load(f)

        self.knn             = clf_data["knn"]
        self.boost_weight    = clf_data["boost_weight"]          # 0.6
        self.domain_keywords = clf_data["domain_keywords"]       # {domain: [kws]}
        self.all_classes     = list(clf_data["all_classes"])     # 4 domains
        # Normalize numpy str_ to plain str
        self.all_classes = [str(c) for c in self.all_classes]

        logger.info(
            f"Classifier loaded — K={clf_data['k']}, "
            f"boost_weight={self.boost_weight}, "
            f"domains={self.all_classes}"
        )

    # Internal helpers

    def _embed_query(self, question: str) -> np.ndarray:
        """Encode question with the same model + normalization used during training."""
        return self.embed_model.encode(
            question,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def _keyword_boost(self, question: str) -> dict:
        """
        Score each domain by keyword presence.
        Multi-word keywords weighted ×2 (stronger signal).
        Returns normalized {domain: score} in [0, 1].
        """
        scores = {domain: 0.0 for domain in self.domain_keywords}
        for domain, keywords in self.domain_keywords.items():
            for kw in keywords:
                if kw in question:
                    scores[domain] += 2.0 if " " in kw else 1.0

        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        return scores

    # Public API 

    def classify(
        self,
        question: str,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> dict:
        """
        Classify an Arabic legal query into one of 4 law domains.

        Args:
            question:             Arabic legal question string.
            confidence_threshold: Minimum score to assign a domain.
                                  Below this → returns "unknown".

        Returns:
            {
                "law_type"  : "civil" | "penal" | "commercial" |
                              "personal_status" | "unknown",
                "confidence": float   [0.0, 1.0]
            }
        """
        # Step 1 — KNN semantic probabilities
        q_emb     = self._embed_query(question).reshape(1, -1)
        knn_proba = self.knn.predict_proba(q_emb)[0]
        knn_dict  = dict(zip(self.all_classes, knn_proba))

        # Step 2 — Keyword boost (adaptive: 0 weight if no keywords found)
        kw_dict    = self._keyword_boost(question)
        has_signal = any(v > 0 for v in kw_dict.values())
        w          = self.boost_weight if has_signal else 0.0

        # Step 3 — Weighted combination
        final_scores = {
            domain: (1 - w) * knn_dict.get(domain, 0.0)
                  + w       * kw_dict.get(domain, 0.0)
            for domain in self.all_classes
        }

        # Step 4 — Pick top domain, apply threshold
        top_domain = max(final_scores, key=final_scores.get)
        top_score  = round(final_scores[top_domain], 4)

        if top_score < confidence_threshold:
            logger.debug(
                f"Low confidence ({top_score:.3f}) for '{question[:50]}' "
                "→ returning unknown"
            )
            return {"law_type": "unknown", "confidence": top_score}

        logger.debug(f"Classified '{question[:50]}' → {top_domain} ({top_score:.3f})")
        return {"law_type": top_domain, "confidence": top_score}
