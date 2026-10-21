"""
MaintenAI— Cross-Encoder Document Reranker

Reranks candidate retrieval chunks using deep cross-attention for high precision.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rag.chunking.markdown_chunker import DocumentChunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks top candidate chunks using Cross-Encoder model.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
            except Exception as e:
                logger.warning(
                    "Could not load CrossEncoder (%s). Using pass-through reranker: %s",
                    self.model_name,
                    e,
                )
                self._model = "fallback"
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[tuple[DocumentChunk, float]],
        top_k: int = 5,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Rerank (chunk, score) candidate pairs.
        """
        if not candidates:
            return []

        model = self._get_model()
        if model != "fallback":
            pairs = [[query, c.content] for c, _ in candidates]
            scores = model.predict(pairs)
            # Softmax or sigmoid normalization
            norm_scores = 1.0 / (1.0 + np.exp(-scores))
            ranked = sorted(
                zip([c for c, _ in candidates], norm_scores.tolist(), strict=False),
                key=lambda x: x[1],
                reverse=True,
            )
            return ranked[:top_k]

        # Fallback: preserve existing ranking and take top_k
        return candidates[:top_k]
