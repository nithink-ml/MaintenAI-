"""
MaintenAI— Sentence Transformer Embedding Generator

Computes dense vector representations for maintenance manuals and diagnostic queries.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class TextEmbedder:
    """
    Text embedder using SentenceTransformers with fallback and vector normalization.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.warning(
                    "Could not load SentenceTransformer (%s). Using fallback embedder: %s",
                    self.model_name,
                    e,
                )
                self._model = "fallback"
        return self._model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of text strings into float32 numpy array (N, D)."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        model = self._get_model()
        if model != "fallback":
            embeddings = model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=self.normalize_embeddings,
            )
            return np.asarray(embeddings, dtype=np.float32)

        # Fast deterministic fallback with stable cross-process hashing
        import hashlib

        dim = 384
        vectors = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            words = text.lower().split()
            vec = np.zeros(dim, dtype=np.float32)
            for w in words:
                idx = int(hashlib.md5(w.encode("utf-8"), usedforsecurity=False).hexdigest()[:8], 16) % dim
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            vectors[i] = vec
        return vectors

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string into (D,) float32 array."""
        return self.embed_texts([query])[0]
