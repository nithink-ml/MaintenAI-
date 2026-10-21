"""
MaintenAI— Hybrid Dense & Sparse Retrieval Engine

Combines dense vector similarity with BM25 keyword matching via Reciprocal Rank Fusion (RRF).
Crucial for industrial maintenance where exact model numbers, fault codes (HDF, TWF, E-102),
and semantic descriptions must both be accurately retrieved.
"""

from __future__ import annotations

import collections
import logging
import math
import re
from typing import TYPE_CHECKING

import numpy as np

from rag.embeddings.embedder import TextEmbedder

if TYPE_CHECKING:
    from rag.chunking.markdown_chunker import DocumentChunk

logger = logging.getLogger(__name__)


class BM25Index:
    """Lightweight in-memory BM25 index for sparse keyword retrieval."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_lengths: list[int] = []
        self.doc_freqs: list[dict[str, int]] = []
        self.idf: dict[str, float] = {}

    STOP_WORDS = {
        "a",
        "about",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "who",
        "will",
        "with",
    }

    def _tokenize(self, text: str) -> list[str]:
        words = re.findall(r"\b\w+\b", text.lower())
        return [w for w in words if w not in self.STOP_WORDS and len(w) > 1]

    def fit(self, documents: list[str]) -> None:
        self.corpus_size = len(documents)
        if self.corpus_size == 0:
            return

        self.doc_lengths = []
        self.doc_freqs = []
        df_counts: dict[str, int] = collections.defaultdict(int)

        total_length = 0
        for doc in documents:
            tokens = self._tokenize(doc)
            length = len(tokens)
            self.doc_lengths.append(length)
            total_length += length

            freqs: dict[str, int] = collections.defaultdict(int)
            for t in tokens:
                freqs[t] += 1
            self.doc_freqs.append(freqs)

            for t in set(tokens):
                df_counts[t] += 1

        self.avgdl = total_length / self.corpus_size if self.corpus_size > 0 else 1.0

        self.idf = {}
        for word, count in df_counts.items():
            # Standard Lucene BM25 IDF
            self.idf[word] = math.log(1.0 + (self.corpus_size - count + 0.5) / (count + 0.5))

    def score(self, query: str) -> np.ndarray:
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        if self.corpus_size == 0:
            return scores

        tokens = self._tokenize(query)
        for token in tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]
            for idx in range(self.corpus_size):
                tf = self.doc_freqs[idx].get(token, 0)
                if tf == 0:
                    continue
                doc_len = self.doc_lengths[idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                scores[idx] += idf_val * (numerator / (denominator + 1e-8))

        return scores


class HybridRetriever:
    """
    Hybrid retriever fusing dense vector search and sparse BM25 ranking via RRF.
    """

    def __init__(
        self,
        embedder: TextEmbedder | None = None,
        rrf_k: int = 60,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ) -> None:
        self.embedder = embedder or TextEmbedder()
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        self.chunks: list[DocumentChunk] = []
        self.embeddings: np.ndarray | None = None
        self.bm25 = BM25Index()

    def index(self, chunks: list[DocumentChunk]) -> None:
        """Index a collection of document chunks."""
        self.chunks = list(chunks)
        if not self.chunks:
            return

        texts = [f"{c.section_title} - {c.content}" for c in self.chunks]
        self.embeddings = self.embedder.embed_texts(texts)
        self.bm25.fit(texts)
        logger.info("HybridRetriever indexed %d chunks", len(self.chunks))

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        equipment_type: str | None = None,
        min_relevance: float = 0.0,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Execute hybrid search and return top K chunks with combined scores.
        """
        if not self.chunks or self.embeddings is None:
            return []

        # 1. Sparse BM25 scores
        sparse_scores = self.bm25.score(query)
        sparse_rank_indices = np.argsort(-sparse_scores)

        # 2. Dense vector cosine similarity
        query_vec = self.embedder.embed_query(query)
        dense_scores = np.dot(self.embeddings, query_vec)
        dense_rank_indices = np.argsort(-dense_scores)

        # 3. Reciprocal Rank Fusion (RRF) with Normalized Calibration
        rrf_scores: dict[int, float] = collections.defaultdict(float)
        max_possible_score = (self.dense_weight + self.sparse_weight) / (self.rrf_k + 1)

        # Only grant dense score if dense similarity has meaningful semantic alignment (> 0.15)
        for rank, idx in enumerate(dense_rank_indices):
            if dense_scores[idx] > 0.15:
                rrf_scores[idx] += (self.dense_weight / (self.rrf_k + rank + 1)) * float(
                    dense_scores[idx]
                )

        for rank, idx in enumerate(sparse_rank_indices):
            # Only add if BM25 score > 0
            if sparse_scores[idx] > 0:
                rrf_scores[idx] += self.sparse_weight / (self.rrf_k + rank + 1)

        # 4. Filter by equipment_type if specified and normalize scores to [0.0, 1.0]
        ranked_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results: list[tuple[DocumentChunk, float]] = []
        for idx, raw_score in ranked_items:
            chunk = self.chunks[idx]
            if (
                equipment_type
                and chunk.equipment_type != "general"
                and chunk.equipment_type != equipment_type
            ):
                continue

            # Calibrate score to [0.0, 1.0]
            calibrated_score = round(min(1.0, raw_score / max(max_possible_score, 1e-8)), 4)

            if calibrated_score >= min_relevance:
                results.append((chunk, float(calibrated_score)))

            if len(results) >= top_k:
                break

        return results
