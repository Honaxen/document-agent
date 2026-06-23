"""
retrieval.py
------------
Hybrid retrieval: FAISS (dense) + BM25 (sparse) with score fusion.

Upgrade from v1:
  v1 — only FAISS semantic search
  v2 — BM25 + FAISS, scores combined with Reciprocal Rank Fusion (RRF)

Why hybrid?
  FAISS is great for semantic similarity ("what does X mean?")
  BM25 is great for exact keyword matches ("what is the value of X?")
  Combining both covers cases neither handles alone.
"""

import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


# ─────────────────────────────────────────────
# BM25 retriever
# ─────────────────────────────────────────────

class BM25Retriever:
    """Sparse keyword-based retrieval using BM25Okapi."""

    def __init__(self):
        self.bm25 = None
        self.chunks = []

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'\b[a-z]{2,}\b', text.lower())

    def build(self, chunks: list[str]) -> None:
        self.chunks = chunks
        tokenized = [self._tokenize(chunk) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """
        Returns list of (chunk_index, score) sorted by score descending.
        """
        if self.bm25 is None:
            raise ValueError("BM25 index not built. Call build() first.")

        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]


# ─────────────────────────────────────────────
# FAISS retriever (dense)
# ─────────────────────────────────────────────

class FAISSRetriever:
    """Dense semantic retrieval using sentence-transformers + FAISS."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def build(self, chunks: list[str]) -> None:
        self.chunks = chunks
        embeddings = self.model.encode(
            chunks, show_progress_bar=False).astype(np.float32)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """
        Returns list of (chunk_index, distance) sorted by distance ascending.
        Lower distance = more similar.
        """
        if self.index is None:
            raise ValueError("FAISS index not built. Call build() first.")

        query_embedding = self.model.encode([query]).astype(np.float32)
        distances, indices = self.index.search(query_embedding, top_k)
        return [(int(indices[0][i]), float(distances[0][i])) for i in range(len(indices[0]))]


# ─────────────────────────────────────────────
# Reciprocal Rank Fusion
# ─────────────────────────────────────────────

def _reciprocal_rank_fusion(
    bm25_results: list[tuple[int, float]],
    faiss_results: list[tuple[int, float]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """
    Combine BM25 and FAISS results using Reciprocal Rank Fusion.

    RRF score = 1/(k + rank_in_bm25) + 1/(k + rank_in_faiss)
    k=60 is the standard constant that smooths out high-rank advantages.

    Returns list of (chunk_index, rrf_score) sorted by score descending.
    """
    scores: dict[int, float] = {}

    for rank, (idx, _) in enumerate(bm25_results):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

    for rank, (idx, _) in enumerate(faiss_results):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results


# ─────────────────────────────────────────────
# VectorStore — unified interface (backwards compatible)
# ─────────────────────────────────────────────

class VectorStore:
    """
    Hybrid retrieval: FAISS (dense) + BM25 (sparse) + RRF fusion.

    Backwards compatible with v1 — existing code using
    VectorStore.build() and VectorStore.retrieve() works unchanged.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.faiss = FAISSRetriever(model_name)
        self.bm25 = BM25Retriever()
        self.chunks = []

    def build(self, chunks: list[str]) -> None:
        """Build both FAISS and BM25 indices."""
        self.chunks = chunks
        self.faiss.build(chunks)
        self.bm25.build(chunks)

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """
        Hybrid retrieval: combine FAISS + BM25 with RRF, return top_k chunks.

        Args:
            query: User question
            top_k: Number of chunks to return

        Returns:
            List of relevant text chunks (best first)
        """
        if not self.chunks:
            raise ValueError("Index not built. Call build() first.")

        candidate_k = min(top_k * 3, len(self.chunks))

        bm25_results = self.bm25.retrieve(query, top_k=candidate_k)
        faiss_results = self.faiss.retrieve(query, top_k=candidate_k)

        fused = _reciprocal_rank_fusion(bm25_results, faiss_results)
        top_indices = [idx for idx, _ in fused[:top_k]]

        return [self.chunks[i] for i in top_indices if i < len(self.chunks)]

    def retrieve_with_scores(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Same as retrieve() but returns scores too — useful for evaluation.

        Returns:
            List of {"chunk": str, "rrf_score": float, "index": int}
        """
        if not self.chunks:
            raise ValueError("Index not built. Call build() first.")

        candidate_k = min(top_k * 3, len(self.chunks))

        bm25_results = self.bm25.retrieve(query, top_k=candidate_k)
        faiss_results = self.faiss.retrieve(query, top_k=candidate_k)

        fused = _reciprocal_rank_fusion(bm25_results, faiss_results)

        return [
            {
                "chunk": self.chunks[idx],
                "rrf_score": round(score, 6),
                "index": idx,
            }
            for idx, score in fused[:top_k]
            if idx < len(self.chunks)
        ]

    @property
    def size(self) -> int:
        return len(self.chunks)
