"""
retrieval.py
------------
Vector store and semantic retrieval using FAISS.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class VectorStore:
    """FAISS-based vector store for semantic chunk retrieval."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def build(self, chunks: list) -> None:
        """
        Embed chunks and build FAISS index.

        Args:
            chunks: List of text chunks from ingestion
        """
        self.chunks = chunks
        embeddings = self.model.encode(chunks, show_progress_bar=False).astype(np.float32)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """
        Retrieve top-k most relevant chunks for a query.

        Args:
            query: User question
            top_k: Number of chunks to retrieve

        Returns:
            List of relevant text chunks
        """
        if self.index is None:
            raise ValueError("Index not built. Call build() first.")

        query_embedding = self.model.encode([query]).astype(np.float32)
        distances, indices = self.index.search(query_embedding, top_k)

        return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]

    @property
    def size(self) -> int:
        """Number of chunks in the vector store."""
        return len(self.chunks)