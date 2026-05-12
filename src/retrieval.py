"""
Embedding and retrieval utilities.

The default implementation uses sentence-transformers for embeddings
and scikit-learn cosine similarity for retrieval.

Students may replace this with FAISS, Chroma, LlamaIndex, LangChain,
or another approved method, but must justify the design in the report.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


Chunk = Dict[str, Any]


class EmbeddingRetriever:
    """
    Simple embedding-based retriever.
    """

    def __init__(self, embedding_model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for this starter retriever. "
                "Install with: pip install sentence-transformers"
            ) from exc

        self.model = SentenceTransformer(embedding_model_name)
        self.chunks: List[Chunk] = []
        self.embeddings: np.ndarray | None = None

    def build_index(self, chunks: List[Chunk]) -> None:
        """
        Create embeddings for all chunks.
        """
        if not chunks:
            raise ValueError("No chunks supplied to build_index().")

        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]
        self.embeddings = np.asarray(self.model.encode(texts, show_progress_bar=True))

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """
        Retrieve top-k chunks for a query.
        """
        if self.embeddings is None:
            raise RuntimeError("Index has not been built. Call build_index() first.")

        query_embedding = np.asarray(self.model.encode([query]))
        scores = cosine_similarity(query_embedding, self.embeddings)[0]

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]
