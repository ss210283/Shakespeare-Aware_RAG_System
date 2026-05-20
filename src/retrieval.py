"""
Embedding-based retrieval for the Shakespeare-aware RAG system.

Two ways to use this module:

1. The top-level ``retrieve`` function (recommended for assignment-style code):

       from retrieval import retrieve
       hits = retrieve("Why does Macbeth kill Duncan?", top_k=3)
       # hits is List[Dict] — each dict is the chunk's metadata
       # ({play, act, scene, speaker, text, source_id, ...}) plus a "score" key.

   The first call lazily loads the index built by ``python src/build_index.py``
   (``data/processed/embeddings.npy`` + ``data/processed/chunks_meta.json``).
   Later calls reuse the same model and matrix.

2. The :class:`EmbeddingRetriever` class for explicit lifecycle control or for
   building the index in memory from a list of chunks (used by
   ``rag_chatbot.py``).

Scoring is cosine similarity computed directly with NumPy — no FAISS, no
sklearn. Both the index and the query vector are L2-normalised at build /
encode time, so cosine similarity reduces to a single ``embeddings @ q``
matrix-vector product.
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import DATA_DIR, EMBEDDING_MODEL_NAME


Chunk = Dict[str, Any]

EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
META_PATH = DATA_DIR / "chunks_meta.json"


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the top-k scores in descending order."""
    k = max(1, min(k, scores.shape[0]))
    if k == scores.shape[0]:
        return np.argsort(-scores)
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part])]


# ---------------------------------------------------------------------------
# Retriever class
# ---------------------------------------------------------------------------

class EmbeddingRetriever:
    """Cosine retriever over MiniLM embeddings, NumPy-only.

    The class can either:

    * load a pre-built index from disk (``load_index``); or
    * embed a list of chunks in memory on the fly (``build_index``) — handy
      for tests and for the scaffolded ``rag_chatbot.py`` path.
    """

    def __init__(self, embedding_model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = embedding_model_name
        self._model = None  # lazy: only loaded when we actually need to encode
        self.embeddings: Optional[np.ndarray] = None
        self.chunks: List[Chunk] = []

    # ---- lazy model loader ----

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required. Install with "
                    "`pip install -r requirements.txt`."
                ) from e
            self._model = SentenceTransformer(self.model_name)
        return self._model

    # ---- index construction / loading ----

    def build_index(self, chunks: List[Chunk]) -> None:
        """Embed ``chunks`` in memory (used by tests and the chatbot scaffold)."""
        if not chunks:
            raise ValueError("No chunks supplied to build_index().")
        texts = [c.get("text", "") for c in chunks]
        if any(not t.strip() for t in texts):
            raise ValueError("Some chunks have empty text.")
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float32")
        # Guarantee unit norm even if encode() didn't (float32 rounding etc).
        self.embeddings = _l2_normalise(embeddings)
        self.chunks = chunks

    def load_index(
        self,
        embeddings_path: Path = EMBEDDINGS_PATH,
        meta_path: Path = META_PATH,
    ) -> None:
        if not embeddings_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Saved index not found at {embeddings_path} / {meta_path}.\n"
                "Run `python src/build_index.py` first."
            )
        embeddings = np.load(embeddings_path).astype("float32")
        with meta_path.open("r", encoding="utf-8") as f:
            chunks = json.load(f)
        if len(chunks) != embeddings.shape[0]:
            raise RuntimeError(
                f"index/metadata mismatch: {embeddings.shape[0]} embeddings "
                f"vs {len(chunks)} metadata rows. Rebuild the index."
            )
        # Defensive: re-normalise in case the saved file wasn't normalised.
        self.embeddings = _l2_normalise(embeddings)
        self.chunks = chunks

    # ---- query ----

    def _encode_query(self, query: str) -> np.ndarray:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        q = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")[0]
        # Belt-and-suspenders normalisation.
        n = np.linalg.norm(q)
        return q / max(n, 1e-12)

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Return ``[(chunk_dict, cosine_score), ...]`` for the top-``top_k`` hits.

        This signature is kept for backward compatibility with
        ``rag_chatbot.py``. New code should prefer the top-level
        :func:`retrieve` function, which returns chunk dicts directly with a
        ``score`` field.
        """
        if self.embeddings is None:
            raise RuntimeError(
                "Index not initialised. Call load_index() or build_index() first."
            )
        q = self._encode_query(query)
        scores = self.embeddings @ q  # cosine because both sides are unit-norm
        idx = _topk_indices(scores, top_k)
        return [(self.chunks[i], float(scores[i])) for i in idx]


# ---------------------------------------------------------------------------
# Module-level singleton + top-level retrieve()
# ---------------------------------------------------------------------------

_default_retriever: Optional[EmbeddingRetriever] = None
_default_lock = Lock()


def _get_default_retriever() -> EmbeddingRetriever:
    global _default_retriever
    with _default_lock:
        if _default_retriever is None:
            r = EmbeddingRetriever()
            r.load_index()
            _default_retriever = r
        return _default_retriever


def retrieve(query: str, top_k: int = 3) -> List[Chunk]:
    """Top-``top_k`` chunks for ``query``.

    Each returned item is a copy of the chunk's metadata
    (``{play, act, scene, speaker, text, source_id, ...}``) with an extra
    ``score`` field containing the cosine similarity (in ``[-1, 1]``).
    Results are sorted from highest to lowest score.

    Lazily loads ``data/processed/embeddings.npy`` and
    ``data/processed/chunks_meta.json`` on first call; subsequent calls reuse
    the loaded index.
    """
    retriever = _get_default_retriever()
    pairs = retriever.retrieve(query, top_k=top_k)
    return [{**chunk, "score": score} for chunk, score in pairs]


# ---------------------------------------------------------------------------
# Manual smoke test:  `python src/retrieval.py "your question"`
# ---------------------------------------------------------------------------

def _cli() -> int:
    import sys
    from chunking import format_chunk_for_display  # local import to avoid cycle
    query = " ".join(sys.argv[1:]) or "Why does Macbeth kill Duncan?"
    print(f"[INFO] query: {query!r}")
    hits = retrieve(query, top_k=3)
    for rank, hit in enumerate(hits, start=1):
        print("=" * 80)
        print(f"Rank {rank} | score={hit['score']:.4f} | source_id={hit.get('source_id')}")
        print(format_chunk_for_display(hit))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
