"""
Build the embedding index for the Shakespeare-aware RAG system.

Reads
-----
    data/processed/chunks.jsonl       (produced by `python src/chunking.py`)

Writes
------
    data/processed/embeddings.npy     float32 array, shape (N, D), L2-normalised
                                      so cosine similarity == dot product
    data/processed/chunks_meta.json   list of N chunk-metadata dicts in the
                                      same order as the rows of embeddings.npy

Retrieval-quality decisions baked into this script
--------------------------------------------------
1. Stage directions are *not* indexed. Chunks with ``speaker == "STAGE_DIRECTION"``
   are scene-setting prose ("Enter Macbeth.", "Exeunt.") that should never be
   retrieved as an answer to a content question. They remain in chunks.jsonl
   for traceability; they are only excluded from the searchable index.
2. Very short utterances are *not* indexed either. Chunks whose raw text has
   fewer than ``MIN_WORDS_TO_INDEX`` words (default 5) are one-line replies
   like "My worthy Cawdor!" or "Where is Duncan's body?" — they carry no
   semantic content but match by keyword overlap and crowd out substantive
   speeches. They remain in chunks.jsonl for traceability.
3. The *embedded* text is prefixed with metadata
   ``"From {play}, Act {a}, Scene {s}. {speaker} says: {text}"`` so that very
   short utterances (medians around 17 tokens, p25 = 10) still carry stable
   contextual signal and don't collapse to bare keyword matches. The displayed
   text stored in ``chunks_meta.json`` stays the original quote, so retrieval
   results show the raw line, not the prefixed string.

Usage
-----
    python src/build_index.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from config import DATA_DIR, EMBEDDING_MODEL_NAME


CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
META_PATH = DATA_DIR / "chunks_meta.json"

# Speaker label used by the dataset for stage directions. Kept in sync with
# src/chunking.py.
STAGE_DIRECTION_SPEAKER = "STAGE_DIRECTION"

# Drop chunks whose raw text has fewer than this many whitespace-separated
# words. Tuned to remove one-line replies that match by keyword overlap but
# carry no usable answer text (e.g. "My worthy Cawdor!", "Where is Duncan's
# body?"). The diagnostic in results/scene_length_diagnostic.json shows that
# speaker-turn chunks have p25 = 10 tokens, so cutting below 5 words removes
# the noisiest tail without touching the medium-length turns.
MIN_WORDS_TO_INDEX = 5


def _word_count(text: str) -> int:
    return len(text.split()) if isinstance(text, str) else 0


def _is_searchable(chunk: Dict[str, Any]) -> bool:
    """Return True if the chunk should be embedded into the retrieval index."""
    if chunk.get("speaker") == STAGE_DIRECTION_SPEAKER:
        return False
    text = chunk.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return False
    if _word_count(text) < MIN_WORDS_TO_INDEX:
        return False
    return True


def _to_embedding_text(chunk: Dict[str, Any]) -> str:
    """Construct the string that is actually fed to the embedding model.

    The retrieval problem with raw text is that very short character lines
    (e.g. a chunk whose entire content is ``"Macbeth."``) embed as pure
    keyword matches and dominate any query that mentions the keyword.
    Prefixing with play / act / scene / speaker injects stable contextual
    signal so short chunks stay useful, while leaving long chunks effectively
    unchanged (a 15-token header is small next to a soliloquy).
    """
    play = chunk.get("play") or "Unknown play"
    speaker = chunk.get("speaker") or "UNKNOWN"
    act = chunk.get("act")
    scene = chunk.get("scene")
    text = chunk.get("text", "")

    if act is not None and scene is not None:
        location = f"Act {act}, Scene {scene}"
    elif act is not None:
        location = f"Act {act}"
    else:
        location = ""

    if location:
        header = f"From {play}, {location}. {speaker} says: "
    else:
        header = f"From {play}. {speaker} says: "
    return header + text


def load_chunks(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/chunking.py` first to "
            f"build the speaker-turn chunks."
        )
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno} is not valid JSON: {e}")
    return out


def build_index(batch_size: int = 64, show_progress: bool = True) -> None:
    all_chunks = load_chunks(CHUNKS_PATH)
    if not all_chunks:
        raise RuntimeError(f"{CHUNKS_PATH} is empty.")

    # Filter out stage directions, empty text, and chunks that are too short
    # to embed meaningfully.
    chunks: List[Dict[str, Any]] = [c for c in all_chunks if _is_searchable(c)]
    n_stage = sum(1 for c in all_chunks if c.get("speaker") == STAGE_DIRECTION_SPEAKER)
    n_short = sum(
        1 for c in all_chunks
        if c.get("speaker") != STAGE_DIRECTION_SPEAKER
        and isinstance(c.get("text"), str)
        and c.get("text", "").strip()
        and _word_count(c["text"]) < MIN_WORDS_TO_INDEX
    )
    n_empty = (len(all_chunks) - len(chunks)) - n_stage - n_short
    print(f"[INFO] loaded {len(all_chunks):,} chunks from {CHUNKS_PATH}")
    print(f"[INFO] dropped {len(all_chunks) - len(chunks)} non-searchable chunks "
          f"({n_stage} stage directions, {n_short} too short (< {MIN_WORDS_TO_INDEX} words), "
          f"{n_empty} empty-text); indexing {len(chunks):,}.")

    embed_texts = [_to_embedding_text(c) for c in chunks]

    # Show what the prefix looks like in practice (first non-stage-direction
    # chunk) so the user can sanity-check it in the terminal.
    print(f"[INFO] sample embedded text: {embed_texts[0]!r}")

    print(f"[INFO] loading embedding model: {EMBEDDING_MODEL_NAME}")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise SystemExit(
            "sentence-transformers is required. Install with "
            "`pip install -r requirements.txt`."
        ) from e
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print(f"[INFO] encoding {len(embed_texts):,} chunks (batch_size={batch_size}) ...")
    t0 = time.time()
    embeddings = model.encode(
        embed_texts,
        batch_size=batch_size,
        normalize_embeddings=True,   # cosine == dot product downstream
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    ).astype("float32")
    dt = time.time() - t0
    print(f"[INFO] encoded in {dt:.1f}s — embeddings shape = {embeddings.shape}, dtype = {embeddings.dtype}")

    # Sanity: every row should now be unit-norm (within float32 tolerance).
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        bad = int((np.abs(norms - 1.0) > 1e-3).sum())
        print(f"[WARN] {bad} embeddings are not unit-norm; re-normalising.", file=sys.stderr)
        embeddings = embeddings / np.maximum(norms[:, None], 1e-12)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    print(f"[INFO] wrote {EMBEDDINGS_PATH}")
    print(f"[INFO] wrote {META_PATH}")

    # Quick sanity print
    sample = chunks[0]
    keys = ("play", "act", "scene", "speaker", "source_id")
    sample_meta = {k: sample.get(k) for k in keys}
    print(f"[INFO] chunks_meta.json[0] = {sample_meta}")
    print(f"[INFO] embeddings.npy[0][:6] = {embeddings[0][:6].tolist()}")
    print("[DONE] index built.")


if __name__ == "__main__":
    build_index()
