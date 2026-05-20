"""Build and persist the retrieval index.

Run once before starting the chatbot:
  python3 src/build_index.py

Outputs in data/processed/:
  chunks.jsonl        — all speaker-turn chunks (unfiltered, for reference)
  embeddings.npy      — float32 embedding matrix for indexable chunks
  chunks_meta.json    — metadata for indexed chunks (parallel to embeddings.npy)
  scene_lookup.json   — scene_id → full scene text, used for Parent Document Retrieval
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from config import DATA_DIR, DEFAULT_TOP_K, EMBEDDING_MODEL_NAME
from data_loader import load_all_plays
from chunking import (
    create_chunks,
    format_chunk_for_display,
    write_chunks_jsonl,
    _truncate_to_tokens,
)
from retrieval import EmbeddingRetriever

CHUNKS_PATH      = DATA_DIR / "chunks.jsonl"
EMBEDDINGS_PATH  = DATA_DIR / "embeddings.npy"
CHUNKS_META_PATH = DATA_DIR / "chunks_meta.json"
SCENE_LOOKUP_PATH = DATA_DIR / "scene_lookup.json"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

Chunk = Dict[str, Any]


def _is_indexable(chunk: Chunk) -> bool:
    """Return False for stage directions and chunks too short to embed meaningfully."""
    if chunk.get("speaker") == "STAGE_DIRECTION":
        return False
    return len(chunk.get("text", "").split()) >= 4


def _neighbor_embed_text(
    chunks: List[Chunk],
    original_embeds: List[str],
    idx: int,
    scene_id: str,
) -> str:
    if 0 <= idx < len(chunks) and chunks[idx]["scene_id"] == scene_id:
        return original_embeds[idx]
    return ""


def _add_sliding_window(chunks: List[Chunk]) -> List[Chunk]:
    """Enrich each chunk's embed_text with its same-scene neighbors.

    Uses a snapshot of embed_texts taken before enrichment so neighbors
    always carry their original content, not each other's window context.
    Truncation is re-applied after enrichment to stay within 256 tokens.
    """
    original_embeds = [c["embed_text"] for c in chunks]
    for i, chunk in enumerate(chunks):
        sid = chunk["scene_id"]
        prev = _neighbor_embed_text(chunks, original_embeds, i - 1, sid)
        nxt  = _neighbor_embed_text(chunks, original_embeds, i + 1, sid)
        parts = []
        if prev:
            parts.append(f"[Prev: {prev}]")
        parts.append(chunk["embed_text"])
        if nxt:
            parts.append(f"[Next: {nxt}]")
        chunk["embed_text"] = _truncate_to_tokens(" ".join(parts))
    return chunks


def _build_scene_lookup() -> Dict[str, str]:
    """Map scene_id → full scene text from the raw play JSON files."""
    lookup: Dict[str, str] = {}
    for play_path in sorted(RAW_DIR.glob("*.json")):
        play_key = play_path.stem
        data = json.loads(play_path.read_text(encoding="utf-8"))
        for scene in data.get("scenes", []):
            sid = f"{play_key}_{scene['act']}_{scene['scene']}"
            lookup[sid] = scene.get("text", "")
    return lookup


def build_and_save() -> EmbeddingRetriever:
    records = load_all_plays()
    all_chunks = create_chunks(records)
    print(f"Created {len(all_chunks)} chunks from {len(records)} records.")

    write_chunks_jsonl(all_chunks, CHUNKS_PATH)
    print(f"Saved all chunks → {CHUNKS_PATH}")

    to_index = [c for c in all_chunks if _is_indexable(c)]
    n_filtered = len(all_chunks) - len(to_index)
    print(f"Filtered {n_filtered} non-indexable chunks; indexing {len(to_index)}.")

    to_index = _add_sliding_window(to_index)
    print("Applied sliding window context.")

    scene_lookup = _build_scene_lookup()
    SCENE_LOOKUP_PATH.write_text(
        json.dumps(scene_lookup, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved scene lookup → {SCENE_LOOKUP_PATH} ({len(scene_lookup)} scenes)")

    retriever = EmbeddingRetriever(EMBEDDING_MODEL_NAME)
    retriever.build_index(to_index)

    np.save(EMBEDDINGS_PATH, retriever.embeddings.astype(np.float32))
    with CHUNKS_META_PATH.open("w", encoding="utf-8") as f:
        json.dump(retriever.chunks, f, ensure_ascii=False)
    print(f"Saved embeddings → {EMBEDDINGS_PATH}")
    print(f"Saved chunk metadata → {CHUNKS_META_PATH}")

    return retriever


def main() -> None:
    retriever = build_and_save()

    query = "Why does Macbeth kill Duncan?"
    results = retriever.retrieve(query, top_k=DEFAULT_TOP_K)
    print(f"\nSmoke test — Query: {query}\n")
    for rank, (chunk, score) in enumerate(results, start=1):
        print("=" * 80)
        print(f"Rank {rank} | Score: {score:.4f}")
        print(format_chunk_for_display(chunk))


if __name__ == "__main__":
    main()
