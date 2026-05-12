"""
Chunking utilities.

Students should implement and justify a chunking strategy.
This starter file provides a simple default: one input record becomes one retrieval chunk.
"""

from __future__ import annotations

from typing import Any, Dict, List


Record = Dict[str, Any]
Chunk = Dict[str, Any]


def _get_text(record: Record) -> str:
    """
    Extract text from a record using common field names.
    Adapt this function if your dataset uses different names.
    """
    for key in ["text", "utterance", "excerpt", "content", "passage"]:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    # Fallback: combine selected fields if no obvious text field exists.
    parts = []
    for key in ["speaker", "summary", "modern_summary"]:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())

    return " ".join(parts).strip()


def create_chunks(records: List[Record]) -> List[Chunk]:
    """
    Convert structured records into retrieval chunks.

    This is intentionally simple. Students should consider whether a different strategy
    is better, such as:
    - scene-level chunks;
    - speaker-turn chunks;
    - overlapping fixed-size chunks;
    - summary-enhanced chunks.
    """
    chunks: List[Chunk] = []

    for i, record in enumerate(records):
        text = _get_text(record)
        if not text:
            continue

        chunk = {
            "chunk_id": record.get("source_id") or record.get("id") or f"chunk_{i:06d}",
            "play": record.get("play", record.get("play_key", "unknown")),
            "act": record.get("act", None),
            "scene": record.get("scene", None),
            "speaker": record.get("speaker", None),
            "text": text,
            "metadata": record,
        }
        chunks.append(chunk)

    return chunks


def format_chunk_for_display(chunk: Chunk) -> str:
    """
    Format a retrieved chunk for display to the user.
    """
    play = chunk.get("play", "Unknown play")
    act = chunk.get("act", "?")
    scene = chunk.get("scene", "?")
    speaker = chunk.get("speaker", "")

    header = f"{play}, Act {act}, Scene {scene}"
    if speaker:
        header += f", Speaker: {speaker}"

    return f"[{header}]\n{chunk.get('text', '')}"
