"""Chunking utilities for the Shakespeare RAG system."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

MAX_TOKENS = 256

Record = Dict[str, Any]
Chunk = Dict[str, Any]

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        from config import EMBEDDING_MODEL_NAME
        _tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    return _tokenizer


def _truncate_to_tokens(text: str) -> str:
    """Truncate to MAX_TOKENS, cutting from the end to preserve the beginning."""
    tokenizer = _get_tokenizer()
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= MAX_TOKENS:
        return text
    return tokenizer.decode(ids[:MAX_TOKENS], skip_special_tokens=True)


def _extract_raw_text(record: Record) -> str:
    for key in ["text", "utterance", "excerpt", "content", "passage"]:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _build_scene_id(record: Record) -> str:
    play_key = record.get("play_key", record.get("play", "unknown")).lower().replace(" ", "_")
    return f"{play_key}_{record.get('act')}_{record.get('scene')}"


def _build_embed_text(record: Record, raw_text: str, speaker: Optional[str]) -> str:
    """Build the text fed to the embedding model.

    Format: {scene_summary} From {play}, Act {act}, Scene {scene}. {speaker} says: {text}
    scene_summary leads so modern-English queries align with archaic dialogue.
    Truncation applies to the full assembled string, preserving the front.
    """
    summary = record.get("scene_summary", "").strip()
    play = record.get("play", "")
    act = record.get("act", "")
    scene = record.get("scene", "")
    location = f"From {play}, Act {act}, Scene {scene}."
    attribution = f"{speaker} says:" if speaker else ""
    parts = [p for p in [summary, location, attribution, raw_text] if p]
    return _truncate_to_tokens(" ".join(parts))


def create_chunks(records: List[Record]) -> List[Chunk]:
    """Convert speaker-turn records to chunk dicts.

    No filtering — all filtering decisions belong in build_index.py.
    Each chunk carries both 'text' (shown to user) and 'embed_text' (fed to MiniLM).
    """
    chunks: List[Chunk] = []
    for i, record in enumerate(records):
        raw_text = _extract_raw_text(record)
        if not raw_text:
            continue
        speaker: Optional[str] = record.get("speaker") or None
        chunk_id = (
            record.get("turn_id")
            or record.get("source_id")
            or record.get("id")
            or f"chunk_{i:06d}"
        )
        chunks.append({
            "chunk_id":   chunk_id,
            "source_id":  chunk_id,
            "play":       record.get("play", record.get("play_key", "unknown")),
            "act":        record.get("act"),
            "scene":      record.get("scene"),
            "scene_id":   _build_scene_id(record),
            "speaker":    speaker,
            "text":       raw_text,
            "embed_text": _build_embed_text(record, raw_text, speaker),
        })
    return chunks


def write_chunks_jsonl(chunks: List[Chunk], path: Path) -> None:
    """Write all chunks to a single JSONL file, one chunk per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def format_chunk_for_display(chunk: Chunk) -> str:
    play = chunk.get("play", "Unknown play")
    act = chunk.get("act", "?")
    scene = chunk.get("scene", "?")
    speaker = chunk.get("speaker", "")
    header = f"{play}, Act {act}, Scene {scene}"
    if speaker:
        header += f" | {speaker}"
    return f"[{header}]\n{chunk.get('text', '')}"
