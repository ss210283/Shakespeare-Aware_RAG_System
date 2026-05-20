"""
Chunking utilities for the Shakespeare-aware RAG system.

Chunking strategy: SPEAKER TURN.

Justification (data-driven)
---------------------------
The retriever uses ``sentence-transformers/all-MiniLM-L6-v2``, whose
``max_seq_length`` is 256 tokens; anything longer is silently truncated and
its content does not contribute to the embedding. We therefore measured the
token-length distribution of two candidate chunking units with the MiniLM
tokenizer (see ``src/diagnose_scene_lengths.py``,
``results/scene_length_diagnostic.json``):

  Scene-level chunks   (n=73):    median 1117, p95 3977, max 7497,
                                  70 of 73 scenes (95.9%) exceed 256 tokens.
  Speaker-turn chunks  (n=2993):  median   17, p95  126, max  675,
                                   46 of 2993 turns (1.5%) exceed 256 tokens.

Scene-level chunking is unusable as-is: almost every scene would be truncated
by the embedder. Per-utterance chunks (the other extreme) avoid truncation
but most utterances are too short to embed meaningfully (one-line replies
like "Ay." or "My lord?") and they fragment a single character's continuous
speech across many tiny chunks.

The speaker-turn strategy is the middle ground: consecutive utterances by
the same speaker inside a scene are concatenated into one chunk. This keeps
the semantic unit a human reader would treat as "one thing this character
said", lines up with character- or motive-specific questions, and ~98.5% of
the resulting chunks fit inside the MiniLM context window without truncation.

Stage directions (speaker == "STAGE_DIRECTION") are kept as their own chunks
so that scene-setting context is not lost, but they are not merged into the
surrounding character's speech.

Output
------
`python src/chunking.py` reads the three play files listed in
`config.PLAY_FILES` and writes one JSON record per line to
`data/processed/chunks.jsonl`. Each record contains the metadata required by
the assignment: ``{play, act, scene, speaker, text, source_id}`` plus a few
extra fields (``chunk_id``, ``source_ids``, ``utterance_count``) that are
useful for downstream debugging but can be ignored by the retriever.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from config import DATA_DIR, PLAY_FILES


Record = Dict[str, Any]
Chunk = Dict[str, Any]

# Speaker label used by the dataset for stage directions / scene markers.
STAGE_DIRECTION_SPEAKER = "STAGE_DIRECTION"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_text(record: Record) -> str:
    """Extract a text field from a record using common field names."""
    for key in ["text", "utterance", "excerpt", "content", "passage"]:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    parts: List[str] = []
    for key in ["speaker", "summary", "modern_summary"]:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return " ".join(parts).strip()


def _iter_scenes(play_obj: Any) -> Iterable[Record]:
    """Yield scene records from a loaded play JSON object."""
    if isinstance(play_obj, dict) and isinstance(play_obj.get("scenes"), list):
        yield from play_obj["scenes"]
    elif isinstance(play_obj, list):
        yield from play_obj
    else:
        raise ValueError(
            "Unexpected play JSON shape: expected a dict with a 'scenes' list "
            "or a list of scene records."
        )


def _load_play_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find dataset file: {path}\n"
            "Place the provided play JSON files in data/processed/."
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalise_speaker(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


# ---------------------------------------------------------------------------
# Speaker-turn chunking
# ---------------------------------------------------------------------------

def _flush_turn(
    buffer: List[Record],
    chunks: List[Chunk],
    scene: Record,
    play_key: Optional[str],
) -> None:
    """Materialise the buffered utterances into a single speaker-turn chunk."""
    if not buffer:
        return

    speaker = _normalise_speaker(buffer[0].get("speaker")) or "UNKNOWN"
    texts = [_get_text(u) for u in buffer]
    texts = [t for t in texts if t]
    if not texts:
        return
    text = " ".join(texts)

    source_ids = [
        u.get("source_id") or u.get("utterance_id")
        for u in buffer
        if u.get("source_id") or u.get("utterance_id")
    ]
    primary_source_id = source_ids[0] if source_ids else None

    if len(source_ids) > 1 and primary_source_id:
        # e.g. macbeth_1_3_0014..0016 - preserves a deterministic chunk id
        chunk_id = f"{primary_source_id}..{source_ids[-1].split('_')[-1]}"
    else:
        chunk_id = primary_source_id or (
            f"{scene.get('scene_id', 'scene')}_{speaker}_{len(chunks):05d}"
        )

    chunk: Chunk = {
        "chunk_id": chunk_id,
        "play": scene.get("play") or buffer[0].get("play") or play_key or "unknown",
        "act": scene.get("act", buffer[0].get("act")),
        "scene": scene.get("scene", buffer[0].get("scene")),
        "speaker": speaker,
        "text": text,
        "source_id": primary_source_id or chunk_id,
        "source_ids": source_ids,
        "utterance_count": len(buffer),
    }
    chunks.append(chunk)


def create_speaker_turn_chunks(
    scenes: Iterable[Record],
    play_key: Optional[str] = None,
    include_stage_directions: bool = True,
) -> List[Chunk]:
    """Group consecutive utterances by the same speaker into one chunk.

    Parameters
    ----------
    scenes:
        Iterable of scene records, each expected to have an ``utterances`` list.
    play_key:
        Fallback play identifier if a scene record does not have a ``play`` field.
    include_stage_directions:
        If True (default), stage directions are emitted as their own chunks
        (one per run of consecutive stage directions). If False, they are
        skipped entirely.
    """
    chunks: List[Chunk] = []

    for scene in scenes:
        utterances = scene.get("utterances") if isinstance(scene, dict) else None
        if not isinstance(utterances, list):
            # Scene record without an utterances list — fall back to emitting
            # the scene's full text as a single chunk so we don't silently drop it.
            text = _get_text(scene)
            if text:
                chunks.append({
                    "chunk_id": scene.get("scene_id") or f"scene_{len(chunks):05d}",
                    "play": scene.get("play") or play_key or "unknown",
                    "act": scene.get("act"),
                    "scene": scene.get("scene"),
                    "speaker": None,
                    "text": text,
                    "source_id": scene.get("scene_id") or f"scene_{len(chunks):05d}",
                    "source_ids": [scene.get("scene_id")] if scene.get("scene_id") else [],
                    "utterance_count": 0,
                })
            continue

        buffer: List[Record] = []
        current_speaker: Optional[str] = None

        for utt in utterances:
            speaker = _normalise_speaker(utt.get("speaker"))
            if not include_stage_directions and speaker == STAGE_DIRECTION_SPEAKER:
                _flush_turn(buffer, chunks, scene, play_key)
                buffer = []
                current_speaker = None
                continue

            if speaker != current_speaker and buffer:
                _flush_turn(buffer, chunks, scene, play_key)
                buffer = []

            buffer.append(utt)
            current_speaker = speaker

        _flush_turn(buffer, chunks, scene, play_key)

    return chunks


# ---------------------------------------------------------------------------
# Public API: dispatch / backwards-compatible helpers
# ---------------------------------------------------------------------------

def create_chunks(records: List[Record], strategy: str = "speaker_turn") -> List[Chunk]:
    """Convert structured records into retrieval chunks.

    The default strategy is ``"speaker_turn"`` (see module docstring). The
    legacy ``"record"`` strategy emits one chunk per input record and is kept
    for backwards compatibility with the starter ``build_index.py``.
    """
    if strategy == "speaker_turn":
        return create_speaker_turn_chunks(records)

    if strategy == "record":
        chunks: List[Chunk] = []
        for i, record in enumerate(records):
            text = _get_text(record)
            if not text:
                continue
            chunks.append({
                "chunk_id": record.get("source_id") or record.get("id") or f"chunk_{i:06d}",
                "play": record.get("play", record.get("play_key", "unknown")),
                "act": record.get("act"),
                "scene": record.get("scene"),
                "speaker": record.get("speaker"),
                "text": text,
                "source_id": record.get("source_id") or record.get("id") or f"chunk_{i:06d}",
                "metadata": record,
            })
        return chunks

    raise ValueError(f"Unknown chunking strategy: {strategy!r}")


def format_chunk_for_display(chunk: Chunk) -> str:
    """Format a retrieved chunk for display to the user."""
    play = chunk.get("play", "Unknown play")
    act = chunk.get("act", "?")
    scene = chunk.get("scene", "?")
    speaker = chunk.get("speaker", "")

    header = f"{play}, Act {act}, Scene {scene}"
    if speaker:
        header += f", Speaker: {speaker}"
    return f"[{header}]\n{chunk.get('text', '')}"


# ---------------------------------------------------------------------------
# Driver: build data/processed/chunks.jsonl
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = ("play", "act", "scene", "speaker", "text", "source_id")


def _project_chunk(chunk: Chunk) -> Chunk:
    """Return a chunk with the required metadata fields first, then extras."""
    out: Chunk = {k: chunk.get(k) for k in REQUIRED_FIELDS}
    for k in ("chunk_id", "source_ids", "utterance_count"):
        if k in chunk:
            out[k] = chunk[k]
    return out


def write_chunks_jsonl(chunks: List[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(_project_chunk(chunk), ensure_ascii=False))
            f.write("\n")


def build_chunks_for_all_plays(
    include_stage_directions: bool = True,
) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for play_key, path in PLAY_FILES.items():
        play_obj = _load_play_json(path)
        scenes = list(_iter_scenes(play_obj))
        chunks = create_speaker_turn_chunks(
            scenes,
            play_key=play_key,
            include_stage_directions=include_stage_directions,
        )
        all_chunks.extend(chunks)
    return all_chunks


def main() -> None:
    chunks = build_chunks_for_all_plays(include_stage_directions=True)
    out_path = DATA_DIR / "chunks.jsonl"
    write_chunks_jsonl(chunks, out_path)

    # Simple summary so the user knows what was produced.
    by_play: Dict[str, int] = {}
    stage_direction_count = 0
    for c in chunks:
        by_play[c.get("play", "unknown")] = by_play.get(c.get("play", "unknown"), 0) + 1
        if c.get("speaker") == STAGE_DIRECTION_SPEAKER:
            stage_direction_count += 1

    print(f"Wrote {len(chunks)} speaker-turn chunks to {out_path}")
    for play, n in by_play.items():
        print(f"  {play}: {n} chunks")
    print(f"  (of which stage-direction chunks: {stage_direction_count})")


if __name__ == "__main__":
    main()
