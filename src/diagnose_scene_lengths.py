"""
Diagnostic: token-length distribution of Shakespeare chunks under the MiniLM
tokenizer.

Why this script exists
----------------------
Before committing to a chunking strategy, we want to see whether the natural
scene-level units actually fit inside the embedding model's context window.
The retriever uses ``sentence-transformers/all-MiniLM-L6-v2``, whose
``max_seq_length`` is 256 tokens; anything longer gets silently truncated,
which silently degrades retrieval recall.

This script tokenizes:

1. Scene-level chunks (one chunk per scene, from
   ``dataset/*_scene_chunks.jsonl``), and
2. Speaker-turn chunks (the output of ``src/chunking.py``,
   ``data/processed/chunks.jsonl``)

with the MiniLM tokenizer and reports the per-play and overall token-length
distribution, plus the number of chunks that exceed 256 tokens. The
side-by-side comparison is what justifies switching from scene-level to
speaker-turn chunking.

Usage
-----
    python src/diagnose_scene_lengths.py

Writes a JSON summary to ``results/scene_length_diagnostic.json``.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from config import DATA_DIR, EMBEDDING_MODEL_NAME, PROJECT_ROOT, RESULTS_DIR


DATASET_DIR = PROJECT_ROOT / "dataset"

SCENE_CHUNK_FILES = {
    "Hamlet": DATASET_DIR / "hamlet_scene_chunks.jsonl",
    "Macbeth": DATASET_DIR / "macbeth_scene_chunks.jsonl",
    "Romeo and Juliet": DATASET_DIR / "romeo_and_juliet_scene_chunks.jsonl",
}

SPEAKER_TURN_CHUNKS_PATH = DATA_DIR / "chunks.jsonl"

# MiniLM-L6-v2 truncates input at 256 tokens (special tokens included).
MINILM_MAX_SEQ_LEN = 256


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[dict]:
    out: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_scene_chunks() -> Dict[str, List[dict]]:
    by_play: Dict[str, List[dict]] = {}
    for play, path in SCENE_CHUNK_FILES.items():
        if not path.exists():
            print(f"[WARN] scene chunk file missing: {path}", file=sys.stderr)
            continue
        by_play[play] = _load_jsonl(path)
    return by_play


def load_speaker_turn_chunks() -> Dict[str, List[dict]]:
    if not SPEAKER_TURN_CHUNKS_PATH.exists():
        print(
            f"[WARN] {SPEAKER_TURN_CHUNKS_PATH} not found. "
            "Run `python src/chunking.py` first if you want the speaker-turn comparison.",
            file=sys.stderr,
        )
        return {}
    chunks = _load_jsonl(SPEAKER_TURN_CHUNKS_PATH)
    by_play: Dict[str, List[dict]] = defaultdict(list)
    for c in chunks:
        by_play[c.get("play", "unknown")].append(c)
    return dict(by_play)


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def _load_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "The `transformers` package is required for this diagnostic. "
            "Install with `pip install -r requirements.txt`."
        ) from e
    print(f"[INFO] loading tokenizer for {model_name} ...")
    return AutoTokenizer.from_pretrained(model_name)


def _tokenize_lengths(tokenizer, texts: Sequence[str]) -> List[int]:
    """Return token counts (with special tokens) for each text."""
    if not texts:
        return []
    # Use the fast tokenizer's batch API but disable truncation/padding so
    # we measure the *true* token length, not the truncated one.
    encoded = tokenizer(
        list(texts),
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return [len(ids) for ids in encoded["input_ids"]]


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def _percentile(sorted_values: Sequence[int], p: float) -> int:
    if not sorted_values:
        return 0
    if len(sorted_values) == 1:
        return sorted_values[0]
    # Nearest-rank percentile, capped at the last index.
    k = max(0, min(len(sorted_values) - 1, math.ceil(p * len(sorted_values)) - 1))
    return sorted_values[k]


def _summarise(lengths: Sequence[int], threshold: int = MINILM_MAX_SEQ_LEN) -> Dict[str, float]:
    if not lengths:
        return {"count": 0}
    s = sorted(lengths)
    return {
        "count": len(s),
        "min": s[0],
        "p25": _percentile(s, 0.25),
        "p50": _percentile(s, 0.50),
        "mean": round(statistics.mean(s), 1),
        "p75": _percentile(s, 0.75),
        "p90": _percentile(s, 0.90),
        "p95": _percentile(s, 0.95),
        "p99": _percentile(s, 0.99),
        "max": s[-1],
        "stdev": round(statistics.pstdev(s), 1) if len(s) > 1 else 0.0,
        f"over_{threshold}": sum(1 for x in s if x > threshold),
        f"over_{threshold}_pct": round(100 * sum(1 for x in s if x > threshold) / len(s), 1),
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

COLUMNS = (
    ("count", 6),
    ("min", 5),
    ("p25", 5),
    ("p50", 5),
    ("mean", 7),
    ("p75", 5),
    ("p90", 5),
    ("p95", 5),
    ("p99", 5),
    ("max", 6),
    ("stdev", 6),
    (f"over_{MINILM_MAX_SEQ_LEN}", 9),
    (f"over_{MINILM_MAX_SEQ_LEN}_pct", 8),
)


def _print_table(title: str, summaries: Dict[str, Dict[str, float]]) -> None:
    print(f"\n--- {title} ---")
    header = f"{'play':28s}" + " ".join(f"{name:>{w}}" for name, w in COLUMNS)
    print(header)
    print("-" * len(header))
    # Stable row order: per-play sorted, then ALL last
    keys = sorted(k for k in summaries if k != "ALL") + (["ALL"] if "ALL" in summaries else [])
    for k in keys:
        s = summaries[k]
        cells = [f"{k:28s}"]
        for name, w in COLUMNS:
            val = s.get(name, "")
            if isinstance(val, float):
                cells.append(f"{val:>{w}.1f}")
            else:
                cells.append(f"{val:>{w}}")
        print(" ".join(cells))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _diagnose(label: str, by_play: Dict[str, List[dict]], tokenizer) -> Dict[str, Dict[str, float]]:
    summaries: Dict[str, Dict[str, float]] = {}
    all_lengths: List[int] = []
    for play, records in by_play.items():
        texts = [r.get("text", "") for r in records if isinstance(r.get("text"), str)]
        lengths = _tokenize_lengths(tokenizer, texts)
        summaries[play] = _summarise(lengths)
        all_lengths.extend(lengths)
    if all_lengths:
        summaries["ALL"] = _summarise(all_lengths)
    _print_table(label, summaries)
    return summaries


def main() -> int:
    tokenizer = _load_tokenizer(EMBEDDING_MODEL_NAME)
    print(f"[INFO] MiniLM max_seq_length used as overflow threshold = {MINILM_MAX_SEQ_LEN}")

    print("\n[INFO] loading scene-level chunks ...")
    scenes_by_play = load_scene_chunks()
    print("[INFO] loading speaker-turn chunks (data/processed/chunks.jsonl) ...")
    turns_by_play = load_speaker_turn_chunks()

    scene_summary = _diagnose(
        "Scene-level chunks (dataset/*_scene_chunks.jsonl)",
        scenes_by_play,
        tokenizer,
    )

    turn_summary: Dict[str, Dict[str, float]] = {}
    if turns_by_play:
        turn_summary = _diagnose(
            "Speaker-turn chunks (data/processed/chunks.jsonl)",
            turns_by_play,
            tokenizer,
        )

    # Headline comparison
    print("\n--- Headline ---")
    if "ALL" in scene_summary:
        s = scene_summary["ALL"]
        print(
            f"Scene-level   : n={s['count']:>4}  median={s['p50']:>4}  p95={s['p95']:>4}  "
            f"max={s['max']:>4}  over {MINILM_MAX_SEQ_LEN} tok = {s[f'over_{MINILM_MAX_SEQ_LEN}']} "
            f"({s[f'over_{MINILM_MAX_SEQ_LEN}_pct']}%)"
        )
    if "ALL" in turn_summary:
        s = turn_summary["ALL"]
        print(
            f"Speaker-turn  : n={s['count']:>4}  median={s['p50']:>4}  p95={s['p95']:>4}  "
            f"max={s['max']:>4}  over {MINILM_MAX_SEQ_LEN} tok = {s[f'over_{MINILM_MAX_SEQ_LEN}']} "
            f"({s[f'over_{MINILM_MAX_SEQ_LEN}_pct']}%)"
        )

    # Persist for the report
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "scene_length_diagnostic.json"
    payload = {
        "model": EMBEDDING_MODEL_NAME,
        "max_seq_length": MINILM_MAX_SEQ_LEN,
        "scene_level": scene_summary,
        "speaker_turn": turn_summary,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
