"""
Verification script for data/processed/chunks.jsonl.

Runs a battery of automated checks against the chunked output and the raw
utterance files in dataset/, then prints sample chunks for manual inspection.

Usage:
    python src/verify_chunks.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from config import DATA_DIR, PROJECT_ROOT


REQUIRED_FIELDS = ("play", "act", "scene", "speaker", "text", "source_id")
STAGE_DIRECTION_SPEAKER = "STAGE_DIRECTION"

CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
DATASET_DIR = PROJECT_ROOT / "dataset"

# play key (used by chunker) -> raw utterance file
RAW_UTTERANCE_FILES = {
    "Hamlet": DATASET_DIR / "hamlet_utterances.jsonl",
    "Macbeth": DATASET_DIR / "macbeth_utterances.jsonl",
    "Romeo and Juliet": DATASET_DIR / "romeo_and_juliet_utterances.jsonl",
}


# ---------- pretty printing ----------

PASS = "[ OK ]"
FAIL = "[FAIL]"
INFO = "[INFO]"


def _hr(title: str = "") -> None:
    if title:
        print(f"\n=== {title} ===")
    else:
        print("-" * 60)


# ---------- loaders ----------

def load_chunks(path: Path) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"chunks.jsonl line {lineno} is not valid JSON: {e}")
    return chunks


def load_raw_utterances(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ---------- checks ----------

def check_file_exists(path: Path) -> bool:
    if not path.exists():
        print(f"{FAIL} {path} does not exist. Did you run `python src/chunking.py`?")
        return False
    print(f"{PASS} file exists: {path} ({path.stat().st_size:,} bytes)")
    return True


def check_required_fields(chunks: List[Dict[str, Any]]) -> bool:
    missing_rows = []
    empty_text_rows = []
    for i, c in enumerate(chunks):
        missing = [k for k in REQUIRED_FIELDS if k not in c]
        if missing:
            missing_rows.append((i, missing))
        text = c.get("text")
        if not isinstance(text, str) or not text.strip():
            empty_text_rows.append(i)

    ok = not missing_rows and not empty_text_rows
    if missing_rows:
        print(f"{FAIL} {len(missing_rows)} rows missing required fields. First: {missing_rows[:3]}")
    else:
        print(f"{PASS} every row has all required fields {REQUIRED_FIELDS}")
    if empty_text_rows:
        print(f"{FAIL} {len(empty_text_rows)} rows have empty text. First indices: {empty_text_rows[:5]}")
    else:
        print(f"{PASS} no rows have empty text")
    return ok


def check_speaker_turn_invariant(chunks: List[Dict[str, Any]]) -> bool:
    """No two adjacent chunks in the same (play, act, scene) share a speaker."""
    violations = []
    prev_key = None
    prev_speaker = None
    for i, c in enumerate(chunks):
        key = (c.get("play"), c.get("act"), c.get("scene"))
        spk = c.get("speaker")
        if key == prev_key and spk == prev_speaker and spk is not None:
            violations.append((i, key, spk))
        prev_key = key
        prev_speaker = spk

    if violations:
        print(f"{FAIL} {len(violations)} adjacent chunks share a speaker within the same scene")
        for v in violations[:3]:
            print(f"       at index {v[0]} scene={v[1]} speaker={v[2]}")
        return False
    print(f"{PASS} no two adjacent chunks in the same scene share a speaker "
          "(speaker-turn merging is working)")
    return True


def check_source_id_format(chunks: List[Dict[str, Any]]) -> bool:
    """source_id should be present and unique enough to point back to the dataset."""
    sids = [c.get("source_id") for c in chunks]
    seen = Counter(sids)
    dups = [s for s, n in seen.items() if n > 1 and s is not None]
    if dups:
        print(f"{FAIL} {len(dups)} duplicated source_ids. Sample: {dups[:3]}")
        return False
    print(f"{PASS} all {len(sids)} source_ids are unique")
    return True


def check_utterance_accounting(chunks: List[Dict[str, Any]]) -> bool:
    """Total utterance_count across chunks should equal raw utterance counts."""
    ok = True
    chunks_by_play = defaultdict(list)
    for c in chunks:
        chunks_by_play[c.get("play")].append(c)

    for play, path in RAW_UTTERANCE_FILES.items():
        if not path.exists():
            print(f"{INFO} skipping accounting for {play}: {path} not found")
            continue
        raw = load_raw_utterances(path)
        raw_total = len(raw)
        chunked_total = sum(c.get("utterance_count", 0) for c in chunks_by_play.get(play, []))
        # utterance_count==0 happens for scene-only fallback; allow >= raw or equal
        status = PASS if chunked_total == raw_total else FAIL
        if chunked_total != raw_total:
            ok = False
        print(f"{status} {play}: {chunked_total} utterances accounted for in chunks "
              f"vs {raw_total} in raw utterance file")
    return ok


def check_per_play_counts(chunks: List[Dict[str, Any]]) -> None:
    by_play = Counter(c.get("play") for c in chunks)
    print(f"{INFO} chunk counts per play:")
    for play, n in by_play.most_common():
        print(f"       {play}: {n}")
    stage = sum(1 for c in chunks if c.get("speaker") == STAGE_DIRECTION_SPEAKER)
    print(f"{INFO} stage-direction chunks: {stage}")


def text_length_stats(chunks: List[Dict[str, Any]]) -> None:
    lens = sorted(len(c["text"]) for c in chunks)
    if not lens:
        return
    n = len(lens)

    def pct(p):
        return lens[min(n - 1, int(p * n))]
    print(f"{INFO} text length (chars): "
          f"min={lens[0]} p25={pct(0.25)} p50={pct(0.5)} "
          f"p75={pct(0.75)} p95={pct(0.95)} max={lens[-1]}")


def print_samples(chunks: List[Dict[str, Any]], n: int = 3) -> None:
    """Print a few interesting chunks: longest, a multi-utterance turn, a random one."""
    _hr("Sample chunks")

    longest = max(chunks, key=lambda c: len(c["text"]))
    multi = next(
        (c for c in chunks
         if c.get("utterance_count", 0) > 1 and c.get("speaker") != STAGE_DIRECTION_SPEAKER),
        None,
    )
    rng = random.Random(0)
    rand_chunks = rng.sample(chunks, min(n, len(chunks)))

    print("\n-- Longest chunk (likely a soliloquy) --")
    _print_chunk(longest, truncate=400)

    if multi:
        print("\n-- A multi-utterance speaker turn --")
        _print_chunk(multi)

    for i, c in enumerate(rand_chunks, 1):
        print(f"\n-- Random sample {i} --")
        _print_chunk(c)


def _print_chunk(c: Dict[str, Any], truncate: int = 300) -> None:
    meta_keys = ("play", "act", "scene", "speaker", "source_id")
    meta = {k: c.get(k) for k in meta_keys}
    print(json.dumps(meta, ensure_ascii=False))
    text = c.get("text", "")
    if len(text) > truncate:
        text = text[:truncate] + f"... [{len(c['text'])} chars total]"
    print(f"  text: {text}")
    extras = {k: c[k] for k in ("chunk_id", "source_ids", "utterance_count") if k in c}
    if extras:
        print(f"  extras: {extras}")


def spot_check_against_raw(chunks: List[Dict[str, Any]]) -> None:
    """Pick one chunk and verify its text == concatenation of the named source_ids in the raw file."""
    _hr("Spot check vs raw utterances")
    target = next(
        (c for c in chunks
         if c.get("utterance_count", 0) > 1
         and c.get("speaker") != STAGE_DIRECTION_SPEAKER),
        None,
    )
    if target is None:
        print(f"{INFO} no suitable multi-utterance chunk found to spot-check")
        return

    path = RAW_UTTERANCE_FILES.get(target["play"])
    if not path or not path.exists():
        print(f"{INFO} raw utterance file for {target['play']} not available; skipping")
        return

    raw_by_sid = {r["source_id"]: r for r in load_raw_utterances(path) if "source_id" in r}
    sids = target.get("source_ids", [])
    pieces = [raw_by_sid[s]["text"].strip() for s in sids if s in raw_by_sid]
    expected = " ".join(pieces)
    ok = expected == target["text"].strip()
    status = PASS if ok else FAIL
    print(f"{status} reconstructed text from raw matches chunk text "
          f"(chunk_id={target.get('chunk_id')}, speaker={target['speaker']}, n={len(sids)})")
    if not ok:
        print(f"       expected: {expected[:200]}...")
        print(f"       got:      {target['text'][:200]}...")


# ---------- main ----------

def main() -> int:
    print(f"Verifying {CHUNKS_PATH}")
    if not check_file_exists(CHUNKS_PATH):
        return 1

    chunks = load_chunks(CHUNKS_PATH)
    print(f"{INFO} loaded {len(chunks)} chunks")

    _hr("Schema checks")
    ok1 = check_required_fields(chunks)
    ok2 = check_source_id_format(chunks)

    _hr("Speaker-turn correctness")
    ok3 = check_speaker_turn_invariant(chunks)

    _hr("Accounting vs raw utterances")
    ok4 = check_utterance_accounting(chunks)

    _hr("Distribution")
    check_per_play_counts(chunks)
    text_length_stats(chunks)

    spot_check_against_raw(chunks)

    print_samples(chunks)

    _hr("Result")
    all_ok = ok1 and ok2 and ok3 and ok4
    print(("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
