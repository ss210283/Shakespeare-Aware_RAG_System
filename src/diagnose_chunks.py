"""
Diagnostic: token length distribution across chunking granularities.

all-MiniLM-L6-v2 silently truncates input beyond 256 tokens.
This script compares scene-level vs utterance-level chunks to inform
the choice of chunking strategy for the retrieval index.
"""

from pathlib import Path
from typing import Dict, List
import json
import statistics

from transformers import AutoTokenizer

DATASET_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MAX_TOKENS = 256  # Hard input limit for all-MiniLM-L6-v2

# Each entry is a chunk type label mapped to its per-play JSONL files
CHUNK_FILE_SETS: Dict[str, Dict[str, Path]] = {
    "Scene-level": {
        "Hamlet": DATASET_DIR / "hamlet_scene_chunks.jsonl",
        "Macbeth": DATASET_DIR / "macbeth_scene_chunks.jsonl",
        "Romeo and Juliet": DATASET_DIR / "romeo_and_juliet_scene_chunks.jsonl",
    },
    "Utterance-level": {
        "Hamlet": DATASET_DIR / "hamlet_utterances.jsonl",
        "Macbeth": DATASET_DIR / "macbeth_utterances.jsonl",
        "Romeo and Juliet": DATASET_DIR / "romeo_and_juliet_utterances.jsonl",
    },
    "Speaker-turn-level": {
        "Hamlet": DATASET_DIR / "hamlet_speaker_turn_chunks.jsonl",
        "Macbeth": DATASET_DIR / "macbeth_speaker_turn_chunks.jsonl",
        "Romeo and Juliet": DATASET_DIR / "romeo_and_juliet_speaker_turn_chunks.jsonl",
    },
}

# Each tuple is (lower_bound, upper_bound); None means no upper bound
LENGTH_BUCKETS: List[tuple] = [
    (1,   50),
    (51,  100),
    (101, 150),
    (151, 200),
    (201, 250),
    (251, 300),
    (301, None),
]


def load_chunks(file_path: Path) -> List[dict]:
    with file_path.open("r", encoding="utf-8") as file_handle:
        return [json.loads(line) for line in file_handle if line.strip()]


def measure_token_lengths(chunks: List[dict], tokenizer) -> List[int]:
    return [len(tokenizer.encode(chunk["text"])) for chunk in chunks]


def compute_statistics(token_lengths: List[int]) -> dict:
    sorted_token_lengths = sorted(token_lengths)
    total_count = len(sorted_token_lengths)
    return {
        "count": total_count,
        "min": sorted_token_lengths[0],
        "max": sorted_token_lengths[-1],
        "mean": round(statistics.mean(sorted_token_lengths), 1),
        "median": int(statistics.median(sorted_token_lengths)),
        "p90": sorted_token_lengths[int(total_count * 0.90)],
        "p95": sorted_token_lengths[int(total_count * 0.95)],
        "over_limit_count": sum(1 for length in sorted_token_lengths if length > EMBEDDING_MAX_TOKENS),
    }


def print_distribution_table(token_lengths: List[int], title: str) -> None:
    print(f"\n  {title}")
    print(f"  {'Token range':<22} {'Count':>6}  {'%':>6}")
    print(f"  {'-' * 38}")
    for lower_bound, upper_bound in LENGTH_BUCKETS:
        if upper_bound is None:
            range_label = f"{lower_bound}+  [over limit]"
            bucket_count = sum(1 for length in token_lengths if length >= lower_bound)
        else:
            range_label = f"{lower_bound}-{upper_bound}"
            bucket_count = sum(1 for length in token_lengths if lower_bound <= length <= upper_bound)
        bucket_percentage = bucket_count / len(token_lengths) * 100
        print(f"  {range_label:<22} {bucket_count:>6}  {bucket_percentage:>5.1f}%")


def print_summary_statistics(stats: dict) -> None:
    print(
        f"\n  min={stats['min']}  max={stats['max']}  "
        f"mean={stats['mean']}  median={stats['median']}  "
        f"p90={stats['p90']}  p95={stats['p95']}"
    )
    over_limit_count = stats["over_limit_count"]
    over_limit_percentage = over_limit_count / stats["count"] * 100
    print(f"  Chunks over {EMBEDDING_MAX_TOKENS}-token limit: {over_limit_count}/{stats['count']} ({over_limit_percentage:.1f}%)")


def run_diagnostic_for_chunk_type(
    play_files: Dict[str, Path], chunk_type_label: str, tokenizer
) -> None:
    print(f"\n{'#' * 50}")
    print(f"  CHUNK TYPE: {chunk_type_label}")
    print(f"{'#' * 50}")

    all_token_lengths: List[int] = []

    for play_name, file_path in play_files.items():
        chunks = load_chunks(file_path)
        token_lengths = measure_token_lengths(chunks, tokenizer)
        play_stats = compute_statistics(token_lengths)

        print("\n" + "=" * 50)
        print_distribution_table(token_lengths, play_name)
        print_summary_statistics(play_stats)

        all_token_lengths.extend(token_lengths)

    print("\n" + "=" * 50)
    combined_stats = compute_statistics(all_token_lengths)
    print_distribution_table(all_token_lengths, "ALL THREE PLAYS (COMBINED)")
    print_summary_statistics(combined_stats)


def run_diagnostic() -> None:
    print(f"Tokenizer : {EMBEDDING_MODEL_NAME}")
    print(f"Max input : {EMBEDDING_MAX_TOKENS} tokens")

    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)

    for chunk_type_label, play_files in CHUNK_FILE_SETS.items():
        run_diagnostic_for_chunk_type(play_files, chunk_type_label, tokenizer)

    print()


if __name__ == "__main__":
    run_diagnostic()
