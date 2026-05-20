"""
Data loading utilities.

This file assumes that the processed Shakespeare dataset is available in JSON format.
Students may modify this loader if the provided dataset structure differs.

Expected examples:
1. A file containing a list of records:
   [
     {"play": "Macbeth", "act": 1, "scene": 3, "speaker": "MACBETH", "text": "..."}
   ]

2. A file containing a dictionary with a "records" or "scenes" key:
   {"records": [...]} or {"scenes": [...]}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from config import PLAY_FILES


Record = Dict[str, Any]


def _extract_records(obj: Any) -> List[Record]:
    """
    Extract a list of records from a JSON object.

    Students should adapt this function if their dataset schema differs.
    """
    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        for key in ["records", "utterances", "scenes", "chunks", "data"]:
            if key in obj and isinstance(obj[key], list):
                return obj[key]

    raise ValueError(
        "Could not extract records. Expected a list or a dictionary containing "
        "one of: records, utterances, scenes, chunks, data."
    )


def load_jsonl_records(path: Path) -> List[Record]:
    """
    Load one processed Shakespeare JSONL file (one JSON object per line).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find dataset file: {path}\n"
            "Place the provided dataset files in data/processed/."
        )

    with path.open("r", encoding="utf-8") as file_handle:
        return [json.loads(line) for line in file_handle if line.strip()]


def load_all_plays() -> List[Record]:
    """
    Load records from all three compulsory plays.
    """
    all_records: List[Record] = []

    for play_key, path in PLAY_FILES.items():
        records = load_jsonl_records(path)
        for r in records:
            r.setdefault("play_key", play_key)
        all_records.extend(records)

    return all_records


if __name__ == "__main__":
    records = load_all_plays()
    print(f"Loaded {len(records)} records.")
    print("First three records:")
    print(json.dumps(records[0], indent=2, ensure_ascii=False))
    print(json.dumps(records[1], indent=2, ensure_ascii=False))
    print(json.dumps(records[2], indent=2, ensure_ascii=False))
