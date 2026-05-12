"""Minimal loader for the Shakespeare SLM/RAG teaching dataset."""
import json
from pathlib import Path


def load_play(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_scene_chunks(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


if __name__ == "__main__":
    dataset_dir = Path(__file__).resolve().parent
    chunks = load_scene_chunks(dataset_dir / "macbeth_scene_chunks.jsonl")
    print(f"Loaded {len(chunks)} Macbeth scene chunks")
    print(chunks[0].keys())
    print(chunks[0]["scene_summary"])
