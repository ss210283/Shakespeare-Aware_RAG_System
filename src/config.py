"""
Configuration for the Assignment 2 starter code.

Students should adjust these values to match their own implementation.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROMPT_DIR = PROJECT_ROOT / "prompts"
RESULTS_DIR = PROJECT_ROOT / "results"

PLAY_FILES = {
    "hamlet": DATA_DIR / "hamlet_speaker_turn_chunks.jsonl",
    "macbeth": DATA_DIR / "macbeth_speaker_turn_chunks.jsonl",
    "romeo_and_juliet": DATA_DIR / "romeo_and_juliet_speaker_turn_chunks.jsonl",
}

DEFAULT_TOP_K = 8

# Suggested lightweight embedding model.
# Students may change this and justify the choice in the report.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
