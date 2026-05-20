"""
Build speaker-turn-level chunks from utterance-level JSONL files.

A speaker turn = consecutive utterances by the same speaker within the same scene.
When the speaker changes (or the scene changes), a new turn begins.
"""

from pathlib import Path
from typing import List
import json

DATASET_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

PLAY_FILES = {
    "hamlet":          DATASET_DIR / "hamlet_utterances.jsonl",
    "macbeth":         DATASET_DIR / "macbeth_utterances.jsonl",
    "romeo_and_juliet": DATASET_DIR / "romeo_and_juliet_utterances.jsonl",
}


def load_utterances(file_path: Path) -> List[dict]:
    with file_path.open("r", encoding="utf-8") as file_handle:
        return [json.loads(line) for line in file_handle if line.strip()]


def build_speaker_turns(utterances: List[dict]) -> List[dict]:
    sorted_utterances = sorted(utterances, key=lambda u: u["utterance_id"])

    speaker_turns = []
    current_turn = None
    turn_index = 1

    for utterance in sorted_utterances:
        same_scene = (
            current_turn is not None
            and utterance["act"] == current_turn["act"]
            and utterance["scene"] == current_turn["scene"]
            and utterance["speaker"] == current_turn["speaker"]
        )

        if same_scene:
            current_turn["text"] += " " + utterance["text"]
        else:
            if current_turn is not None:
                speaker_turns.append(current_turn)

            current_turn = {
                "turn_id": f"{utterance['play'].lower().replace(' ', '_')}_turn_{turn_index:04d}",
                "play": utterance["play"],
                "act": utterance["act"],
                "scene": utterance["scene"],
                "speaker": utterance["speaker"],
                "text": utterance["text"],
                "scene_summary": utterance.get("scene_summary", ""),
                "keywords": utterance.get("keywords", []),
            }
            turn_index += 1

    if current_turn is not None:
        speaker_turns.append(current_turn)

    return speaker_turns


def save_speaker_turns(speaker_turns: List[dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as file_handle:
        for turn in speaker_turns:
            file_handle.write(json.dumps(turn, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(speaker_turns)} speaker turns → {output_path.name}")


def main() -> None:
    for play_key, utterance_file in PLAY_FILES.items():
        print(f"\nProcessing {play_key}...")
        utterances = load_utterances(utterance_file)
        speaker_turns = build_speaker_turns(utterances)
        output_path = DATASET_DIR / f"{play_key}_speaker_turn_chunks.jsonl"
        save_speaker_turns(speaker_turns, output_path)


if __name__ == "__main__":
    main()