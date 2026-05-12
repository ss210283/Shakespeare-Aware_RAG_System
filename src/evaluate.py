"""
Evaluation scaffold.

This script creates a CSV template for evaluation results.
Students should extend it to run both baseline and RAG systems and then
manually or semi-automatically score the outputs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

from config import RESULTS_DIR


QUESTIONS_PATH = RESULTS_DIR / "instructor_questions.json"
OUTPUT_PATH = RESULTS_DIR / "evaluation_results_template.csv"


def load_questions(path: Path = QUESTIONS_PATH) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Question file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def create_evaluation_template() -> None:
    questions = load_questions()

    fieldnames = [
        "question_id",
        "question",
        "question_type",
        "expected_focus",
        "system",
        "retrieved_passages",
        "generated_response",
        "correctness_score",
        "grounding_score",
        "retrieval_relevance_score",
        "usefulness_score",
        "style_quality_score",
        "comments",
    ]

    rows = []
    for q in questions:
        for system_name in ["baseline", "rag"]:
            rows.append({
                "question_id": q.get("question_id", ""),
                "question": q.get("question", ""),
                "question_type": q.get("type", ""),
                "expected_focus": q.get("expected_focus", ""),
                "system": system_name,
                "retrieved_passages": "",
                "generated_response": "",
                "correctness_score": "",
                "grounding_score": "",
                "retrieval_relevance_score": "",
                "usefulness_score": "",
                "style_quality_score": "",
                "comments": "",
            })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote evaluation template to: {OUTPUT_PATH}")


if __name__ == "__main__":
    create_evaluation_template()
