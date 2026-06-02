"""
Evaluation script.

Default behaviour: loads both the RAG and Baseline systems, runs every
question in the questions JSON through each system, and writes a CSV with
retrieved_passages and generated_response pre-filled.
Score columns (correctness, grounding, etc.) are left empty for manual scoring.

Usage:
    # Full evaluation — instructor questions only (Q1–Q5)
    python3 src/evaluate.py

    # Full evaluation — merge instructor + group questions (Q1–Q5 + Q6–Q13)
    python3 src/evaluate.py --merge results/group_questions.json \
                            --output results/evaluation_results.csv

    # Merge more than one extra file
    python3 src/evaluate.py --merge results/group_questions.json results/extra.json

    # Custom primary questions file
    python3 src/evaluate.py --questions results/my_questions.json \
                            --output results/my_results.csv

    # Only generate a blank CSV template without loading any model
    python3 src/evaluate.py --template-only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure src/ is importable when the script is run from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import RESULTS_DIR

QUESTIONS_PATH = RESULTS_DIR / "instructor_questions.json"
OUTPUT_PATH = RESULTS_DIR / "evaluation_results.csv"
TEMPLATE_PATH = RESULTS_DIR / "evaluation_results_template.csv"

FIELDNAMES = [
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


def load_questions(path: Path = QUESTIONS_PATH) -> List[Dict[str, Any]]:
    """Load evaluation questions from a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Question file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _format_passages(chunks_with_scores: List[Tuple[Dict[str, Any], float]]) -> str:
    lines: List[str] = []
    for rank, (chunk, score) in enumerate(chunks_with_scores, start=1):
        play = chunk.get("play", "Unknown play")
        act = chunk.get("act", "?")
        scene = chunk.get("scene", "?")
        speaker = chunk.get("speaker", "UNKNOWN")
        text = chunk.get("text", "").strip().replace("\n", " ")
        lines.append(
            f"[{rank}] {play}, Act {act}, Scene {scene} | {speaker} (score={score:.2f}): {text}"
        )
    return "\n".join(lines) if lines else "No passages retrieved"


def _build_row(
    q: Dict[str, Any],
    system_name: str,
    response: str,
    passages_str: str,
) -> Dict[str, Any]:
    return {
        "question_id": q.get("question_id", ""),
        "question": q.get("question", ""),
        "question_type": q.get("type", ""),
        "expected_focus": q.get("expected_focus", ""),
        "system": system_name,
        "retrieved_passages": passages_str,
        "generated_response": response,
        "correctness_score": "",
        "grounding_score": "",
        "retrieval_relevance_score": "",
        "usefulness_score": "",
        "style_quality_score": "",
        "comments": "",
    }


def _write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✓ Results written to: {output_path}")


def create_evaluation_template(output_path: Path = TEMPLATE_PATH) -> None:
    # Write a blank CSV template without running any model.
    questions = load_questions()
    rows: List[Dict[str, Any]] = []
    for q in questions:
        for system_name in ("baseline", "rag"):
            rows.append(_build_row(q, system_name, "", ""))
    _write_csv(rows, output_path)
    print(f"Wrote blank evaluation template to: {output_path}")


def run_evaluation(
    questions_path: Path = QUESTIONS_PATH,
    output_path: Path = OUTPUT_PATH,
    merge_paths: List[Path] = None,
) -> None:
    """
    Run all questions through both systems and write a pre-filled CSV.
    Score columns are left empty for manual annotation.
    """
    questions = load_questions(questions_path)
    print(f"Loaded {len(questions)} question(s) from {questions_path}")

    for mp in merge_paths or []:
        extra = load_questions(mp)
        questions.extend(extra)
        print(f"Merged  {len(extra)} question(s) from {mp}")

    total = len(questions)
    print(f"Total   {total} question(s)\n")

    # --- Load systems once ---
    print("Loading RAG system (this may take ~30 s on first run)…")
    from rag_chatbot import RAGSystem

    rag = RAGSystem()
    print("  RAG system ready.\n")

    print("Loading Baseline system…")
    from baseline import BaselineSystem

    baseline = BaselineSystem()
    print("  Baseline system ready.\n")

    rows: List[Dict[str, Any]] = []

    for idx, q in enumerate(questions, start=1):
        query = q.get("question", "")
        qid = q.get("question_id", f"Q{idx}")
        print(f"[{idx}/{total}] {qid}: {query}")

        # --- RAG ---
        print("  → RAG…", end=" ", flush=True)
        rag_response, chunks_with_scores = rag.answer(query)
        passages_str = _format_passages(chunks_with_scores)
        rows.append(_build_row(q, "rag", rag_response, passages_str))
        print("done")

        # --- Baseline ---
        print("  → Baseline…", end=" ", flush=True)
        baseline_response = baseline.answer(query)
        rows.append(
            _build_row(
                q,
                "baseline",
                baseline_response,
                "N/A (baseline has no retrieval)",
            )
        )
        print("done\n")

    _write_csv(rows, output_path)
    print("All score columns are empty — please fill them in manually.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the Shakespeare RAG and Baseline systems."
    )
    parser.add_argument(
        "--questions",
        default=str(QUESTIONS_PATH),
        help=f"Path to questions JSON file (default: {QUESTIONS_PATH})",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        help=f"Path for the output CSV (default: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--merge",
        nargs="+",
        metavar="FILE",
        default=[],
        help="Additional question JSON files to merge with --questions (e.g. results/group_questions.json)",
    )
    parser.add_argument(
        "--template-only",
        action="store_true",
        help="Only write a blank CSV template without loading any model.",
    )
    args = parser.parse_args()

    if args.template_only:
        create_evaluation_template(
            Path(args.output).with_name("evaluation_results_template.csv")
        )
    else:
        run_evaluation(
            Path(args.questions),
            Path(args.output),
            merge_paths=[Path(p) for p in args.merge],
        )
