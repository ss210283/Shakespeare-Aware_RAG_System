"""
Baseline system scaffold.

Students must implement a baseline for comparison with the RAG system.
A baseline may be:
- prompt-only generation without retrieval;
- simple keyword search;
- retrieval-only response without generation;
- another justified minimal approach.

The baseline must be described and compared against the improved RAG system.
"""

from __future__ import annotations


def baseline_answer(query: str) -> str:
    """
    Placeholder baseline.

    Replace this with a real baseline method.
    """
    # TODO: Implement a meaningful baseline.
    return (
        "[BASELINE PLACEHOLDER]\n"
        "Implement a baseline system and compare it with your RAG-based system."
    )


if __name__ == "__main__":
    question = "Who is Hamlet?"
    print("Question:", question)
    print(baseline_answer(question))
