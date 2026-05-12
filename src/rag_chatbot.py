"""
Minimal RAG chatbot scaffold.

This file deliberately leaves the language-model call as a placeholder.
Students must connect it to their chosen local model or approved hosted API.

The starter implementation prints the RAG prompt so that the retrieval and
prompt construction pipeline can be tested before generation is added.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from config import DEFAULT_TOP_K, EMBEDDING_MODEL_NAME, PROMPT_DIR
from data_loader import load_all_plays
from chunking import create_chunks, format_chunk_for_display
from retrieval import EmbeddingRetriever


Chunk = Dict[str, Any]


def load_system_prompt() -> str:
    prompt_path = PROMPT_DIR / "system_prompt.txt"
    return prompt_path.read_text(encoding="utf-8")


def build_rag_prompt(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    """
    Build a prompt for a RAG-based answer.
    """
    system_prompt = load_system_prompt()

    context_blocks = []
    for rank, (chunk, score) in enumerate(retrieved, start=1):
        context_blocks.append(
            f"[Context {rank} | similarity={score:.4f}]\n"
            f"{format_chunk_for_display(chunk)}"
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""{system_prompt}

Retrieved context:
{context}

User question:
{query}

Answer:
"""
    return prompt


def generate_answer(prompt: str) -> str:
    """
    Placeholder language-model interface.

    Students must replace this with one of:
    - a local HuggingFace model;
    - an approved hosted API;
    - another justified SLM interface.

    The returned answer must be conditioned on the retrieved context.
    """
    # TODO: Replace this placeholder with a real model call.
    return (
        "[PLACEHOLDER ANSWER]\n"
        "Connect this function to your selected language model or API. "
        "Your final answer should use the retrieved context and should not invent unsupported details."
    )


def main() -> None:
    records = load_all_plays()
    chunks = create_chunks(records)

    retriever = EmbeddingRetriever(EMBEDDING_MODEL_NAME)
    retriever.build_index(chunks)

    print("Shakespeare-aware RAG chatbot scaffold.")
    print("Type 'quit' to exit.\n")

    while True:
        query = input("Question: ").strip()
        if query.lower() in {"quit", "exit"}:
            break

        retrieved = retriever.retrieve(query, top_k=DEFAULT_TOP_K)
        prompt = build_rag_prompt(query, retrieved)
        answer = generate_answer(prompt)

        print("\nRetrieved evidence:")
        for rank, (chunk, score) in enumerate(retrieved, start=1):
            print("-" * 80)
            print(f"Rank {rank} | Score: {score:.4f}")
            print(format_chunk_for_display(chunk))

        print("\nGenerated answer:")
        print(answer)
        print("\n")


if __name__ == "__main__":
    main()
