"""
Build and test a simple retrieval index.

This script is a sanity check that:
1. the dataset can be loaded;
2. chunks can be created;
3. embeddings can be generated;
4. retrieval returns plausible passages.
"""

from config import DEFAULT_TOP_K, EMBEDDING_MODEL_NAME
from data_loader import load_all_plays
from chunking import create_chunks, format_chunk_for_display
from retrieval import EmbeddingRetriever


def main() -> None:
    records = load_all_plays()
    chunks = create_chunks(records)

    print(f"Loaded {len(records)} records.")
    print(f"Created {len(chunks)} retrieval chunks.")

    retriever = EmbeddingRetriever(EMBEDDING_MODEL_NAME)
    retriever.build_index(chunks)

    query = "Why does Macbeth kill Duncan?"
    results = retriever.retrieve(query, top_k=DEFAULT_TOP_K)

    print("\nQuery:", query)
    print("\nTop retrieved chunks:\n")

    for rank, (chunk, score) in enumerate(results, start=1):
        print("=" * 80)
        print(f"Rank {rank} | Score: {score:.4f}")
        print(format_chunk_for_display(chunk))


if __name__ == "__main__":
    main()
