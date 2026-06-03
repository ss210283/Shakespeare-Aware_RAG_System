"""Shakespeare-aware RAG chatbot.

Pipeline: query → retrieve → score filter → build prompt (PDR) → generate → citation guard
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure src/ is on sys.path regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from chunking import format_chunk_for_display
from config import DATA_DIR, DEFAULT_TOP_K, EMBEDDING_MODEL_NAME, PROMPT_DIR
from retrieval import EmbeddingRetriever

Chunk = Dict[str, Any]

_LOCAL_MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "qwen2.5-1.5b-instruct"
# Use local weights if present, otherwise auto-download from HuggingFace on first run.
MODEL_NAME = str(_LOCAL_MODEL_PATH) if _LOCAL_MODEL_PATH.exists() else "Qwen/Qwen2.5-1.5B-Instruct"
EMBEDDINGS_PATH   = DATA_DIR / "embeddings.npy"
CHUNKS_META_PATH  = DATA_DIR / "chunks_meta.json"
SCENE_LOOKUP_PATH = DATA_DIR / "scene_lookup.json"

MIN_RETRIEVAL_SCORE = 0.25
MAX_SCENE_CHARS = 1500  # ~300-400 tokens per scene; keeps PDR useful without OOM on MPS

_STYLISED_KEYWORDS = frozenset([
    "shakespearean", "shakespeare's style", "in the style of",
    "stylised", "stylized", "in shakespeare",
])


def _is_stylised_request(query: str) -> bool:
    return any(kw in query.lower() for kw in _STYLISED_KEYWORDS)


class RAGSystem:

    def __init__(self):
        print("Loading language model...")
        self.tokenizer, self.model = self._load_model()
        self.system_prompt = (PROMPT_DIR / "system_prompt.txt").read_text(encoding="utf-8").strip()

        self.retriever = EmbeddingRetriever(EMBEDDING_MODEL_NAME)
        if EMBEDDINGS_PATH.exists() and CHUNKS_META_PATH.exists():
            print("Loading cached embeddings...")
            self.retriever.embeddings = np.load(EMBEDDINGS_PATH)
            with CHUNKS_META_PATH.open("r", encoding="utf-8") as f:
                self.retriever.chunks = json.load(f)
            print(f"Index ready: {len(self.retriever.chunks)} chunks.\n")
        else:
            raise FileNotFoundError(
                "Retrieval index not found. Run `python3 src/build_index.py` first."
            )

        self.scene_lookup: Dict[str, str] = {}
        if SCENE_LOOKUP_PATH.exists():
            self.scene_lookup = json.loads(SCENE_LOOKUP_PATH.read_text(encoding="utf-8"))

    def answer(self, query: str, top_k: int = DEFAULT_TOP_K) -> Tuple[str, List[Tuple[Chunk, float]]]:
        retrieved = self.retriever.retrieve(query, top_k=top_k)

        if retrieved[0][1] < MIN_RETRIEVAL_SCORE:
            msg = (
                "[Out of scope] This question does not appear to be about "
                "Hamlet, Macbeth, or Romeo and Juliet."
            )
            return msg, retrieved

        is_stylised = _is_stylised_request(query)
        prompt = self._build_rag_prompt(query, retrieved)
        raw_answer = self._generate(prompt)
        return self._apply_citation_guard(raw_answer, is_stylised), retrieved

    def _get_scene_contexts(self, retrieved: List[Tuple[Chunk, float]]) -> List[Tuple[str, str]]:
        """Return (label, scene_text) for each unique scene in the retrieved chunks."""
        seen: set = set()
        contexts = []
        for chunk, _ in retrieved:
            sid = chunk.get("scene_id")
            if sid and sid not in seen and sid in self.scene_lookup:
                seen.add(sid)
                label = f"{chunk.get('play')}, Act {chunk.get('act')}, Scene {chunk.get('scene')}"
                contexts.append((label, self.scene_lookup[sid]))
        return contexts

    def _build_rag_prompt(self, query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
        parts = []

        scene_contexts = self._get_scene_contexts(retrieved)
        if scene_contexts:
            scene_blocks = [
                f"[{label}]\n{text[:MAX_SCENE_CHARS]}{'...' if len(text) > MAX_SCENE_CHARS else ''}"
                for label, text in scene_contexts
            ]
            parts.append("Scene context:\n" + "\n\n".join(scene_blocks))
            parts.append("---")

        passage_blocks = [
            f"[source_id: {c.get('source_id', c.get('chunk_id', ''))} | score: {s:.3f}]\n"
            f"{format_chunk_for_display(c)}"
            for c, s in retrieved
        ]
        parts.append(
            "Retrieved passages (cite these using [source_id]):\n"
            + "\n\n".join(passage_blocks)
        )
        parts.append(f"User question:\n{query}")
        return "\n\n".join(parts)

    def _apply_citation_guard(self, answer: str, is_stylised: bool) -> str:
        if is_stylised:
            # Enforce 150-word hard limit required by the assignment spec
            words = answer.split()
            if len(words) > 150:
                answer = " ".join(words[:150]) + " …"
            return f"[Creative output — not textual evidence]\n{answer}"
        return answer

    def _generate(self, user_content: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": user_content},
        ]
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        if hasattr(tokenized, "input_ids"):
            input_ids     = tokenized.input_ids.to(self.model.device)
            attention_mask = tokenized.attention_mask.to(self.model.device)
        else:
            input_ids     = tokenized.to(self.model.device)
            attention_mask = torch.ones_like(input_ids)

        prompt_length = input_ids.shape[-1]

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=300,
                do_sample=False,
            )

        generated_tokens = outputs[0][prompt_length:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    @staticmethod
    def _load_model():
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            device_map="auto",
        )
        return tokenizer, model


def _print_result(answer: str, retrieved: List[Tuple[Chunk, float]]) -> None:
    print("\nRetrieved evidence:")
    for rank, (chunk, score) in enumerate(retrieved, start=1):
        print("-" * 60)
        print(f"Rank {rank} | Score: {score:.4f}")
        print(format_chunk_for_display(chunk))
    print("\nGenerated answer:")
    print(answer)
    print()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Shakespeare-aware RAG chatbot")
    parser.add_argument("--query",  type=str, default=None)
    parser.add_argument("--top_k",  type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    system = RAGSystem()

    if args.query:
        answer, retrieved = system.answer(args.query, top_k=args.top_k)
        _print_result(answer, retrieved)
        return

    print("Shakespeare-aware RAG chatbot ready. Type 'quit' to exit.\n")
    while True:
        query = input("Question: ").strip()
        if not query:
            continue
        if query.lower() in {"quit", "exit"}:
            break
        answer, retrieved = system.answer(query, top_k=args.top_k)
        _print_result(answer, retrieved)


if __name__ == "__main__":
    main()
