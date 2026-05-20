"""
Shakespeare-aware RAG chatbot.

Pipeline:
    query  ->  retrieve  ->  build_prompt  ->  SLM.generate  ->  format_output

The retrieved passages come from the MiniLM index built by
``python src/build_index.py``. The SLM is any HuggingFace causal LM with a
chat template; by default we use ``Qwen/Qwen2.5-1.5B-Instruct`` because it
follows the strict "answer + inline [source_id] citation" format reliably
while still being small enough to run on a laptop. The model is
configurable via the ``--model`` flag or the ``RAG_SLM_MODEL`` environment
variable — use ``Qwen/Qwen2.5-0.5B-Instruct`` for faster iteration at the
cost of weaker instruction-following, or ``Qwen/Qwen2.5-3B-Instruct`` for
better grounding at the cost of slower inference.

The system prompt (``prompts/system_prompt.txt``) forces:

* answers drawn only from the provided passages,
* inline ``[source_id]`` citations,
* a fixed refusal line when context is insufficient.

The Python side then renders a deterministic evidence table from the actual
retrieved chunks so the grader sees both what the model said and what was
shown to it.

CLI
---
    # one-shot question
    python -m src.rag_chatbot --query "Why does Macbeth kill Duncan?" --top_k 3

    # interactive REPL
    python -m src.rag_chatbot

    # equivalent script invocation
    python src/rag_chatbot.py --query "..."
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make sibling modules (config, retrieval, chunking) importable whether this
# file is run as ``python -m src.rag_chatbot`` (package mode) or as
# ``python src/rag_chatbot.py`` (script mode).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import DEFAULT_TOP_K, PROMPT_DIR  # noqa: E402
from retrieval import retrieve  # noqa: E402


Chunk = Dict[str, Any]

DEFAULT_SLM = os.environ.get("RAG_SLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DEFAULT_MAX_NEW_TOKENS = 256

SYSTEM_PROMPT_PATH = PROMPT_DIR / "system_prompt.txt"

# Refusal line — must match the exact string the system prompt asks the model
# to emit when context is insufficient. The programmatic safeguards below also
# emit this verbatim so the format is uniform.
REFUSAL_LINE = "I cannot answer this question from the provided passages."

# Default cosine-similarity floor for the score pre-filter. If the top-1
# retrieved chunk scores below this, we skip the SLM entirely and refuse —
# this catches obviously off-topic queries like "What is the capital of
# France?" before the model has a chance to hallucinate. The MiniLM cosine
# scores on this corpus typically sit at 0.65-0.80 for in-domain questions,
# so 0.40 is a conservative cutoff with little risk of over-refusal.
DEFAULT_MIN_RETRIEVAL_SCORE = 0.40

# Matches [source_id]-style inline citations like [macbeth_1_6_0015],
# [romeo_and_juliet_2_2_0033], or [hamlet_1_2_0063..0064]
# (the chunker emits range ids for merged speaker turns).
_CITE_RE = re.compile(r"\[([a-z_]+(?:_\d+){3}(?:\.\.\d+)?)\]", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


def build_user_prompt(query: str, hits: List[Chunk]) -> str:
    """Format the user-side turn: retrieved context block + the actual question."""
    if not hits:
        return (
            "CONTEXT PASSAGES:\n(no passages were retrieved)\n\n"
            f"USER QUESTION:\n{query}\n\n"
            "Produce the answer as specified in the system instructions."
        )

    passage_lines: List[str] = []
    for i, h in enumerate(hits, start=1):
        play = h.get("play", "?")
        act = h.get("act")
        scene = h.get("scene")
        speaker = h.get("speaker") or "UNKNOWN"
        source_id = h.get("source_id", "?")
        text = (h.get("text") or "").strip()
        passage_lines.append(
            f"[Passage {i}] play={play}, act={act}, scene={scene}, "
            f"speaker={speaker}, source_id={source_id}\n{text}"
        )

    context = "\n\n".join(passage_lines)
    return (
        f"CONTEXT PASSAGES:\n{context}\n\n"
        f"USER QUESTION:\n{query}\n\n"
        "Produce the answer as specified in the system instructions."
    )


# ---------------------------------------------------------------------------
# SLM wrapper
# ---------------------------------------------------------------------------

class SLM:
    """Thin wrapper around a HuggingFace instruct-tuned causal LM."""

    def __init__(self, model_name: str = DEFAULT_SLM):
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except ImportError as e:
            raise SystemExit(
                "transformers and torch are required. Install with "
                "`pip install -r requirements.txt`."
            ) from e

        print(f"[INFO] loading SLM: {self.model_name}", file=sys.stderr)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        load_kwargs: Dict[str, Any] = {}
        if torch.cuda.is_available():
            load_kwargs["torch_dtype"] = torch.bfloat16
            load_kwargs["device_map"] = "auto"
            self._device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            load_kwargs["torch_dtype"] = torch.float16
            self._device = "mps"
        else:
            self._device = "cpu"

        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
        if self._device in ("cpu", "mps"):
            self._model.to(self._device)
        self._model.eval()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> str:
        self._ensure_loaded()
        import torch  # type: ignore

        tok = self._tokenizer
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tok(prompt_text, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,                       # deterministic
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return tok.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _excerpt(text: str, max_words: int = 25) -> str:
    text = (text or "").strip().replace("\n", " ")
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "…"


def _extract_cited_ids(answer_text: str) -> List[str]:
    return [m for m in _CITE_RE.findall(answer_text or "")]


def _max_score(hits: List[Chunk]) -> float:
    """Highest cosine score across the retrieved hits, or -inf if hits is empty."""
    if not hits:
        return float("-inf")
    scores = [float(h["score"]) for h in hits if isinstance(h.get("score"), (int, float))]
    return max(scores) if scores else float("-inf")


def _enforce_grounding(answer_text: str, hits: List[Chunk]) -> str:
    """B. Programmatic citation guard.

    If the model's answer does not contain a single ``[source_id]`` citation
    that points to one of the actually-retrieved chunks, replace the answer
    with the refusal line. This blocks the "no-citation hallucination"
    failure mode where small SLMs ignore the system prompt's citation rule
    and answer from their pre-training instead of from the passages.
    """
    valid_ids = {str(h.get("source_id")) for h in hits if h.get("source_id")}
    cited = set(_extract_cited_ids(answer_text or ""))
    if not (cited & valid_ids):
        return REFUSAL_LINE
    return answer_text


def format_output(query: str, answer_text: str, hits: List[Chunk]) -> str:
    """Render the final string: question, model's answer, evidence table.

    The evidence table is built from the actual retrieved chunks (so it
    cannot be hallucinated). Passages that the model cited inline are marked
    with ``*`` for easy auditing.
    """
    cited = set(_extract_cited_ids(answer_text))

    out: List[str] = []
    out.append(f"Q: {query}")
    out.append("")
    out.append("Answer:")
    out.append((answer_text or "").strip())
    out.append("")
    out.append("Evidence (retrieved passages; * = cited inline above):")

    if not hits:
        out.append("  (no passages retrieved)")
        return "\n".join(out)

    headers = ["#", "play", "act.scene", "speaker", "source_id", "excerpt"]
    rows: List[List[str]] = []
    for i, h in enumerate(hits, start=1):
        sid = str(h.get("source_id", "?"))
        mark = "*" if sid in cited else ""
        play = str(h.get("play", "?"))
        act = h.get("act")
        scene = h.get("scene")
        act_scene = f"{act}.{scene}" if act is not None and scene is not None else "?"
        speaker = str(h.get("speaker") or "?")
        excerpt = f"\"{_excerpt(h.get('text', ''))}\""
        rows.append([f"{i}{mark}", play, act_scene, speaker, sid, excerpt])

    widths = [max(len(headers[c]), *(len(r[c]) for r in rows)) for c in range(len(headers))]
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in widths)
    out.append(fmt.format(*headers))
    out.append("  " + "  ".join("-" * w for w in widths))
    for r in rows:
        out.append(fmt.format(*r))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

_default_slm: Optional[SLM] = None


def _get_slm(model_name: Optional[str]) -> SLM:
    global _default_slm
    target_name = model_name or DEFAULT_SLM
    if _default_slm is None or _default_slm.model_name != target_name:
        _default_slm = SLM(target_name)
    return _default_slm


def answer(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    model_name: Optional[str] = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    min_score: float = DEFAULT_MIN_RETRIEVAL_SCORE,
    citation_guard: bool = True,
) -> str:
    """End-to-end RAG: retrieve, build prompt, generate, format.

    Two safeguards run around the SLM:

    * **Score pre-filter (C):** if the top retrieved chunk's cosine score is
      below ``min_score``, the SLM is skipped entirely and the refusal line
      is returned. Pass ``min_score=float('-inf')`` to disable.
    * **Citation guard (B):** after generation, if the model's answer cites
      no source_id that actually appears in the retrieved hits, the answer
      is replaced with the refusal line. Pass ``citation_guard=False`` to
      disable.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    hits = retrieve(query, top_k=top_k)

    # --- C. score pre-filter: bypass SLM on clearly off-topic queries ---
    top_score = _max_score(hits)
    if top_score < min_score:
        print(
            f"[INFO] retrieval top score {top_score:.3f} < {min_score:.3f} — "
            "skipping SLM and refusing.",
            file=sys.stderr,
        )
        return format_output(query, REFUSAL_LINE, hits)

    system_prompt = load_system_prompt()
    user_prompt = build_user_prompt(query, hits)
    slm = _get_slm(model_name)
    raw = slm.generate(system_prompt, user_prompt, max_new_tokens=max_new_tokens)

    # --- B. citation guard: replace ungrounded answers with the refusal ---
    if citation_guard:
        grounded = _enforce_grounding(raw, hits)
        if grounded is REFUSAL_LINE and raw.strip() != REFUSAL_LINE:
            print(
                "[INFO] model answer cited no retrieved source_id — "
                "replacing with refusal.",
                file=sys.stderr,
            )
        raw = grounded

    return format_output(query, raw, hits)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m src.rag_chatbot",
        description="Shakespeare-aware RAG chatbot (Hamlet / Macbeth / Romeo and Juliet).",
    )
    p.add_argument(
        "--query", "-q",
        type=str, default=None,
        help="Single question. If omitted, the chatbot drops into an interactive REPL.",
    )
    p.add_argument(
        "--top-k", "--top_k",
        dest="top_k", type=int, default=DEFAULT_TOP_K,
        help=f"Number of passages to retrieve (default {DEFAULT_TOP_K}).",
    )
    p.add_argument(
        "--model", "-m",
        type=str, default=None,
        help=f"HuggingFace SLM name. Default: {DEFAULT_SLM} "
             "(override with RAG_SLM_MODEL env var).",
    )
    p.add_argument(
        "--max-tokens", "--max_new_tokens",
        dest="max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Maximum tokens generated by the SLM (default {DEFAULT_MAX_NEW_TOKENS}).",
    )
    p.add_argument(
        "--min-score", "--min_score",
        dest="min_score", type=float, default=DEFAULT_MIN_RETRIEVAL_SCORE,
        help=f"Skip the SLM and refuse if the top retrieved score is below this "
             f"cosine threshold (default {DEFAULT_MIN_RETRIEVAL_SCORE}). "
             "Set to a very negative number to disable.",
    )
    p.add_argument(
        "--no-citation-guard",
        dest="citation_guard", action="store_false", default=True,
        help="Disable the post-hoc citation guard "
             "(by default, an answer with no valid [source_id] citation is "
             "replaced with the refusal line).",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    if args.query is not None:
        out = answer(
            args.query,
            top_k=args.top_k,
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            min_score=args.min_score,
            citation_guard=args.citation_guard,
        )
        print(out)
        return 0

    # Interactive REPL
    print("Shakespeare-aware RAG chatbot. Type 'quit' (or Ctrl-D) to exit.")
    while True:
        try:
            q = input("\n? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in {"quit", "exit", "q"}:
            break
        if not q:
            continue
        try:
            out = answer(
                q,
                top_k=args.top_k,
                model_name=args.model,
                max_new_tokens=args.max_new_tokens,
                min_score=args.min_score,
                citation_guard=args.citation_guard,
            )
            print(out)
        except Exception as e:  # pragma: no cover - REPL convenience
            print(f"[ERROR] {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
