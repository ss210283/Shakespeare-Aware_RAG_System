"""
Baseline system: prompt-only generation without retrieval.
The model answers solely from its pretrained knowledge — no context is injected.
This serves as the comparison baseline for the RAG system.
"""

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

_LOCAL_MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "qwen2.5-1.5b-instruct"
MODEL_NAME = str(_LOCAL_MODEL_PATH) if _LOCAL_MODEL_PATH.exists() else "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "baseline_system_prompt.txt"
)


class BaselineSystem:

    def __init__(
        self,
        model_path: str = MODEL_NAME,
        prompt_path: Path = PROMPT_PATH,
    ):
        self.tokenizer, self.model = self._load_model_and_tokenizer(model_path)
        self.system_prompt = Path(prompt_path).read_text(encoding="utf-8").strip()

    def answer(self, query: str) -> str:
        messages = self._build_chat_messages(query)
        prompt_tokens, prompt_length, attention_mask = self._tokenize_messages(messages)
        outputs = self._generate_response_tokens(prompt_tokens, attention_mask)
        return self._decode_response(outputs, prompt_length)

    def _build_chat_messages(self, query: str) -> list:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]

    def _tokenize_messages(self, messages: list) -> tuple:
        tokenized_chat = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        )

        # Newer transformers returns BatchEncoding; older returns a raw tensor
        if hasattr(tokenized_chat, "input_ids"):
            prompt_tokens = tokenized_chat.input_ids.to(self.model.device)
            attention_mask = tokenized_chat.attention_mask.to(self.model.device)
        else:
            prompt_tokens = tokenized_chat.to(self.model.device)
            attention_mask = torch.ones_like(prompt_tokens)
        return prompt_tokens, prompt_tokens.shape[-1], attention_mask

    def _generate_response_tokens(
        self, prompt_tokens: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():  # disable gradient computation during inference
            return self.model.generate(
                prompt_tokens,
                attention_mask=attention_mask,
                max_new_tokens=300,
                do_sample=False,  # greedy decoding for deterministic output
            )

    def _decode_response(self, outputs: torch.Tensor, prompt_length: int) -> str:

        # Slice off the prompt tokens; keep only the newly generated part
        generated_tokens = outputs[0][prompt_length:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    @staticmethod
    def _load_model_and_tokenizer(model_path: str):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
        )
        return tokenizer, model


if __name__ == "__main__":
    print("Loading model...")
    system = BaselineSystem()
    print("Model loaded. Type 'quit' to exit.\n")

    while True:
        query = input("Q: ").strip()
        if query.lower() in {"quit", "exit"}:
            break
        if not query:
            continue
        print(f"A: {system.answer(query)}\n")
