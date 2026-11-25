from __future__ import annotations

DEFAULT_MODEL = "google/flan-t5-small"


class LocalIncidentLLM:
    """Run incident-analysis prompts with a small local seq2seq model."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        max_input_tokens: int = 1024,
        max_new_tokens: int = 256,
    ) -> None:
        if max_input_tokens < 1 or max_new_tokens < 1:
            raise ValueError("Token limits must be positive.")
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "The local LLM dependencies are missing. Install with: "
                'python -m pip install -e ".[llm]"'
            ) from error

        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except (OSError, RuntimeError, ValueError) as error:
            raise RuntimeError(
                f"Unable to load local LLM '{model_name}'. "
                "The first run downloads the model from Hugging Face."
            ) from error

    def generate(self, prompt: str, *, max_new_tokens: int | None = None) -> str:
        """Generate an answer from a complete RAG prompt."""
        if not prompt.strip():
            raise ValueError("The LLM prompt cannot be empty.")
        generation_limit = max_new_tokens or self.max_new_tokens
        if generation_limit < 1:
            raise ValueError("max_new_tokens must be positive.")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=generation_limit,
            do_sample=False,
        )
        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        if not answer:
            raise RuntimeError("The local LLM returned an empty response.")
        return answer
