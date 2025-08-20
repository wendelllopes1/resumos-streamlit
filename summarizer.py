from typing import List
from transformers import pipeline, AutoTokenizer

DEFAULT_MODEL = "sshleifer/distilbart-cnn-12-6"

class TextSummarizer:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: int = -1):
        self.model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self._pipe = pipeline(
            "summarization",
            model=model_name,
            tokenizer=self._tokenizer,
            device=device
        )
        self.max_input = getattr(self._pipe.tokenizer, "model_max_length", 1024)

    def _chunk_text(self, text: str, max_tokens: int = 850, overlap: int = 50) -> List[str]:
        tokens = self._tokenizer.encode(text, add_special_tokens=False)
        chunks, i = [], 0
        while i < len(tokens):
            window = tokens[i:i+max_tokens]
            chunks.append(self._tokenizer.decode(window, skip_special_tokens=True))
            i += max_tokens - overlap
        return chunks

    def summarize(self, text: str, max_summary_tokens: int = 140, min_summary_tokens: int = 40, recursive: bool = True) -> str:
        text = text.strip()
        if not text:
            return ""
        token_count = len(self._tokenizer.encode(text, add_special_tokens=False))
        if token_count <= self.max_input - 32:
            return self._pipe(text, max_length=max_summary_tokens, min_length=min_summary_tokens, do_sample=False)[0]["summary_text"].strip()
        parts = self._chunk_text(text, max_tokens=min(self.max_input - 64, 850))
        partials = []
        for p in parts:
            s = self._pipe(p, max_length=max_summary_tokens, min_length=min_summary_tokens, do_sample=False)[0]["summary_text"].strip()
            partials.append(s)
        joined = "\n".join(partials)
        if recursive:
            return self._pipe(joined, max_length=max_summary_tokens, min_length=min_summary_tokens, do_sample=False)[0]["summary_text"].strip()
        return joined
