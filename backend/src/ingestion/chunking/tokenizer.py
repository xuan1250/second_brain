"""Docling tokenizer bindings, offline BaseTokenizer fallback shim, and FastRegexTokenizer."""
from __future__ import annotations

import re
from typing import Any

try:
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
    from docling_core.types.doc import DoclingDocument
except (ImportError, OSError):
    HybridChunker = None  # type: ignore

    class BaseTokenizer:  # type: ignore
        def __init__(self, max_tokens: int = 256, **kwargs: Any) -> None:
            self.max_tokens = max_tokens

        def count_tokens(self, text: str) -> int:
            return 0

    DoclingDocument = None  # type: ignore


class FastRegexTokenizer(BaseTokenizer):
    """Deterministic, zero-network tokenizer aligned with BAAI/bge-m3 token budgets."""

    max_tokens: int = 256
    model_name: str = "BAAI/bge-m3"

    def __init__(self, max_tokens: int = 256, **kwargs: Any) -> None:
        super().__init__(max_tokens=max_tokens, **kwargs)

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))

    def get_max_tokens(self) -> int:
        return self.max_tokens

    def get_tokenizer(self) -> Any:
        return self.count_tokens


def _build_tokenizer(max_tokens: int, use_hf_tokenizer: bool = False) -> BaseTokenizer:
    if use_hf_tokenizer:
        try:
            from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
            from transformers import AutoTokenizer

            hf_tok = AutoTokenizer.from_pretrained("BAAI/bge-m3", local_files_only=True)
            return HuggingFaceTokenizer(tokenizer=hf_tok, max_tokens=max_tokens)
        except Exception:
            pass
    return FastRegexTokenizer(max_tokens=max_tokens)
