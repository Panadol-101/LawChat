from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


BGE_M3_TOKENIZER_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


@dataclass(slots=True)
class HuggingFaceTokenizerCounter:
    """Exact chunk-budget counter without loading an embedding model."""

    model_name: str = "BAAI/bge-m3"
    revision: str | None = BGE_M3_TOKENIZER_REVISION
    cache_dir: str = "data/.cache/huggingface"
    local_files_only: bool = False
    _tokenizer: Any = field(init=False, repr=False)
    _count_cache: dict[str, int] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        tokenizer_path = hf_hub_download(
            repo_id=self.model_name,
            filename="tokenizer.json",
            revision=self.revision,
            cache_dir=str(Path(self.cache_dir).resolve()),
            local_files_only=self.local_files_only,
        )
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.no_truncation()

    def count_tokens(self, text: str) -> int:
        cached = self._count_cache.get(text)
        if cached is not None:
            return cached
        value = len(self._tokenizer.encode(text, add_special_tokens=True).ids)
        if len(self._count_cache) >= 50_000:
            self._count_cache.clear()
        self._count_cache[text] = value
        return value

    def count_many_tokens(self, texts: Sequence[str]) -> list[int]:
        return [
            len(item.ids)
            for item in self._tokenizer.encode_batch(
                list(texts), add_special_tokens=True
            )
        ]

    def split_text(self, text: str, *, max_tokens: int) -> list[str]:
        if max_tokens <= 2:
            raise ValueError("max_tokens must be > 2")
        special_tokens = len(
            self._tokenizer.encode("", add_special_tokens=True).ids
        )
        content_budget = max(1, max_tokens - special_tokens)
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        offsets = [item for item in encoding.offsets if item[1] > item[0]]
        if not offsets:
            return [text] if text else []
        chunks: list[str] = []
        for index in range(0, len(offsets), content_budget):
            group = offsets[index:index + content_budget]
            value = text[group[0][0]:group[-1][1]].strip()
            if value:
                chunks.append(value)
        return chunks

    def tail_text(self, text: str, *, max_tokens: int) -> str:
        pieces = self.split_text(text, max_tokens=max_tokens)
        return pieces[-1] if pieces else ""
