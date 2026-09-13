from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from lawchat.retrieval import LegalCitation


@dataclass(frozen=True, slots=True)
class Evidence:
    evidence_id: str
    document_id: str
    chunk_id: str
    text: str
    citation: LegalCitation
    role: str
    token_count: int
    related_document_numbers: tuple[str, ...] = ()
    related_statuses: tuple[str, ...] = ()
    issue_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PackedContext:
    evidence: tuple[Evidence, ...]
    rendered_context: str
    used_tokens: int
    token_budget: int
    dropped_chunk_ids: tuple[str, ...]


class TokenCounter(Protocol):
    def count_tokens(self, text: str) -> int: ...


@dataclass(frozen=True, slots=True)
class TokenBudget:
    model_context_window: int = 16_000
    system_prompt_tokens: int = 2_000
    answer_reserve_tokens: int = 3_000

    def __post_init__(self) -> None:
        if self.model_context_window <= 0:
            raise ValueError("model_context_window must be > 0")
        if self.system_prompt_tokens < 0:
            raise ValueError("system_prompt_tokens must be >= 0")
        if self.answer_reserve_tokens < 0:
            raise ValueError("answer_reserve_tokens must be >= 0")
        if self.evidence_tokens <= 0:
            raise ValueError("evidence token budget must be > 0")

    @property
    def evidence_tokens(self) -> int:
        return (
            self.model_context_window
            - self.system_prompt_tokens
            - self.answer_reserve_tokens
        )
