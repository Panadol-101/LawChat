from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


CHUNKER_VERSION = "4.0.0"


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """
    Structure-aware Legal RAG chunking configuration.

    Token limits are fallback safety rails only. The chunker always prefers
    legal/semantic boundaries before considering token-based splitting.
    """

    target_min_tokens: int = 600
    target_tokens: int = 700
    target_max_tokens: int = 800
    max_child_tokens: int = 1200
    freeform_target_tokens: int = 700
    freeform_max_tokens: int = 1200
    # Hard limit for the final retrieval_text after metadata and structural
    # context have been added. Unlike the strategy-specific limits above,
    # this invariant applies to every indexable chunk.
    max_indexable_tokens: int = 1200
    enable_semantic_overlap: bool = True
    overlap_ratio: float = 0.125
    max_overlap_ratio: float = 0.20
    min_overlap_tokens: int = 60
    overlap_min_source_tokens: int = 600
    min_tail_tokens: int = 100

    # Keep temporal metadata as structured payload in every chunk regardless
    # of this flag. This flag only controls whether it is duplicated into
    # retrieval_text for embedding.
    include_temporal_metadata_in_retrieval_text: bool = False

    # Legal basis / preamble can be useful for queries about legal basis,
    # authority and references.
    index_preamble: bool = True

    def __post_init__(self) -> None:
        if self.max_child_tokens <= 0:
            raise ValueError("max_child_tokens must be > 0")

        if not (
            0 < self.target_min_tokens
            <= self.target_tokens
            <= self.target_max_tokens
        ):
            raise ValueError(
                "target token policy must satisfy 0 < min <= target <= "
                "target_max"
            )

        if self.freeform_target_tokens <= 0:
            raise ValueError("freeform_target_tokens must be > 0")

        if self.freeform_max_tokens < self.freeform_target_tokens:
            raise ValueError(
                "freeform_max_tokens must be >= freeform_target_tokens"
            )

        if self.max_indexable_tokens <= 0:
            raise ValueError("max_indexable_tokens must be > 0")

        if not 0 <= self.overlap_ratio < 0.5:
            raise ValueError("overlap_ratio must be in [0, 0.5)")

        if not self.overlap_ratio <= self.max_overlap_ratio < 0.5:
            raise ValueError(
                "max_overlap_ratio must be >= overlap_ratio and < 0.5"
            )

        if self.min_overlap_tokens < 0:
            raise ValueError("min_overlap_tokens must be >= 0")

        if self.overlap_min_source_tokens < 0:
            raise ValueError("overlap_min_source_tokens must be >= 0")

        if self.min_tail_tokens < 0:
            raise ValueError("min_tail_tokens must be >= 0")

    @property
    def overlap_reserve_tokens(self) -> int:
        if not self.enable_semantic_overlap:
            return 0
        return max(
            self.min_overlap_tokens,
            round(self.max_indexable_tokens * self.overlap_ratio),
        )

    @property
    def base_indexable_tokens(self) -> int:
        # Structured legal units use the full hard limit.  Overlap is added
        # only to prose fragments and is budgeted by the overlap pass itself.
        return self.max_indexable_tokens

    @property
    def effective_target_min_tokens(self) -> int:
        return min(self.target_min_tokens, self.max_indexable_tokens)

    @property
    def effective_target_tokens(self) -> int:
        return min(self.target_tokens, self.max_indexable_tokens)

    @property
    def effective_target_max_tokens(self) -> int:
        return min(self.target_max_tokens, self.max_indexable_tokens)


@dataclass(slots=True)
class ChunkingContext:
    doc_id: str
    structure_type: str
    parsed_document: Mapping[str, Any]
    metadata: Mapping[str, Any]
    config: ChunkingConfig
    factory: Any

    # Strategies can record which parsed nodes have already been consumed.
    consumed_node_ids: set[int] = field(default_factory=set)
