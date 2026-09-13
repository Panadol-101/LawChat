from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal


CandidateSource = Literal["dense", "sparse", "graph"]


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    point_id: str
    chunk_id: str
    source: CandidateSource
    raw_score: float
    rank: int
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.point_id:
            raise ValueError("point_id must not be empty")
        if not self.chunk_id:
            raise ValueError("chunk_id must not be empty")
        if self.rank <= 0:
            raise ValueError("rank must be > 0")


@dataclass(frozen=True, slots=True)
class FusedCandidate:
    point_id: str
    chunk_id: str
    score: float
    source_ranks: dict[str, int]
    source_scores: dict[str, float]
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RRFSettings:
    k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError("RRF k must be > 0")
        if self.dense_weight < 0 or self.sparse_weight < 0:
            raise ValueError("RRF weights must be >= 0")
        if self.dense_weight == 0 and self.sparse_weight == 0:
            raise ValueError("at least one RRF weight must be > 0")

    @classmethod
    def from_env(cls) -> "RRFSettings":
        return cls(
            k=int(os.getenv("RRF_K", "60")),
            dense_weight=float(os.getenv("RRF_DENSE_WEIGHT", "1.0")),
            sparse_weight=float(os.getenv("RRF_SPARSE_WEIGHT", "1.0")),
        )

    def weights(self) -> dict[str, float]:
        return {
            "dense": self.dense_weight,
            "sparse": self.sparse_weight,
            "graph": 1.0,
        }


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[RetrievalCandidate]],
    *,
    settings: RRFSettings | None = None,
    limit: int | None = None,
) -> list[FusedCandidate]:
    settings = settings or RRFSettings()
    if limit is not None and limit <= 0:
        raise ValueError("limit must be > 0")

    fused: dict[str, _MutableFusion] = {}
    weights = settings.weights()
    for source, candidates in rankings.items():
        weight = weights.get(source, 1.0)
        if weight == 0:
            continue
        seen_in_source: set[str] = set()
        for fallback_rank, candidate in enumerate(candidates, start=1):
            if candidate.point_id in seen_in_source:
                continue
            seen_in_source.add(candidate.point_id)
            rank = candidate.rank or fallback_rank
            current = fused.get(candidate.point_id)
            if current is None:
                current = _MutableFusion(
                    point_id=candidate.point_id,
                    chunk_id=candidate.chunk_id,
                    payload=dict(candidate.payload),
                )
                fused[candidate.point_id] = current
            elif current.chunk_id != candidate.chunk_id:
                raise ValueError(
                    f"point {candidate.point_id} maps to multiple chunk IDs"
                )
            current.score += weight / (settings.k + rank)
            current.source_ranks[source] = rank
            current.source_scores[source] = candidate.raw_score
            current.payload.update(candidate.payload)

    ordered = sorted(
        fused.values(),
        key=lambda item: (
            -item.score,
            min(item.source_ranks.values()),
            item.point_id,
        ),
    )
    if limit is not None:
        ordered = ordered[:limit]
    return [item.freeze() for item in ordered]


@dataclass(slots=True)
class _MutableFusion:
    point_id: str
    chunk_id: str
    payload: dict
    score: float = 0.0
    source_ranks: dict[str, int] | None = None
    source_scores: dict[str, float] | None = None

    def __post_init__(self) -> None:
        self.source_ranks = {} if self.source_ranks is None else self.source_ranks
        self.source_scores = {} if self.source_scores is None else self.source_scores

    def freeze(self) -> FusedCandidate:
        assert self.source_ranks is not None
        assert self.source_scores is not None
        return FusedCandidate(
            point_id=self.point_id,
            chunk_id=self.chunk_id,
            score=self.score,
            source_ranks=dict(self.source_ranks),
            source_scores=dict(self.source_scores),
            payload=dict(self.payload),
        )
