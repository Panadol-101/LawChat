from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any

from lawchat.database.models import LegalStatus

if TYPE_CHECKING:
    from .graph import (
        DocumentStatusResolution,
        LegalGraphDocument,
        LegalGraphEdge,
        SeedResolution,
    )


ACTIVE_LEGAL_STATUSES = (
    LegalStatus.EFFECTIVE.value,
    LegalStatus.PARTIALLY_EFFECTIVE.value,
)


@dataclass(frozen=True, slots=True)
class LegalIssue:
    issue_id: str
    question: str
    search_query: str


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    query: str
    as_of: date | None = None
    limit: int = 10
    candidate_limit: int = 50
    max_candidate_limit: int = 500
    # ``None`` means the temporal policy chooses the appropriate statuses.
    # A non-empty tuple is an explicit caller override (for benchmarks/admins).
    statuses: tuple[str, ...] | None = None
    document_numbers: tuple[str, ...] = ()
    document_types: tuple[str, ...] = ()
    authorities: tuple[str, ...] = ()
    legal_fields: tuple[str, ...] = ()
    # Internal, already-resolved intent propagated from the original question
    # to decomposed issue queries. It is not exposed by the HTTP request schema.
    resolved_temporal_intent: str | None = None

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query must not be empty")
        if self.limit <= 0:
            raise ValueError("limit must be > 0")
        if self.candidate_limit < self.limit:
            raise ValueError("candidate_limit must be >= limit")
        if self.max_candidate_limit < self.candidate_limit:
            raise ValueError("max_candidate_limit must be >= candidate_limit")
        if self.statuses == ():
            raise ValueError("statuses must not be empty")
        known_statuses = {item.value for item in LegalStatus}
        unknown = set(self.statuses or ()) - known_statuses
        if unknown:
            raise ValueError(f"Unknown legal statuses: {sorted(unknown)}")
        if self.resolved_temporal_intent not in {
            None, "current_law", "status_lookup", "historical"
        }:
            raise ValueError("invalid resolved_temporal_intent")


@dataclass(frozen=True, slots=True)
class ParsedLegalQuery:
    original_query: str
    semantic_query: str
    as_of: date
    document_numbers: tuple[str, ...] = ()
    referenced_articles: tuple[str, ...] = ()
    referenced_clauses: tuple[str, ...] = ()
    referenced_points: tuple[str, ...] = ()
    relationship_types: tuple[str, ...] = ()
    relationship_direction: str | None = None
    has_explicit_date: bool = False


@dataclass(frozen=True, slots=True)
class HydratedLegalChunk:
    point_id: str
    chunk_id: str
    chunk_type: str
    text: str
    retrieval_text: str | None
    chunk_metadata: dict[str, Any]
    parent_chunk_id: str | None
    parent_text: str | None
    document_id: str
    title: str
    document_number: str | None
    document_type: str | None
    authority: str | None
    legal_field: str | None
    source_url: str
    article: str | None
    clause: str | None
    point: str | None
    status: str
    valid_from: date
    valid_to: date | None
    status_scope: str = "document"
    version_id: str | None = None
    version_source_url: str | None = None
    version_source_revision: str | None = None
    content_valid_from: date | None = None
    content_valid_to: date | None = None


@dataclass(frozen=True, slots=True)
class LegalCitation:
    document_id: str
    title: str
    document_number: str | None
    article: str | None
    clause: str | None
    point: str | None
    source_url: str
    as_of: date
    status: str
    status_scope: str = "document"
    version_id: str | None = None
    version_source_url: str | None = None
    version_source_revision: str | None = None
    content_valid_from: date | None = None
    content_valid_to: date | None = None

    @property
    def label(self) -> str:
        location = " ".join(
            value
            for value in (
                f"Điều {self.article}" if self.article else None,
                f"Khoản {self.clause}" if self.clause else None,
                f"Điểm {self.point}" if self.point else None,
            )
            if value
        )
        document = self.document_number or self.title
        return f"{location}, {document}" if location else document


@dataclass(frozen=True, slots=True)
class RetrievedLegalChunk:
    point_id: str
    chunk_id: str
    score: float
    text: str
    context_text: str
    chunk_type: str
    citation: LegalCitation
    authority: str | None = None
    legal_field: str | None = None
    source_ranks: dict[str, int] = field(default_factory=dict)
    source_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_text: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievalResponse:
    query: str
    semantic_query: str
    as_of: date
    results: tuple[RetrievedLegalChunk, ...]
    searched_candidates: int
    rejected_candidates: int
    retrieval_sources: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    temporal_intent: str = "current_law"
    temporal_explicit_as_of: bool = False
    historical_content_available: bool = True
    seed_documents: tuple[LegalGraphDocument, ...] = ()
    seed_resolutions: tuple[SeedResolution, ...] = ()
    status_resolution: DocumentStatusResolution | None = None
    related_documents: tuple[LegalGraphDocument, ...] = ()
    graph_edges: tuple[LegalGraphEdge, ...] = ()
    timings: dict[str, float] = field(default_factory=dict)
    legal_issues: tuple[LegalIssue, ...] = ()
    retrieval_trace: dict[str, tuple[str, ...]] = field(default_factory=dict)
