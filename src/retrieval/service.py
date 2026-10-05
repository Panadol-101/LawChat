from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from database import LegalMetadataFilter

from .dense import DenseSearchFilter, DenseSearchResult
from .hydration import (
    ChunkHydrator,
    LegalContextBuilder,
    hydration_rejection_reasons,
)
from .models import (
    HydratedLegalChunk,
    LegalCitation,
    RetrievalRequest,
    RetrievalResponse,
    RetrievedLegalChunk,
)
from .query_parser import LegalQueryParser
from .temporal import TemporalPolicy


class CandidateSearcher(Protocol):
    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        score_threshold: float | None = None,
        filters: DenseSearchFilter | None = None,
    ) -> list[DenseSearchResult]: ...


class LegalRetrievalService:
    """Orchestrate semantic ranking and deterministic legal validation."""

    def __init__(
        self,
        searcher: CandidateSearcher,
        hydrator: ChunkHydrator,
        *,
        physical_collection: str | None = None,
        query_parser: LegalQueryParser | None = None,
        context_builder: LegalContextBuilder | None = None,
        temporal_policy: TemporalPolicy | None = None,
        content_scope: str = "current",
        historical_content_enabled: bool = False,
    ) -> None:
        self.searcher = searcher
        self.hydrator = hydrator
        self.physical_collection = physical_collection
        self.query_parser = query_parser or LegalQueryParser()
        self.context_builder = context_builder or LegalContextBuilder()
        self.temporal_policy = temporal_policy or TemporalPolicy()
        self.content_scope = content_scope
        self.historical_content_enabled = historical_content_enabled

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        parsed = self.query_parser.parse(request.query, as_of=request.as_of)
        document_numbers = _merge_values(
            request.document_numbers,
            parsed.document_numbers,
        )
        temporal = self.temporal_policy.decide(parsed, request)
        metadata_filters = LegalMetadataFilter(
            as_of=parsed.as_of,
            statuses=temporal.allowed_statuses,
            document_numbers=document_numbers,
            document_types=request.document_types,
            authorities=request.authorities,
            legal_fields=request.legal_fields,
            content_scope=self.content_scope,
            temporal_intent=temporal.intent.value,
        )
        # Status is deliberately not filtered from Qdrant. Its payload stores a
        # snapshot value, while PostgreSQL owns the valid-time history.
        dense_filters = DenseSearchFilter(
            document_numbers=document_numbers,
            document_types=request.document_types,
            authorities=request.authorities,
            legal_fields=request.legal_fields,
            content_as_of=(
                parsed.as_of if self.content_scope == "historical" else None
            ),
        )

        candidate_limit = request.candidate_limit
        candidates: list[DenseSearchResult] = []
        hydrated: list[HydratedLegalChunk] = []
        while True:
            candidates = self.searcher.search(
                parsed.semantic_query,
                limit=candidate_limit,
                filters=dense_filters,
            )
            hydrated = self.hydrator.hydrate(
                [candidate.point_id for candidate in candidates],
                metadata_filters,
                collection=self.physical_collection,
            )
            if (
                len(hydrated) >= request.limit
                or len(candidates) < candidate_limit
                or candidate_limit >= request.max_candidate_limit
            ):
                break
            candidate_limit = min(request.max_candidate_limit, candidate_limit * 2)

        rejection_reasons = hydration_rejection_reasons(
            self.hydrator,
            [candidate.point_id for candidate in candidates],
            hydrated,
            metadata_filters,
            collection=self.physical_collection,
        )
        warnings = tuple(dict.fromkeys((*temporal.warnings, *rejection_reasons)))

        score_by_point = {candidate.point_id: candidate.score for candidate in candidates}
        results = tuple(
            self._to_result(chunk, score_by_point[chunk.point_id], parsed.as_of)
            for chunk in hydrated[: request.limit]
        )
        return RetrievalResponse(
            query=parsed.original_query,
            semantic_query=parsed.semantic_query,
            as_of=parsed.as_of,
            results=results,
            searched_candidates=len(candidates),
            rejected_candidates=len(candidates) - len(hydrated),
            rejection_reasons=rejection_reasons,
            retrieval_sources=("dense",),
            warnings=warnings,
            temporal_intent=temporal.intent.value,
            temporal_explicit_as_of=temporal.explicit_as_of,
            historical_content_available=(
                not temporal.historical_content_required
                or (self.historical_content_enabled and bool(hydrated))
            ),
        )

    def _to_result(
        self,
        chunk: HydratedLegalChunk,
        score: float,
        as_of,
    ) -> RetrievedLegalChunk:
        citation = LegalCitation(
            document_id=chunk.document_id,
            title=chunk.title,
            document_number=chunk.document_number,
            article=chunk.article,
            clause=chunk.clause,
            point=chunk.point,
            source_url=(
                chunk.version_source_url
                if self.content_scope == "historical" and chunk.version_source_url
                else chunk.source_url
            ),
            as_of=as_of,
            status=chunk.status,
            status_scope=chunk.status_scope,
            version_id=chunk.version_id,
            version_source_url=chunk.version_source_url,
            version_source_revision=chunk.version_source_revision,
            content_valid_from=chunk.content_valid_from,
            content_valid_to=chunk.content_valid_to,
        )
        return RetrievedLegalChunk(
            point_id=chunk.point_id,
            chunk_id=chunk.chunk_id,
            score=score,
            text=chunk.text,
            context_text=self.context_builder.build(chunk),
            chunk_type=chunk.chunk_type,
            citation=citation,
            authority=chunk.authority,
            legal_field=chunk.legal_field,
            source_scores={"dense": score},
            metadata=chunk.chunk_metadata,
            parent_text=chunk.parent_text,
        )


def _merge_values(*groups: Sequence[str]) -> tuple[str, ...]:
    normalized = (
        value.strip()
        for group in groups
        for value in group
        if value and value.strip()
    )
    return tuple(dict.fromkeys(normalized))
