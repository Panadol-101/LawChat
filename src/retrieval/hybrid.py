from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import replace
from math import ceil
from time import perf_counter
from typing import Protocol

from database import LegalMetadataFilter

from .fusion import FusedCandidate, RetrievalCandidate, RRFSettings, reciprocal_rank_fusion
from .dense import DenseSearchFilter, DenseSearchResult
from .graph import (
    DocumentStatusResolution,
    LegalGraphResolver,
    LegalSeedResolver,
)
from .hydration import (
    ChunkHydrator,
    LegalContextBuilder,
    hydration_rejection_reasons,
)
from .models import (
    HydratedLegalChunk,
    LegalCitation,
    ParsedLegalQuery,
    RetrievalRequest,
    RetrievalResponse,
    RetrievedLegalChunk,
)
from .query_parser import LegalQueryParser
from .reranker import LegalReranker, RerankerSettings
from .sparse import SparseSearchFilter
from .temporal import TemporalIntent, TemporalPolicy


# One worker: a cross-encoder call that outlives its timeout keeps running
# (CPU work cannot be interrupted), and later calls queue behind it instead of
# multiplying CPU load; they time out and fall back to the RRF order.
_RERANK_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lawchat-rerank")


class RerankerTimeout(RuntimeError):
    pass


class RerankerError(RuntimeError):
    pass


class HybridSearchUnavailable(RuntimeError):
    pass


class DenseCandidateSearcher(Protocol):
    def search(
        self,
        query: str,
        *,
        limit: int,
        filters: DenseSearchFilter | None = None,
    ) -> list[DenseSearchResult]: ...


class SparseCandidateSearcher(Protocol):
    def search(
        self,
        query: str,
        *,
        limit: int,
        filters: SparseSearchFilter | None = None,
    ) -> list[RetrievalCandidate]: ...


class HybridRetrievalService:
    def __init__(
        self,
        dense_searcher: DenseCandidateSearcher,
        sparse_searcher: SparseCandidateSearcher | None,
        hydrator: ChunkHydrator,
        *,
        physical_collection: str | None = None,
        rrf_settings: RRFSettings | None = None,
        query_parser: LegalQueryParser | None = None,
        context_builder: LegalContextBuilder | None = None,
        graph_resolver: LegalGraphResolver | None = None,
        seed_resolver: LegalSeedResolver | None = None,
        temporal_policy: TemporalPolicy | None = None,
        reranker: LegalReranker | None = None,
        reranker_settings: RerankerSettings | None = None,
    ) -> None:
        self.dense_searcher = dense_searcher
        self.sparse_searcher = sparse_searcher
        self.hydrator = hydrator
        self.physical_collection = physical_collection
        self.rrf_settings = rrf_settings or RRFSettings.from_env()
        self.query_parser = query_parser or LegalQueryParser()
        self.context_builder = context_builder or LegalContextBuilder()
        self.graph_resolver = graph_resolver
        self.seed_resolver = seed_resolver
        self.temporal_policy = temporal_policy or TemporalPolicy()
        self.reranker = reranker
        self.reranker_settings = reranker_settings or RerankerSettings.from_env()

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        total_started = perf_counter()
        parse_started = perf_counter()
        parsed = self.query_parser.parse(request.query, as_of=request.as_of)
        timings: dict[str, float] = {
            "query_parse": perf_counter() - parse_started,
            "dense": 0.0,
            "sparse": 0.0,
            "fusion": 0.0,
            "hydration": 0.0,
            "reranker": 0.0,
        }
        graph_started = perf_counter()
        document_numbers = _merge_values(
            request.document_numbers,
            parsed.document_numbers,
        )
        temporal = self.temporal_policy.decide(parsed, request)
        seed_resolutions = ()
        if self.seed_resolver is not None and document_numbers:
            seed_resolutions = self.seed_resolver.resolve_seeds(
                document_numbers,
                as_of=parsed.as_of,
                query=parsed.original_query,
                preferred_statuses=temporal.allowed_statuses,
            )
        seed_documents = tuple(item.document for item in seed_resolutions)

        relationship_types = parsed.relationship_types
        relationship_direction = parsed.relationship_direction
        inactive_seed = any(
            seed.status not in {"EFFECTIVE", "PARTIALLY_EFFECTIVE"}
            for seed in seed_documents
        )
        if temporal.intent is TemporalIntent.STATUS_LOOKUP:
            # Words such as "bãi bỏ" often occur inside quoted document text.
            # A status question must keep the explicitly named document as the
            # retrieval target; graph expansion is explanatory metadata only.
            relationship_types = ("REPLACES", "REPEALS")
            relationship_direction = "incoming"
        elif not relationship_types and temporal.expand_replacements and inactive_seed:
            relationship_types = ("REPLACES", "REPEALS")
            relationship_direction = "incoming"
        graph_expansion = None
        if (
            self.graph_resolver is not None
            and document_numbers
            and relationship_types
            and relationship_direction
        ):
            graph_expansion = self.graph_resolver.expand(
                document_numbers,
                relationship_types=relationship_types,
                direction=relationship_direction,
                as_of=parsed.as_of,
            )
        graph_document_ids = graph_expansion.document_ids if graph_expansion else ()
        timings["seed_and_graph"] = perf_counter() - graph_started
        explicit_relationship_query = bool(parsed.relationship_types) and (
            temporal.intent is not TemporalIntent.STATUS_LOOKUP
        )
        seed_document_ids = tuple(seed.document_id for seed in seed_documents)
        if graph_document_ids and explicit_relationship_query:
            search_document_ids = graph_document_ids
        elif (
            graph_document_ids
            and inactive_seed
            and temporal.intent is TemporalIntent.CURRENT_LAW
        ):
            # Replace only the inactive seeds: an in-force document the user
            # also named (e.g. the current code beside its predecessor) stays.
            active_seed_ids = tuple(
                seed.document_id
                for seed in seed_documents
                if seed.status in {"EFFECTIVE", "PARTIALLY_EFFECTIVE"}
            )
            search_document_ids = tuple(
                dict.fromkeys((*active_seed_ids, *graph_document_ids))
            )
        else:
            search_document_ids = seed_document_ids
        search_document_numbers = () if search_document_ids else document_numbers
        effective_statuses = temporal.allowed_statuses
        if (
            request.statuses is None
            and graph_expansion
            and graph_expansion.statuses
            and explicit_relationship_query
        ):
            effective_statuses = graph_expansion.statuses
        metadata_filters = LegalMetadataFilter(
            as_of=parsed.as_of,
            statuses=effective_statuses,
            document_numbers=search_document_numbers,
            document_ids=search_document_ids,
            document_types=request.document_types,
            authorities=request.authorities,
            legal_fields=request.legal_fields,
            temporal_intent=temporal.intent.value,
        )
        dense_filters = DenseSearchFilter(
            doc_ids=search_document_ids,
            document_numbers=search_document_numbers,
            document_types=request.document_types,
            authorities=request.authorities,
            legal_fields=request.legal_fields,
            articles=parsed.referenced_articles,
            clauses=parsed.referenced_clauses,
            points=parsed.referenced_points,
        )
        sparse_filters = SparseSearchFilter(
            doc_ids=search_document_ids,
            document_numbers=search_document_numbers,
            document_types=request.document_types,
            authorities=request.authorities,
            legal_fields=request.legal_fields,
            article_hints=parsed.referenced_articles,
            clause_hints=parsed.referenced_clauses,
            point_hints=parsed.referenced_points,
            require_structure=bool(
                parsed.referenced_articles
                or parsed.referenced_clauses
                or parsed.referenced_points
            ),
        )

        candidate_limit = request.candidate_limit
        warnings: list[str] = list(temporal.warnings)
        if document_numbers and self.seed_resolver is None:
            warnings.append("legal seed resolver is unavailable")
        elif document_numbers and not seed_documents:
            warnings.append("no exact seed document was resolved")
        if graph_expansion and graph_document_ids:
            warnings.append(
                f"legal graph expanded to {len(graph_document_ids)} related document(s)"
            )
        elif relationship_types and self.graph_resolver is None:
            warnings.append("legal graph is unavailable")
        elif graph_expansion is not None and not graph_document_ids:
            warnings.append("legal graph found no resolved related documents")
        fused: list[FusedCandidate] = []
        hydrated: list[HydratedLegalChunk] = []
        sources: tuple[str, ...] = ()
        while True:
            if temporal.intent is TemporalIntent.STATUS_LOOKUP and len(seed_document_ids) > 1:
                rankings, iteration_warnings, source_timings = self._search_sources_by_seed(
                    parsed,
                    candidate_limit,
                    dense_filters,
                    sparse_filters,
                    seed_document_ids,
                )
            else:
                rankings, iteration_warnings, source_timings = self._search_sources(
                    parsed,
                    candidate_limit,
                    dense_filters,
                    sparse_filters,
                )
            warnings.extend(iteration_warnings)
            for name, elapsed in source_timings.items():
                timings[name] = timings.get(name, 0.0) + elapsed
            sources = (
                ("graph", *rankings)
                if graph_expansion and graph_document_ids
                else tuple(rankings)
            )
            if not rankings:
                raise HybridSearchUnavailable("dense and sparse retrieval failed")
            fusion_started = perf_counter()
            fused = reciprocal_rank_fusion(
                rankings,
                settings=self.rrf_settings,
                limit=candidate_limit,
            )
            timings["fusion"] += perf_counter() - fusion_started
            hydration_started = perf_counter()
            hydrated = self.hydrator.hydrate(
                [candidate.point_id for candidate in fused],
                metadata_filters,
                collection=self.physical_collection,
            )
            timings["hydration"] += perf_counter() - hydration_started
            exhausted = all(
                len(candidates) < candidate_limit
                for candidates in rankings.values()
            )
            if (
                len(hydrated) >= request.limit
                or exhausted
                or candidate_limit >= request.max_candidate_limit
            ):
                break
            # Audit fix W5 (Phase 5): the legacy ``candidate_limit * 2`` rule
            # doubles every retry, which is too aggressive when the corpus
            # has many REPEALED/EXPIRED candidates that will keep getting
            # filtered out. We now bump by +50% (floor 8) until we reach
            # ``max_candidate_limit``, keeping the per-iteration growth
            # predictable while still converging in 2-3 iterations.
            candidate_limit = min(
                request.max_candidate_limit,
                max(candidate_limit + 8, int(candidate_limit * 1.5)),
            )

        rejection_reasons = hydration_rejection_reasons(
            self.hydrator,
            [candidate.point_id for candidate in fused],
            hydrated,
            metadata_filters,
            collection=self.physical_collection,
        )
        warnings.extend(rejection_reasons)

        fused_by_point = {candidate.point_id: candidate for candidate in fused}
        ranked_hydrated = hydrated
        reranker_scores: dict[str, float] = {}
        reranker_status = "skipped"
        retrieval_quality = "ok"
        if self.reranker is not None and len(hydrated) > 1:
            rerank_candidates = hydrated[: self.reranker_settings.candidate_limit]
            try:
                reranker_started = perf_counter()
                future = _RERANK_EXECUTOR.submit(
                    self.reranker.rerank,
                    parsed.semantic_query,
                    rerank_candidates,
                )
                timeout = self.reranker_settings.timeout_seconds
                try:
                    reranked = future.result(timeout=timeout)
                except FutureTimeout as exc:
                    future.cancel()
                    raise RerankerTimeout(f"exceeded {timeout:g}s") from exc
                chunk_by_point = {
                    chunk.point_id: chunk for chunk in rerank_candidates
                }
                ranked_hydrated = [
                    chunk_by_point[item.point_id] for item in reranked
                ] + hydrated[len(rerank_candidates):]
                reranker_scores = {
                    item.point_id: item.score for item in reranked
                }
                sources = (*sources, "reranker")
                timings["reranker"] = perf_counter() - reranker_started
                reranker_status = "ok"
                retrieval_quality = "ok"
            except RerankerTimeout as exc:
                timings["reranker"] = perf_counter() - reranker_started
                warnings.append(f"reranker timed out: {exc}")
                reranker_status = "timeout"
                retrieval_quality = "degraded"
            except RerankerError as exc:
                timings["reranker"] = perf_counter() - reranker_started
                warnings.append(f"reranker failed: {exc}")
                reranker_status = "failed_permanent"
                retrieval_quality = "degraded"
            except Exception as exc:
                timings["reranker"] = perf_counter() - reranker_started
                warnings.append(f"reranker failed; using RRF order: {exc}")
                reranker_status = "failed_permanent"
                retrieval_quality = "degraded"
        if temporal.intent is TemporalIntent.STATUS_LOOKUP:
            ranked_hydrated, promoted = _promote_confident_seed_match(
                ranked_hydrated, seed_resolutions
            )
            if promoted:
                warnings.append(
                    "ambiguous document number resolved using seed title/authority score"
                )
        results = tuple(
            self._to_result(
                chunk,
                fused_by_point[chunk.point_id],
                parsed.as_of,
                reranker_score=reranker_scores.get(chunk.point_id),
            )
            for chunk in ranked_hydrated[: request.limit]
        )
        status_resolution = _resolve_status_metadata(
            parsed.as_of,
            seed_resolutions,
            results,
            graph_expansion.related_documents if graph_expansion else (),
        ) if temporal.intent is TemporalIntent.STATUS_LOOKUP else None
        timings["total"] = perf_counter() - total_started
        return RetrievalResponse(
            query=parsed.original_query,
            semantic_query=parsed.semantic_query,
            as_of=parsed.as_of,
            results=results,
            searched_candidates=len(fused),
            rejected_candidates=len(fused) - len(hydrated),
            rejection_reasons=rejection_reasons,
            retrieval_sources=sources,
            warnings=tuple(dict.fromkeys(warnings)),
            temporal_intent=temporal.intent.value,
            temporal_explicit_as_of=temporal.explicit_as_of,
            seed_documents=seed_documents,
            seed_resolutions=seed_resolutions,
            status_resolution=status_resolution,
            related_documents=(
                graph_expansion.related_documents if graph_expansion else ()
            ),
            graph_edges=graph_expansion.edges if graph_expansion else (),
            timings={key: round(value, 6) for key, value in timings.items()},
            reranker_status=reranker_status,
            retrieval_quality=retrieval_quality,
        )
    def _search_sources_by_seed(
        self,
        parsed: ParsedLegalQuery,
        limit: int,
        dense_filters: DenseSearchFilter,
        sparse_filters: SparseSearchFilter,
        seed_document_ids: tuple[str, ...],
    ) -> tuple[dict[str, list[RetrievalCandidate]], list[str], dict[str, float]]:
        per_seed_limit = max(2, ceil(limit / len(seed_document_ids)))
        collected: dict[str, list[tuple[int, RetrievalCandidate]]] = {}
        warnings: list[str] = []
        timings: dict[str, float] = {"dense": 0.0, "sparse": 0.0}
        for seed_order, document_id in enumerate(seed_document_ids):
            rankings, seed_warnings, seed_timings = self._search_sources(
                parsed,
                per_seed_limit,
                replace(dense_filters, doc_ids=(document_id,), document_numbers=()),
                replace(sparse_filters, doc_ids=(document_id,), document_numbers=()),
            )
            warnings.extend(seed_warnings)
            for name, elapsed in seed_timings.items():
                timings[name] = timings.get(name, 0.0) + elapsed
            for source, candidates in rankings.items():
                collected.setdefault(source, []).extend(
                    (seed_order, candidate) for candidate in candidates
                )

        merged: dict[str, list[RetrievalCandidate]] = {}
        for source, candidates in collected.items():
            ordered = sorted(
                candidates,
                key=lambda item: (
                    item[0],
                    -item[1].raw_score,
                    item[1].rank,
                    item[1].point_id,
                ),
            )[:limit]
            merged[source] = [
                replace(candidate, rank=rank)
                for rank, (_seed_order, candidate) in enumerate(ordered, start=1)
            ]
        return merged, warnings, timings

    def _search_sources(
        self,
        parsed: ParsedLegalQuery,
        limit: int,
        dense_filters: DenseSearchFilter,
        sparse_filters: SparseSearchFilter,
    ) -> tuple[dict[str, list[RetrievalCandidate]], list[str], dict[str, float]]:
        warnings: list[str] = []
        if self.sparse_searcher is None:
            try:
                dense, dense_seconds = _timed_call(
                    self.dense_searcher.search,
                    parsed.semantic_query,
                    limit=limit,
                    filters=dense_filters,
                )
            except Exception as exc:
                return {}, [f"dense retrieval failed: {exc}"], {}
            return {"dense": _dense_candidates(dense)}, [
                "sparse index is unavailable; using dense retrieval"
            ], {"dense": dense_seconds}

        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(
                _timed_call,
                self.dense_searcher.search,
                parsed.semantic_query,
                limit=limit,
                filters=dense_filters,
            )
            sparse_future = executor.submit(
                _timed_call,
                self.sparse_searcher.search,
                parsed.semantic_query,
                limit=limit,
                filters=sparse_filters,
            )
            rankings: dict[str, list[RetrievalCandidate]] = {}
            timings: dict[str, float] = {}
            try:
                dense, timings["dense"] = dense_future.result()
                rankings["dense"] = _dense_candidates(dense)
            except Exception as exc:
                warnings.append(f"dense retrieval failed: {exc}")
            try:
                rankings["sparse"], timings["sparse"] = sparse_future.result()
            except Exception as exc:
                warnings.append(f"sparse retrieval failed: {exc}")
        return rankings, warnings, timings

    def _to_result(
        self,
        chunk: HydratedLegalChunk,
        candidate: FusedCandidate,
        as_of,
        *,
        reranker_score: float | None = None,
    ) -> RetrievedLegalChunk:
        citation = LegalCitation(
            document_id=chunk.document_id,
            title=chunk.title,
            document_number=chunk.document_number,
            article=chunk.article,
            clause=chunk.clause,
            point=chunk.point,
            source_url=chunk.source_url,
            as_of=as_of,
            status=chunk.status,
            status_scope=chunk.status_scope,
            version_id=chunk.version_id,
            version_source_url=chunk.version_source_url,
            version_source_revision=chunk.version_source_revision,
            content_valid_from=chunk.content_valid_from,
            content_valid_to=chunk.content_valid_to,
        )
        source_scores = dict(candidate.source_scores)
        if reranker_score is not None:
            source_scores["reranker"] = reranker_score
        return RetrievedLegalChunk(
            point_id=chunk.point_id,
            chunk_id=chunk.chunk_id,
            score=reranker_score if reranker_score is not None else candidate.score,
            text=chunk.text,
            context_text=self.context_builder.build(chunk),
            chunk_type=chunk.chunk_type,
            citation=citation,
            authority=chunk.authority,
            legal_field=chunk.legal_field,
            source_ranks=candidate.source_ranks,
            source_scores=source_scores,
            metadata=chunk.chunk_metadata,
            parent_text=chunk.parent_text,
        )


def _dense_candidates(results: list[DenseSearchResult]) -> list[RetrievalCandidate]:
    return [
        RetrievalCandidate(
            point_id=result.point_id,
            chunk_id=str(result.payload["chunk_id"]),
            source="dense",
            raw_score=result.score,
            rank=rank,
            payload=result.payload,
        )
        for rank, result in enumerate(results, start=1)
    ]


def _timed_call(function, *args, **kwargs):
    started = perf_counter()
    result = function(*args, **kwargs)
    return result, perf_counter() - started


def _merge_values(*groups: tuple[str, ...]) -> tuple[str, ...]:
    normalized = (
        value.strip()
        for group in groups
        for value in group
        if value and value.strip()
    )
    return tuple(dict.fromkeys(normalized))


def _promote_confident_seed_match(chunks, seed_resolutions):
    """Keep a confidently disambiguated exact document in the final window."""
    if len(seed_resolutions) < 2:
        return chunks, False
    best, second = seed_resolutions[:2]
    if best.score - second.score < 1.0:
        return chunks, False
    preferred_id = best.document.document_id
    index = next(
        (
            index for index, chunk in enumerate(chunks)
            if chunk.document_id == preferred_id
        ),
        None,
    )
    if index in {None, 0}:
        return chunks, False
    return [chunks[index], *chunks[:index], *chunks[index + 1:]], True


def _resolve_status_metadata(
    as_of,
    seed_resolutions,
    results,
    related_documents,
) -> DocumentStatusResolution | None:
    if not seed_resolutions:
        return None
    by_document_id = {
        item.document.document_id: item for item in seed_resolutions
    }
    selected = next(
        (
            by_document_id[result.citation.document_id]
            for result in results
            if result.citation.document_id in by_document_id
        ),
        seed_resolutions[0],
    )
    top_score = seed_resolutions[0].score
    tied = tuple(
        item.document.document_id
        for item in seed_resolutions
        if item.score == top_score
    )
    document = selected.document
    return DocumentStatusResolution(
        document_id=document.document_id,
        document_number=document.document_number,
        as_of=as_of,
        status=document.status,
        valid_from=document.valid_from,
        valid_to=document.valid_to,
        ambiguous=len(tied) > 1,
        candidate_document_ids=tuple(
            item.document.document_id for item in seed_resolutions
        ),
        related_documents=related_documents,
    )
