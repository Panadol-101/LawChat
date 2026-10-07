from __future__ import annotations

from datetime import date

from retrieval import (
    DenseSearchResult,
    HydratedLegalChunk,
    HybridRetrievalService,
    LegalQueryParser,
    LegalGraphEdge,
    LegalGraphDocument,
    LegalGraphExpansion,
    RetrievalCandidate,
    RetrievalRequest,
    RerankResult,
    RerankerSettings,
    SeedResolution,
)
from retrieval.hybrid import _promote_confident_seed_match


class FakeDense:
    def __init__(self):
        self.filters = None
        self.filter_history = []

    def search(self, query, *, limit, filters=None):
        self.filters = filters
        self.filter_history.append(filters)
        return [
            DenseSearchResult("a", 0.9, {"chunk_id": "chunk-a"}),
            DenseSearchResult("b", 0.8, {"chunk_id": "chunk-b"}),
        ][:limit]


class FakeSparse:
    def __init__(self, *, error=None):
        self.error = error
        self.filters = None
        self.filter_history = []

    def search(self, query, *, limit, filters=None):
        self.filters = filters
        self.filter_history.append(filters)
        if self.error:
            raise self.error
        return [
            RetrievalCandidate("b", "chunk-b", "sparse", 12.0, 1),
            RetrievalCandidate("c", "chunk-c", "sparse", 10.0, 2),
        ][:limit]


class FakeHydrator:
    def __init__(self):
        self.filters = None

    def hydrate(self, point_ids, filters, *, collection=None):
        self.filters = filters
        return [_hydrated(point_id) for point_id in point_ids]


class FakeGraph:
    def expand(self, document_numbers, *, relationship_types, direction, as_of):
        assert document_numbers == ("07/2025/QĐ-UBND",)
        assert relationship_types == ("REPLACES",)
        assert direction == "outgoing"
        return LegalGraphExpansion(
            document_ids=("target-doc",),
            document_numbers=("18/2024/QĐ-UBND",),
            statuses=("EXPIRED",),
            edges=(LegalGraphEdge("source-doc", "target-doc", "REPLACES"),),
            related_documents=(
                LegalGraphDocument(
                    document_id="target-doc",
                    document_number="18/2024/QĐ-UBND",
                    title="Văn bản cũ",
                    status="EXPIRED",
                    role="replaced_document",
                    relationship_type="REPLACES",
                    direction="outgoing",
                ),
            ),
        )


class FakeStatusGraph:
    def expand(self, document_numbers, *, relationship_types, direction, as_of):
        assert relationship_types == ("REPLACES", "REPEALS")
        assert direction == "incoming"
        return LegalGraphExpansion(document_ids=("current-target",))


class FakeSeedResolver:
    def resolve_seeds(self, document_numbers, *, as_of, query, preferred_statuses=()):
        return (
            SeedResolution(
                LegalGraphDocument(
                    document_id="source-doc",
                    document_number=document_numbers[0],
                    title="Văn bản nguồn",
                    status="EFFECTIVE",
                    role="queried_document",
                ),
                score=105.0,
                matched_fields=("document_number", "status"),
            ),
        )


class FakeExpiredSeedResolver:
    def resolve_seeds(self, document_numbers, *, as_of, query, preferred_statuses=()):
        return (
            SeedResolution(
                LegalGraphDocument(
                    document_id="expired-seed",
                    document_number=document_numbers[0],
                    title="Văn bản hết hiệu lực",
                    status="EXPIRED",
                    role="queried_document",
                    valid_from=date(2020, 1, 1),
                ),
                score=105.0,
                matched_fields=("document_number", "status"),
            ),
        )


class FakeAmbiguousSeedResolver:
    def resolve_seeds(self, document_numbers, *, as_of, query, preferred_statuses=()):
        return tuple(
            SeedResolution(
                LegalGraphDocument(
                    document_id=document_id,
                    document_number=document_numbers[0],
                    title=f"Văn bản {document_id}",
                    status="EXPIRED",
                    role="queried_document",
                ),
                score=105.0,
                matched_fields=("document_number", "status"),
            )
            for document_id in ("seed-a", "seed-b", "seed-c")
        )


def test_confident_seed_score_promotes_matching_document_after_rerank():
    class Chunk:
        def __init__(self, document_id):
            self.document_id = document_id

    seeds = (
        SeedResolution(
            LegalGraphDocument(
                "preferred", "14/2002/QĐ-UB", "Sở Giao thông - Vận tải",
                "EXPIRED", "queried_document",
            ),
            score=136.8,
            matched_fields=("document_number", "title", "status"),
        ),
        SeedResolution(
            LegalGraphDocument(
                "other", "14/2002/QĐ-UB", "Văn bản khác",
                "EXPIRED", "queried_document",
            ),
            score=133.5,
            matched_fields=("document_number", "status"),
        ),
    )

    ordered, promoted = _promote_confident_seed_match(
        [Chunk("other"), Chunk("preferred")], seeds
    )

    assert promoted
    assert [item.document_id for item in ordered] == ["preferred", "other"]


class ReverseReranker:
    def rerank(self, query, chunks):
        return [
            RerankResult(chunk.point_id, float(index), rank)
            for rank, (index, chunk) in enumerate(
                reversed(list(enumerate(chunks, start=1))),
                start=1,
            )
        ]


class FailingReranker:
    def rerank(self, query, chunks):
        raise RuntimeError("model timeout")


def _hydrated(point_id):
    return HydratedLegalChunk(
        point_id=point_id,
        chunk_id=f"chunk-{point_id}",
        chunk_type="clause",
        text=f"Nội dung {point_id}",
        retrieval_text=None,
        chunk_metadata={},
        parent_chunk_id=None,
        parent_text=None,
        document_id=f"doc-{point_id}",
        title="Văn bản kiểm thử",
        document_number="100/2019/NĐ-CP",
        document_type="Nghị định",
        authority="Chính phủ",
        legal_field="Giao thông",
        source_url="https://example.test",
        article="5",
        clause="2",
        point=None,
        status="EFFECTIVE",
        valid_from=date(2020, 1, 1),
        valid_to=None,
    )


def _service(sparse):
    return HybridRetrievalService(
        FakeDense(),
        sparse,
        FakeHydrator(),
        query_parser=LegalQueryParser(today=lambda: date(2025, 1, 1)),
    )


def test_hybrid_service_requires_explicit_legal_structure_in_both_indexes():
    sparse = FakeSparse()
    service = _service(sparse)

    service.retrieve(
        RetrievalRequest(
            query="Điều 5 Khoản 2 Điểm c Nghị định 100/2019/NĐ-CP",
            limit=1,
            candidate_limit=2,
        )
    )

    assert service.dense_searcher.filters.articles == ("5",)
    assert service.dense_searcher.filters.clauses == ("2",)
    assert service.dense_searcher.filters.points == ("c",)
    assert sparse.filters.require_structure
    assert sparse.filters.article_hints == ("5",)
    assert sparse.filters.clause_hints == ("2",)
    assert sparse.filters.point_hints == ("c",)


def test_hybrid_service_fuses_sources_and_preserves_diagnostics():
    response = _service(FakeSparse()).retrieve(
        RetrievalRequest(query="quy định giao thông", limit=2, candidate_limit=2)
    )

    assert [item.point_id for item in response.results] == ["b", "a"]
    assert response.retrieval_sources == ("dense", "sparse")
    assert response.results[0].source_ranks == {"dense": 2, "sparse": 1}
    assert response.warnings == ()


def test_hybrid_service_falls_back_when_sparse_fails():
    response = _service(FakeSparse(error=RuntimeError("offline"))).retrieve(
        RetrievalRequest(query="quy định giao thông", limit=1, candidate_limit=2)
    )

    assert response.results[0].point_id == "a"
    assert response.retrieval_sources == ("dense",)
    assert any("sparse retrieval failed" in item for item in response.warnings)


def test_hybrid_service_replaces_source_filter_with_graph_target():
    dense = FakeDense()
    sparse = FakeSparse()
    hydrator = FakeHydrator()
    service = HybridRetrievalService(
        dense,
        sparse,
        hydrator,
        graph_resolver=FakeGraph(),
        seed_resolver=FakeSeedResolver(),
        query_parser=LegalQueryParser(today=lambda: date(2026, 8, 27)),
    )

    response = service.retrieve(
        RetrievalRequest(
            query="Văn bản 07/2025/QĐ-UBND thay thế văn bản nào?",
            limit=1,
            candidate_limit=2,
        )
    )

    assert dense.filters.doc_ids == ("target-doc",)
    assert dense.filters.document_numbers == ()
    assert sparse.filters.doc_ids == ("target-doc",)
    assert hydrator.filters.document_ids == ("target-doc",)
    assert hydrator.filters.document_numbers == ()
    assert hydrator.filters.statuses == ("EXPIRED",)
    assert response.retrieval_sources == ("graph", "dense", "sparse")
    assert response.seed_documents[0].role == "queried_document"
    assert response.related_documents[0].role == "replaced_document"
    assert response.graph_edges[0].relationship_type == "REPLACES"
    assert any("legal graph expanded" in item for item in response.warnings)


def test_hybrid_service_reranks_hydrated_candidates_and_records_score():
    service = HybridRetrievalService(
        FakeDense(),
        FakeSparse(),
        FakeHydrator(),
        reranker=ReverseReranker(),
        reranker_settings=RerankerSettings(candidate_limit=2),
        query_parser=LegalQueryParser(today=lambda: date(2025, 1, 1)),
    )

    response = service.retrieve(
        RetrievalRequest(query="quy định giao thông", limit=2, candidate_limit=2)
    )

    assert [item.point_id for item in response.results] == ["a", "b"]
    assert response.retrieval_sources[-1] == "reranker"
    assert response.results[0].source_scores["reranker"] == 2.0


def test_status_lookup_keeps_seed_as_target_when_quoted_text_mentions_repeal():
    dense = FakeDense()
    hydrator = FakeHydrator()
    service = HybridRetrievalService(
        dense,
        FakeSparse(),
        hydrator,
        graph_resolver=FakeStatusGraph(),
        seed_resolver=FakeExpiredSeedResolver(),
        query_parser=LegalQueryParser(today=lambda: date(2026, 8, 27)),
    )

    response = service.retrieve(
        RetrievalRequest(
            query=(
                "Văn bản 07/2025/QĐ-UBND từng quy định các nội dung cũ bị "
                "bãi bỏ hiện còn hiệu lực không?"
            ),
            limit=1,
            candidate_limit=2,
            statuses=("EXPIRED",),
        )
    )

    assert dense.filters.doc_ids == ("expired-seed",)
    assert hydrator.filters.document_ids == ("expired-seed",)
    assert response.temporal_intent == "status_lookup"
    assert response.status_resolution is not None
    assert response.status_resolution.status == "EXPIRED"


def test_status_lookup_searches_each_ambiguous_seed_independently():
    dense = FakeDense()
    sparse = FakeSparse()
    service = HybridRetrievalService(
        dense,
        sparse,
        FakeHydrator(),
        seed_resolver=FakeAmbiguousSeedResolver(),
        query_parser=LegalQueryParser(today=lambda: date(2026, 8, 27)),
    )

    response = service.retrieve(
        RetrievalRequest(
            query="Văn bản 12/2013/QĐ-UBND hiện còn hiệu lực không?",
            limit=1,
            candidate_limit=6,
            statuses=("EXPIRED",),
        )
    )

    assert [item.doc_ids for item in dense.filter_history] == [
        ("seed-a",),
        ("seed-b",),
        ("seed-c",),
    ]
    assert [item.doc_ids for item in sparse.filter_history] == [
        ("seed-a",),
        ("seed-b",),
        ("seed-c",),
    ]
    assert response.status_resolution is not None
    assert response.status_resolution.ambiguous


def test_hybrid_service_falls_back_to_rrf_when_reranker_fails():
    service = HybridRetrievalService(
        FakeDense(),
        FakeSparse(),
        FakeHydrator(),
        reranker=FailingReranker(),
        query_parser=LegalQueryParser(today=lambda: date(2025, 1, 1)),
    )

    response = service.retrieve(
        RetrievalRequest(query="quy định giao thông", limit=2, candidate_limit=2)
    )

    assert [item.point_id for item in response.results] == ["b", "a"]
    assert any("reranker failed" in item for item in response.warnings)


class SlowReranker:
    def rerank(self, query, chunks):
        import time

        time.sleep(0.5)
        return ReverseReranker().rerank(query, chunks)


def test_hybrid_service_falls_back_to_rrf_when_reranker_times_out():
    service = HybridRetrievalService(
        FakeDense(),
        FakeSparse(),
        FakeHydrator(),
        reranker=SlowReranker(),
        reranker_settings=RerankerSettings(candidate_limit=2, timeout_seconds=0.05),
        query_parser=LegalQueryParser(today=lambda: date(2025, 1, 1)),
    )

    response = service.retrieve(
        RetrievalRequest(query="quy định giao thông", limit=2, candidate_limit=2)
    )

    assert response.reranker_status == "timeout"
    assert [item.point_id for item in response.results] == ["b", "a"]


class FakeMixedSeedResolver:
    def resolve_seeds(self, document_numbers, *, as_of, query, preferred_statuses=()):
        return tuple(
            SeedResolution(
                LegalGraphDocument(
                    document_id=document_id,
                    document_number=number,
                    title=document_id,
                    status=status,
                    role="queried_document",
                ),
                score=105.0,
                matched_fields=("document_number",),
            )
            for document_id, number, status in (
                ("old-code", "10/2012/QH13", "EXPIRED"),
                ("current-code", "45/2019/QH14", "PARTIALLY_EFFECTIVE"),
            )
        )


def test_replacement_expansion_keeps_in_force_seed_documents():
    dense = FakeDense()
    service = HybridRetrievalService(
        dense,
        FakeSparse(),
        FakeHydrator(),
        graph_resolver=FakeStatusGraph(),
        seed_resolver=FakeMixedSeedResolver(),
        query_parser=LegalQueryParser(today=lambda: date(2026, 8, 27)),
    )

    service.retrieve(
        RetrievalRequest(
            query="BLLĐ 2012 quy định thời giờ làm việc thế nào?",
            limit=1,
            candidate_limit=2,
        )
    )

    assert dense.filters.doc_ids == ("current-code", "current-target")


class StructureOnlyEmptyDense(FakeDense):
    """Returns nothing while an article filter is applied (document has no article metadata)."""

    def search(self, query, *, limit, filters=None):
        if filters is not None and filters.articles:
            self.filters = filters
            self.filter_history.append(filters)
            return []
        return super().search(query, limit=limit, filters=filters)


class StructureOnlyEmptySparse(FakeSparse):
    def search(self, query, *, limit, filters=None):
        if filters is not None and filters.require_structure:
            self.filters = filters
            self.filter_history.append(filters)
            return []
        return super().search(query, limit=limit, filters=filters)


def test_provision_filter_with_no_match_falls_back_to_the_named_document():
    dense, sparse = StructureOnlyEmptyDense(), StructureOnlyEmptySparse()
    service = HybridRetrievalService(
        dense,
        sparse,
        FakeHydrator(),
        query_parser=LegalQueryParser(today=lambda: date(2025, 1, 1)),
        seed_resolver=FakeSeedResolver(),
    )

    response = service.retrieve(
        RetrievalRequest(query="Điều 17 Nghị định 100/2019/NĐ-CP", limit=2, candidate_limit=2)
    )

    assert response.results
    assert dense.filter_history[0].articles == ("17",)
    assert dense.filters.articles == ()
    assert dense.filters.doc_ids == ("source-doc",)
    assert not sparse.filters.require_structure
    assert any("provision" in warning for warning in response.warnings)


def test_provision_fallback_is_not_used_without_a_resolved_document():
    dense, sparse = StructureOnlyEmptyDense(), StructureOnlyEmptySparse()
    service = HybridRetrievalService(
        dense,
        sparse,
        FakeHydrator(),
        query_parser=LegalQueryParser(today=lambda: date(2025, 1, 1)),
    )

    response = service.retrieve(
        RetrievalRequest(query="Điều 17 quy định gì", limit=2, candidate_limit=2)
    )

    assert response.results == ()
    assert all(item.articles == ("17",) for item in dense.filter_history)
