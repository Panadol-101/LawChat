from __future__ import annotations

from datetime import date

from retrieval import (
    DenseSearchResult,
    HydratedLegalChunk,
    LegalQueryParser,
    LegalRetrievalService,
    RetrievalRequest,
)


class FakeSearcher:
    def __init__(self):
        self.limits = []
        self.filters = []

    def search(self, query, *, limit, score_threshold=None, filters=None):
        self.limits.append(limit)
        self.filters.append(filters)
        candidates = [
            DenseSearchResult(str(index), 1 - index / 10, {"chunk_id": str(index)})
            for index in range(1, 5)
        ]
        return candidates[:limit]


class FakeHydrator:
    def __init__(self):
        self.filters = []

    def hydrate(self, point_ids, filters, *, collection=None):
        self.filters.append(filters)
        # Point 1 simulates a high-scoring but legally invalid candidate.
        return [_chunk(point_id) for point_id in point_ids if point_id != "1"]


class DiagnosticFakeHydrator(FakeHydrator):
    def rejection_reasons(self, point_ids, filters, *, collection=None):
        del filters, collection
        return {"UNRESOLVED_PROVISION_STATUS": int("1" in point_ids)}


def _chunk(point_id: str) -> HydratedLegalChunk:
    return HydratedLegalChunk(
        point_id=point_id,
        chunk_id=f"doc::article_36::{point_id}",
        chunk_type="clause",
        text=f"Nội dung hợp lệ {point_id}",
        retrieval_text=None,
        chunk_metadata={"article": "36", "clause": point_id},
        parent_chunk_id="doc::article_36",
        parent_text="Điều 36. Quyền đơn phương chấm dứt hợp đồng.",
        document_id="doc",
        title="Bộ luật Lao động",
        document_number="45/2019/QH14",
        document_type="Bộ luật",
        authority="Quốc hội",
        legal_field="Lao động",
        source_url="https://example.test/labor",
        article="36",
        clause=point_id,
        point=None,
        status="EFFECTIVE",
        valid_from=date(2021, 1, 1),
        valid_to=None,
    )


def test_service_expands_candidates_and_keeps_qdrant_ranking():
    searcher = FakeSearcher()
    hydrator = FakeHydrator()
    service = LegalRetrievalService(
        searcher,
        hydrator,
        physical_collection="legal-v2",
        query_parser=LegalQueryParser(today=lambda: date(2025, 1, 1)),
    )

    response = service.retrieve(
        RetrievalRequest(
            query="Điều 36 Bộ luật Lao động",
            limit=2,
            candidate_limit=2,
            max_candidate_limit=4,
        )
    )

    assert searcher.limits == [2, 4]
    assert [item.point_id for item in response.results] == ["2", "3"]
    assert response.rejected_candidates == 1
    assert response.results[0].citation.label == "Điều 36 Khoản 2, 45/2019/QH14"
    assert "Đoạn liên quan:\nNội dung hợp lệ 2" in response.results[0].context_text
    assert hydrator.filters[-1].as_of == date(2025, 1, 1)


def test_service_reports_unresolved_provision_rejection_reason():
    service = LegalRetrievalService(
        FakeSearcher(),
        DiagnosticFakeHydrator(),
        query_parser=LegalQueryParser(today=lambda: date(2025, 1, 1)),
    )

    response = service.retrieve(
        RetrievalRequest(
            query="Điều 19 Nghị định 95/2013/NĐ-CP quy định gì?",
            limit=2,
            candidate_limit=2,
            max_candidate_limit=2,
        )
    )

    assert response.rejection_reasons == {"UNRESOLVED_PROVISION_STATUS": 1}
    assert "UNRESOLVED_PROVISION_STATUS" in response.warnings


def test_service_uses_postgres_for_status_and_merges_document_number():
    searcher = FakeSearcher()
    hydrator = FakeHydrator()
    service = LegalRetrievalService(searcher, hydrator)

    service.retrieve(
        RetrievalRequest(
            query="Nghị định 123/2024/NĐ-CP quy định gì?",
            as_of=date(2025, 1, 1),
            limit=1,
            candidate_limit=2,
            document_numbers=("45/2019/QH14",),
        )
    )

    assert searcher.filters[0].statuses == ()
    assert searcher.filters[0].document_numbers == (
        "45/2019/QH14",
        "123/2024/NĐ-CP",
    )
    assert "EXPIRED" in hydrator.filters[0].statuses
    assert "EFFECTIVE" in hydrator.filters[0].statuses
