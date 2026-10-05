import asyncio
from datetime import date

import pytest
from fastapi import HTTPException

from api.main import (
    SearchBody,
    open_document_source,
    search,
)
from retrieval import AmbiguousTemporalQuery, RetrievalResponse
from sources import SourceResolution


class FakeService:
    def __init__(self, *, error=None):
        self.request = None
        self.error = error

    def retrieve(self, request):
        self.request = request
        if self.error:
            raise self.error
        return RetrievalResponse(
            query=request.query,
            semantic_query=request.query,
            as_of=request.as_of or date(2025, 1, 1),
            results=(),
            searched_candidates=0,
            rejected_candidates=0,
        )


def test_search_endpoint_builds_temporal_retrieval_request():
    service = FakeService()

    response = search(
        SearchBody(
            query="Quy định lao động",
            as_of=date(2025, 1, 1),
            limit=5,
            candidate_limit=25,
            statuses=["EXPIRED"],
        ),
        service,
    )

    assert service.request.statuses == ("EXPIRED",)
    assert service.request.limit == 5
    assert response["as_of"] == "2025-01-01"


def test_search_endpoint_rejects_bad_candidate_limit_or_ambiguous_date():
    with pytest.raises(HTTPException) as bad_limit:
        search(
            SearchBody(query="test", limit=10, candidate_limit=5),
            FakeService(),
        )
    assert bad_limit.value.status_code == 422

    with pytest.raises(HTTPException) as ambiguous:
        search(
            SearchBody(query="test"),
            FakeService(error=AmbiguousTemporalQuery("multiple dates")),
        )
    assert ambiguous.value.status_code == 422


class FakeSourceResolutionService:
    def __init__(self):
        self.document_id = None

    async def resolve(self, document_id):
        self.document_id = document_id
        return SourceResolution(
            url="https://example.test/full-text",
            status="VERIFIED",
            document_number="100/2019/NĐ-CP",
        )


@pytest.mark.parametrize(
    "document_id",
    ["140152", "da85a486-ea3c-46e4-b312-94c75a27140a"],
)
def test_open_document_source_accepts_external_id_and_uuid(document_id):
    service = FakeSourceResolutionService()
    response = asyncio.run(open_document_source(document_id, service))

    assert response.status_code == 302
    assert response.headers["location"] == "https://example.test/full-text"
    assert response.headers["x-lawchat-source-resolution"] == "VERIFIED"
    assert service.document_id == document_id
