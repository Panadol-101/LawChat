import asyncio
from types import SimpleNamespace
from datetime import date

import pytest
from fastapi import HTTPException

from api.main import (
    SearchBody,
    open_document_source,
    search,
)
from retrieval import RetrievalResponse, UnsupportedAsOfDate
from sources import SourceResolution

ADMIN = SimpleNamespace(role="ADMIN")


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
            limit=5,
            candidate_limit=25,
            statuses=["EXPIRED"],
        ),
        service,
        ADMIN,
    )

    assert service.request.statuses == ("EXPIRED",)
    assert service.request.limit == 5
    assert service.request.as_of is None
    assert response["as_of"] == "2025-01-01"


def test_search_endpoint_rejects_bad_candidate_limit_or_unsupported_as_of():
    with pytest.raises(HTTPException) as bad_limit:
        search(
            SearchBody(query="test", limit=10, candidate_limit=5),
            FakeService(),
            ADMIN,
        )
    assert bad_limit.value.status_code == 422

    with pytest.raises(HTTPException) as unsupported:
        search(
            SearchBody(query="test"),
            FakeService(error=UnsupportedAsOfDate("historical as_of")),
            ADMIN,
        )
    assert unsupported.value.status_code == 422


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


def test_search_endpoint_rejects_historical_as_of():
    with pytest.raises(HTTPException) as historical:
        search(
            SearchBody(query="test", as_of=date(2015, 1, 1)),
            FakeService(),
            ADMIN,
        )
    assert historical.value.status_code == 422
