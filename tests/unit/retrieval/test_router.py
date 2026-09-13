from datetime import date

from lawchat.retrieval import (
    RetrievalRequest,
    RetrievalResponse,
    TemporalRetrievalRouter,
)


class Target:
    def __init__(self, name: str):
        self.name = name
        self.calls = 0

    def retrieve(self, request):
        self.calls += 1
        return RetrievalResponse(
            query=request.query,
            semantic_query=request.query,
            as_of=request.as_of or date(2026, 9, 2),
            results=(),
            searched_candidates=0,
            rejected_candidates=0,
            warnings=(self.name,),
        )


def test_router_never_falls_back_to_current_for_historical_content():
    current = Target("current")
    response = TemporalRetrievalRouter(current).retrieve(
        RetrievalRequest(
            query="Vào ngày 01/05/2015 quy định này như thế nào?",
            as_of=date(2015, 5, 1),
        )
    )
    assert current.calls == 0
    assert not response.historical_content_available
    assert response.results == ()


def test_router_uses_dedicated_historical_target():
    current = Target("current")
    historical = Target("historical")
    response = TemporalRetrievalRouter(current, historical).retrieve(
        RetrievalRequest(
            query="Tại thời điểm ngày 01/05/2015 quy định gì?",
            as_of=date(2015, 5, 1),
        )
    )
    assert current.calls == 0
    assert historical.calls == 1
    assert response.warnings == ("historical",)


def test_status_lookup_with_as_of_uses_metadata_capable_current_target():
    current = Target("current")
    TemporalRetrievalRouter(current).retrieve(
        RetrievalRequest(
            query="Văn bản 12/2013/QĐ-UBND còn hiệu lực không?",
            as_of=date(2015, 5, 1),
        )
    )
    assert current.calls == 1
