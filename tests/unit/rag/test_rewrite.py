import asyncio
import json
from types import SimpleNamespace

import pytest

from rag import (
    InvalidStructuredResponseError,
    OpenAICompatibleQueryRewriter,
    OpenAICompatibleSettings,
    ProviderUnavailableError,
    QueryRewrite,
    QueryRewriter,
    QueryRewriteRequest,
    history_for_rewrite,
)
from rag.rewrite import preserve_references


def _rewriter(content):
    return OpenAICompatibleQueryRewriter(
        OpenAICompatibleSettings("http://test", "", "gemma"),
        transport=lambda *args: {
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}]
        },
    )


def test_rewriter_parses_standalone_question():
    result = _rewriter('{"standalone_question":"  Đánh bạc trên không gian mạng bị xử phạt thế nào?  "}').generate(
        QueryRewriteRequest("đánh bài online bị phạt sao")
    )
    assert result.standalone_question == "Đánh bạc trên không gian mạng bị xử phạt thế nào?"


@pytest.mark.parametrize(
    "content",
    [
        '{"standalone_question":""}',
        '{"standalone_question":"x","extra":1}',
        '{"question":"x"}',
    ],
)
def test_rewriter_rejects_malformed_output(content):
    with pytest.raises(InvalidStructuredResponseError):
        _rewriter(content).generate(QueryRewriteRequest("Câu hỏi"))


def test_rewriter_payload_carries_history_and_question():
    request = QueryRewriteRequest(
        "vậy nếu từ 5 tỷ đồng trở lên thì phạt như thế nào",
        (("user", "Đánh cờ bạc trên không gian mạng bị xử phạt thế nào?"),
         ("assistant", "Theo Điều 321 Bộ luật Hình sự ...")),
    )
    payload = _rewriter("{}")._payload(request, stream=False)
    body = json.loads(payload["messages"][1]["content"])
    assert body["question"] == request.question
    assert body["history"][0] == {
        "role": "user", "content": "Đánh cờ bạc trên không gian mạng bị xử phạt thế nào?"
    }


def test_history_keeps_last_two_turns_and_truncates_answers():
    history = [
        SimpleNamespace(role="user", content="cũ"),
        SimpleNamespace(role="assistant", content="cũ"),
        SimpleNamespace(role="user", content="Câu 1"),
        SimpleNamespace(role="assistant", content="a" * 2000),
        SimpleNamespace(role="user", content="Câu 2"),
        SimpleNamespace(role="assistant", content=""),
    ]
    items = history_for_rewrite(history)
    assert [role for role, _ in items] == ["user", "assistant", "user"]
    assert len(items[1][1]) == 600


def test_dropped_legal_references_are_restored():
    restored = preserve_references(
        "Điều 36 Nghị định 123/2024/NĐ-CP quy định gì?",
        "Quy định về hóa đơn điện tử là gì?",
    )
    assert "Điều 36" in restored and "123/2024/NĐ-CP" in restored
    kept = "Điều 36 Nghị định 123/2024/NĐ-CP quy định những nội dung gì?"
    assert preserve_references("Điều 36 Nghị định 123/2024/NĐ-CP quy định gì?", kept) == kept


class _FakeProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    async def generate_async(self, request):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_facade_retries_once_then_returns_rewrite():
    provider = _FakeProvider([
        ProviderUnavailableError("down"),
        QueryRewrite("Đánh bạc trên không gian mạng từ 5 tỷ đồng bị phạt thế nào?", "", True),
    ])
    result = asyncio.run(
        QueryRewriter(provider).rewrite_async(QueryRewriteRequest("vậy nếu từ 5 tỷ"))
    )
    assert provider.calls == 2
    assert result.rewritten
    assert result.original_question == "vậy nếu từ 5 tỷ"
    assert result.standalone_question.startswith("Đánh bạc trên không gian mạng")


def test_facade_returns_none_when_provider_keeps_failing():
    provider = _FakeProvider([ProviderUnavailableError("down")] * 2)
    assert asyncio.run(
        QueryRewriter(provider).rewrite_async(QueryRewriteRequest("Câu hỏi"))
    ) is None
    assert asyncio.run(QueryRewriter(None).rewrite_async(QueryRewriteRequest("x"))) is None
