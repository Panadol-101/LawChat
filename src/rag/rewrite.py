"""LLM query rewriting: turn each question into a standalone retrieval query."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

from retrieval import LegalQueryParser

from .generator import (
    GenerationTelemetry,
    InvalidStructuredResponseError,
    OpenAICompatibleLegalAnswerGenerator,
    TruncatedGenerationError,
    _extract_structured_answer,
    _finish_reason,
)


REWRITE_PROMPT_VERSION = "legal-query-rewrite-v1"
REWRITE_SYSTEM_PROMPT = """Bạn viết lại câu hỏi pháp luật Việt Nam để tìm kiếm văn bản pháp luật.
Không trả lời câu hỏi và không dùng kiến thức pháp luật ngoài câu hỏi và lịch sử hội thoại.
Viết lại câu hỏi hiện tại thành MỘT câu hỏi độc lập, tự đủ nghĩa khi đứng một mình:
- Nếu câu hỏi phụ thuộc lượt trước (ví dụ bắt đầu bằng "vậy", "thế còn", "còn",
  hoặc dùng "điều đó", "trường hợp này", hoặc chỉ nêu thêm một tình tiết như số
  tiền, độ tuổi), hãy bổ sung chủ thể, hành vi, quan hệ pháp luật và văn bản đang
  được bàn từ lịch sử hội thoại. Ví dụ: lượt trước hỏi "Đánh cờ bạc trên không gian
  mạng bị xử phạt thế nào?", câu hiện tại "vậy nếu từ 5 tỷ đồng trở lên thì phạt như
  thế nào" → "Đánh bạc trên không gian mạng với số tiền từ 5 tỷ đồng trở lên bị xử
  phạt như thế nào?".
- Nếu câu hỏi đã độc lập hoặc chuyển sang chủ đề mới, không lấy tình tiết từ lịch sử.
- Dùng thuật ngữ pháp lý chuẩn thay cho cách nói thông thường khi chắc chắn cùng
  nghĩa (ví dụ "đánh bài ăn tiền" → "đánh bạc"), nhưng không đổi ý câu hỏi.
- Giữ nguyên số hiệu văn bản, tên văn bản, Điều/Khoản/Điểm, số tiền, độ tuổi,
  mốc thời gian và mọi tình tiết có trong câu hỏi hiện tại.
- Không thêm tình tiết, giả định hay văn bản mà người dùng và lịch sử không nêu.
- Câu hỏi có nhiều ý thì giữ đủ các ý trong một câu hỏi, không bỏ ý nào.
Chỉ trả JSON đúng schema, không Markdown hoặc văn bản ngoài JSON."""
REWRITE_SCHEMA = {
    "name": "legal_query_rewrite",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "standalone_question": {"type": "string", "minLength": 1, "maxLength": 500},
        },
        "required": ["standalone_question"],
    },
}

_MAX_HISTORY_MESSAGES = 4
_MAX_ASSISTANT_CHARS = 600
_MAX_USER_CHARS = 1000

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class QueryRewriteRequest:
    question: str
    # (role, content) pairs of earlier turns, oldest first.
    history: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class QueryRewrite:
    standalone_question: str
    original_question: str
    rewritten: bool
    telemetry: GenerationTelemetry | None = None


def history_for_rewrite(prior_turns) -> tuple[tuple[str, str], ...]:
    """Latest two turns as (role, content), assistant replies truncated."""
    items = []
    for message in list(prior_turns)[-_MAX_HISTORY_MESSAGES:]:
        role = (getattr(message, "role", "") or "").lower()
        content = (getattr(message, "content", "") or "").strip()
        if not content:
            continue
        if role in {"user", "human"}:
            items.append(("user", content[:_MAX_USER_CHARS]))
        elif role == "assistant":
            items.append(("assistant", content[:_MAX_ASSISTANT_CHARS]))
    return tuple(items)


class OpenAICompatibleQueryRewriter(OpenAICompatibleLegalAnswerGenerator):
    def _payload(self, request: QueryRewriteRequest, *, stream: bool) -> dict:
        payload = {
            "model": self.settings.model,
            "temperature": 0,
            "max_tokens": self.settings.max_tokens,
            "reasoning_effort": self.settings.reasoning_effort,
            "messages": [
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "history": [
                                {"role": role, "content": content}
                                for role, content in request.history
                            ],
                            "question": request.question,
                            "output_schema": REWRITE_SCHEMA["schema"],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "response_format": (
                {"type": "json_schema", "json_schema": REWRITE_SCHEMA}
                if self.settings.response_format == "json_schema"
                else {"type": "json_object"}
            ),
        }
        if stream:
            payload.update(stream=True, stream_options={"include_usage": True})
        return payload

    def _answer_from_response(self, response) -> QueryRewrite:
        if _finish_reason(response) == "length":
            raise TruncatedGenerationError("Query rewrite exceeded output limit")
        value = _extract_structured_answer(
            response, allow_fenced_json=self.settings.allow_fenced_json
        )
        question = value.get("standalone_question")
        if (
            set(value) != {"standalone_question"}
            or not isinstance(question, str)
            or not question.strip()
            or len(question) > 500
        ):
            raise InvalidStructuredResponseError("Invalid query rewrite")
        # original_question is filled in by QueryRewriter.
        return QueryRewrite(question.strip(), "", rewritten=True)


class QueryRewriter:
    """Rewrite facade: retry once, then report failure so callers fall back."""

    def __init__(self, rewriter: OpenAICompatibleQueryRewriter | None) -> None:
        self._rewriter = rewriter

    async def rewrite_async(self, request: QueryRewriteRequest) -> QueryRewrite | None:
        """Return the standalone question, or None when the LLM is unusable."""
        if self._rewriter is None:
            return None
        for attempt in (1, 2):
            try:
                result = await self._rewriter.generate_async(request)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - degrade, never fail closed here
                logger.warning(
                    "query rewrite attempt %d failed: %s", attempt, type(exc).__name__
                )
                continue
            return QueryRewrite(
                standalone_question=preserve_references(
                    request.question, result.standalone_question
                ),
                original_question=request.question,
                rewritten=True,
                telemetry=result.telemetry,
            )
        return None


def preserve_references(original: str, rewritten: str) -> str:
    """Re-append document numbers and articles the rewrite dropped.

    Exact legal references drive filtered retrieval, so a paraphrase must
    never lose one the user typed.
    """
    parser = LegalQueryParser()
    try:
        own = parser.parse(original)
    except ValueError:
        return rewritten
    try:
        new = parser.parse(rewritten)
        kept_documents, kept_articles = set(new.document_numbers), set(new.referenced_articles)
    except ValueError:
        kept_documents, kept_articles = set(), set()
    missing = [
        *(f"Điều {item}" for item in own.referenced_articles if item not in kept_articles),
        *(item for item in own.document_numbers if item not in kept_documents),
    ]
    if not missing:
        return rewritten
    return f"{rewritten} (tham chiếu: {', '.join(missing)})"
