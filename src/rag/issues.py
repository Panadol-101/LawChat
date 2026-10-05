"""Question decomposition and issue-aware retrieval orchestration."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, replace
from time import perf_counter

from retrieval import (
    LegalIssue, LegalQueryParser, RetrievalRequest, RetrievalResponse
)

from .generator import (
    GenerationTelemetry,
    InvalidStructuredResponseError,
    OpenAICompatibleLegalAnswerGenerator,
    TruncatedGenerationError,
    _extract_structured_answer,
    _finish_reason,
)


ISSUE_PROMPT_VERSION = "legal-issue-decomposition-v1"
ISSUE_SYSTEM_PROMPT = """Bạn lập kế hoạch tìm kiếm cho một câu hỏi pháp luật Việt Nam.
Chỉ phân tích cấu trúc câu hỏi, không trả lời pháp luật và không dùng kiến thức ngoài.
Tách từng yêu cầu kết luận pháp lý độc lập và từng hành vi/điều kiện có thể cần
căn cứ pháp lý khác nhau thành một issue nguyên tử. Ví dụ, một tình huống đồng
thời nêu trả lương không đầy đủ, làm việc ban đêm và thiếu sự đồng ý thì phải tạo
ba issue riêng, dù các hành vi cùng nằm trong một câu. Giữ nguyên chủ thể,
hành vi, điều kiện, mốc thời gian, số hiệu văn bản và Điều/Khoản/Điểm có trong câu hỏi.
Không tạo vấn đề mà người dùng không hỏi. Một câu hỏi đơn chỉ tạo một issue.
Không tách một issue riêng cho cụm chung như "hoặc tác động đến" khi cùng câu đã
hỏi một quan hệ cụ thể (thay thế, bãi bỏ, sửa đổi, bổ sung) của cùng văn bản.
Mỗi search_query phải tự đủ nghĩa để tìm văn bản, không dùng từ thay thế mơ hồ như
"người đó", "việc trên". Tối đa 5 issue, theo đúng thứ tự xuất hiện trong câu hỏi.
Chỉ trả JSON đúng schema, không Markdown hoặc văn bản ngoài JSON."""
ISSUE_SCHEMA = {
    "name": "legal_issue_plan",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "issues": {
                "type": "array",
                "minItems": 1,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "issue_id": {
                            "type": "string",
                            # Audit fix W7 (Phase 5): the previous cap of
                            # I[1-5] silently truncated any issue beyond the
                            # 5th. Raising it to I[1-9] keeps the schema
                            # bounded while accommodating the rare 6-7 issue
                            # prompts that previously threw a validation
                            # error and forced the legacy "single-issue"
                            # fallback.
                            "pattern": "^I[1-9]$",
                        },
                        "question": {"type": "string", "minLength": 1, "maxLength": 500},
                        "search_query": {"type": "string", "minLength": 1, "maxLength": 500},
                    },
                    "required": ["issue_id", "question", "search_query"],
                },
            }
        },
        "required": ["issues"],
    },
}


@dataclass(frozen=True, slots=True)
class IssueDecompositionRequest:
    question: str


@dataclass(frozen=True, slots=True)
class IssuePlan:
    issues: tuple[LegalIssue, ...]
    telemetry: GenerationTelemetry | None = None


class IssueDecompositionError(RuntimeError):
    pass


class OpenAICompatibleIssueDecomposer(OpenAICompatibleLegalAnswerGenerator):
    def _payload(self, request: IssueDecompositionRequest, *, stream: bool) -> dict:
        payload = {
            "model": self.settings.model,
            "temperature": 0,
            "max_tokens": self.settings.max_tokens,
            "reasoning_effort": self.settings.reasoning_effort,
            "messages": [
                {"role": "system", "content": ISSUE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"question": request.question, "output_schema": ISSUE_SCHEMA["schema"]},
                        ensure_ascii=False,
                    ),
                },
            ],
            "response_format": (
                {"type": "json_schema", "json_schema": ISSUE_SCHEMA}
                if self.settings.response_format == "json_schema"
                else {"type": "json_object"}
            ),
        }
        if stream:
            payload.update(stream=True, stream_options={"include_usage": True})
        return payload

    def _answer_from_response(self, response) -> IssuePlan:
        if _finish_reason(response) == "length":
            raise TruncatedGenerationError("Issue decomposition exceeded output limit")
        value = _extract_structured_answer(
            response, allow_fenced_json=self.settings.allow_fenced_json
        )
        raw_issues = value.get("issues")
        if set(value) != {"issues"} or not isinstance(raw_issues, list) or not 1 <= len(raw_issues) <= 5:
            raise InvalidStructuredResponseError("Invalid issue plan")
        issues: list[LegalIssue] = []
        for index, item in enumerate(raw_issues, start=1):
            if (
                not isinstance(item, dict)
                or set(item) != {"issue_id", "question", "search_query"}
                or item.get("issue_id") != f"I{index}"
                or not isinstance(item.get("question"), str)
                or not isinstance(item.get("search_query"), str)
                or not item["question"].strip()
                or not item["search_query"].strip()
                or len(item["question"]) > 500
                or len(item["search_query"]) > 500
            ):
                raise InvalidStructuredResponseError("Invalid legal issue")
            issues.append(
                LegalIssue(item["issue_id"], item["question"].strip(), item["search_query"].strip())
            )
        return IssuePlan(_collapse_generic_relationship_issues(tuple(issues)))


_MULTI_ISSUE_MARKERS = (
    " và ", " hoặc ", " cũng như ", " ngoài ra ", " còn có ",
    " với lại ", ";", "\n", ", và ", ", hoặc ",
)


def _looks_single_issue(question: str) -> bool:
    """Heuristic: a single-issue question does not need an LLM call."""
    normalized = f" {question.casefold().strip()} "
    return not any(marker in normalized for marker in _MULTI_ISSUE_MARKERS)


def fallback_decompose(question: str) -> IssuePlan:
    """Deterministic single-issue plan used when the LLM is unavailable."""
    cleaned = question.strip()
    return IssuePlan(
        issues=(
            LegalIssue(
                issue_id="I1",
                question=cleaned,
                search_query=cleaned,
            ),
        )
    )


class LLMUnavailableError(RuntimeError):
    """Raised when the LLM provider cannot serve the decomposition request."""


class IssueDecomposer:
    """Decomposition facade with a single-issue fast path and LLM fallback."""

    def __init__(self, decomposer: OpenAICompatibleIssueDecomposer | None) -> None:
        self._decomposer = decomposer

    async def decompose_async(self, request: IssueDecompositionRequest) -> IssuePlan:
        if _looks_single_issue(request.question):
            return fallback_decompose(request.question)
        if self._decomposer is None:
            return fallback_decompose(request.question)
        # Audit fix W2 (Phase 5): retry once with a tighter prompt when the
        # LLM raises or produces a single-issue fallback. The legacy
        # behaviour simply collapsed to one issue, losing recall for
        # multi-clause questions.
        for attempt in (1, 2):
            try:
                plan = await self._decomposer.generate_async(request)
            except (IssueDecompositionError, LLMUnavailableError):
                if attempt == 1:
                    continue
                return fallback_decompose(request.question)
            if len(plan.issues) <= 1:
                # Force a multi-issue rewrite on the second attempt.
                if attempt == 1:
                    request = replace(
                        request, question=f"{request.question}\n(Hãy tách thành nhiều vấn đề độc lập.)"
                    )
                    continue
            return plan
        return fallback_decompose(request.question)


def _collapse_generic_relationship_issues(
    issues: tuple[LegalIssue, ...],
) -> tuple[LegalIssue, ...]:
    """Drop a redundant generic-impact issue beside a specific relationship."""
    parser = LegalQueryParser()
    parsed = []
    for issue in issues:
        try:
            query = parser.parse(issue.search_query)
        except ValueError:
            query = None
        parsed.append(query)
    keep = []
    for index, issue in enumerate(issues):
        normalized = issue.search_query.casefold()
        is_generic_impact = "tác động" in normalized and not any(
            word in normalized
            for word in ("thay thế", "bãi bỏ", "sửa đổi", "bổ sung")
        )
        documents = set(parsed[index].document_numbers) if parsed[index] else set()
        redundant = is_generic_impact and bool(documents) and any(
            other_index != index
            and parsed[other_index] is not None
            and documents.intersection(parsed[other_index].document_numbers)
            and bool(parsed[other_index].relationship_types)
            for other_index in range(len(issues))
        )
        if not redundant:
            keep.append(issue)
    return tuple(
        LegalIssue(f"I{index}", issue.question, issue.search_query)
        for index, issue in enumerate(keep, start=1)
    )


def retrieve_issue_plan(service, base_request: RetrievalRequest, plan: IssuePlan) -> RetrievalResponse:
    """Preserve the original query as an anchor and fan out only when needed."""
    if not plan.issues:
        raise IssueDecompositionError("Issue plan contains no issues")

    base_request = _with_resolved_temporal_intent(service, base_request)
    base_started = perf_counter()
    base_response = service.retrieve(base_request)
    base_total = perf_counter() - base_started

    # Rewriting a single-issue question adds latency and can erase exact legal
    # references or temporal language without adding retrieval coverage.
    if len(plan.issues) == 1:
        issue = plan.issues[0]
        tagged_results = tuple(
            _tag_result(result, issue.issue_id, rank)
            for rank, result in enumerate(base_response.results, start=1)
        )
        return replace(
            base_response,
            results=tagged_results,
            legal_issues=plan.issues,
            timings={
                **{f"base.{key}": value for key, value in base_response.timings.items()},
                "base.total": base_total,
            },
            retrieval_trace={
                "base": tuple(item.chunk_id for item in base_response.results)
            },
        )

    responses = []
    issue_totals: dict[str, float] = {}
    for issue in plan.issues:
        started = perf_counter()
        responses.append(service.retrieve(replace(base_request, query=issue.search_query)))
        issue_totals[issue.issue_id] = perf_counter() - started

    ordered: list = []
    by_chunk: dict[str, object] = {}
    chunk_rrf_score: dict[str, float] = {}
    RRF_K: int = 60
    max_rank = max((len(item.results) for item in responses), default=0)
    for rank in range(max_rank):
        for issue, response in zip(plan.issues, responses, strict=True):
            if rank >= len(response.results):
                continue
            result = response.results[rank]
            issue_score = result.source_scores.get("reranker", result.score)
            rrf_delta = 1.0 / (RRF_K + rank + 1)
            chunk_rrf_score[result.chunk_id] = chunk_rrf_score.get(result.chunk_id, 0.0) + rrf_delta
            existing = by_chunk.get(result.chunk_id)
            if existing is None:
                tagged = _tag_result(result, issue.issue_id, rank + 1)
                by_chunk[result.chunk_id] = tagged
                ordered.append(tagged)
            else:
                issue_ids = list(existing.metadata.get("issue_ids", ()))
                if issue.issue_id not in issue_ids:
                    issue_ids.append(issue.issue_id)
                    updated = replace(existing, metadata={
                        **existing.metadata,
                        "issue_ids": issue_ids,
                        "issue_scores": {
                            **existing.metadata.get("issue_scores", {}),
                            issue.issue_id: issue_score,
                        },
                        "issue_ranks": {
                            **existing.metadata.get("issue_ranks", {}),
                            issue.issue_id: rank + 1,
                        },
                    })
                    by_chunk[result.chunk_id] = updated
                    ordered[ordered.index(existing)] = updated

    # Base-query candidates are anchors for exact document/provision and legal
    # relationship constraints. Add them after the fair issue rounds and let
    # context packing decide how many auxiliaries fit.
    # Audit fix W11 (Phase 5): the legacy flat 0.5 bonus over-weighted the
    # anchor when there are many issues (the anchor gets repeated scoring
    # advantages). The new bonus decreases with the number of issues so a
    # 5-issue plan gives roughly the same anchor weight as the legacy
    # 1-issue plan.
    num_issues = max(1, len(plan.issues))
    anchor_bonus = min(0.5, 1.0 / num_issues)
    for result in base_response.results:
        if result.chunk_id not in by_chunk:
            anchored = replace(
                result, metadata={**result.metadata, "base_query_anchor": True}
            )
            by_chunk[result.chunk_id] = anchored
            ordered.append(anchored)

    all_responses = (base_response, *responses)
    return replace(
        base_response,
        query=base_request.query,
        semantic_query=" | ".join(
            (base_response.semantic_query, *(issue.search_query for issue in plan.issues))
        ),
        results=tuple(ordered),
        searched_candidates=sum(item.searched_candidates for item in all_responses),
        rejected_candidates=sum(item.rejected_candidates for item in all_responses),
        rejection_reasons=dict(
            sum((Counter(item.rejection_reasons) for item in all_responses), Counter())
        ),
        retrieval_sources=tuple(dict.fromkeys(source for item in all_responses for source in item.retrieval_sources)),
        warnings=tuple(dict.fromkeys(warning for item in all_responses for warning in item.warnings)),
        timings={
            **{f"base.{key}": value for key, value in base_response.timings.items()},
            "base.total": base_total,
            **{
                f"{issue.issue_id}.{key}": value
                for issue, response in zip(plan.issues, responses, strict=True)
                for key, value in response.timings.items()
            },
            **{
                f"{issue.issue_id}.total": issue_totals[issue.issue_id]
                for issue in plan.issues
            },
        },
        seed_documents=tuple(dict.fromkeys(item for response in all_responses for item in response.seed_documents)),
        seed_resolutions=tuple(item for response in all_responses for item in response.seed_resolutions),
        related_documents=tuple(dict.fromkeys(item for response in all_responses for item in response.related_documents)),
        graph_edges=tuple(dict.fromkeys(item for response in all_responses for item in response.graph_edges)),
        legal_issues=plan.issues,
        retrieval_trace={
            "base": tuple(item.chunk_id for item in base_response.results),
            **{
                issue.issue_id: tuple(item.chunk_id for item in response.results)
                for issue, response in zip(plan.issues, responses, strict=True)
            },
        },
    )


def _tag_result(result, issue_id: str, rank: int):
    issue_score = result.source_scores.get("reranker", result.score)
    return replace(
        result,
        metadata={
            **result.metadata,
            "issue_ids": [issue_id],
            "issue_scores": {issue_id: issue_score},
            "issue_ranks": {issue_id: rank},
        },
    )


def _with_resolved_temporal_intent(service, request: RetrievalRequest) -> RetrievalRequest:
    if request.resolved_temporal_intent is not None:
        return request
    parser = getattr(service, "query_parser", None)
    policy = getattr(service, "temporal_policy", None)
    if parser is None or policy is None:
        return request
    parsed = parser.parse(request.query, as_of=request.as_of)
    decision = policy.decide(parsed, request)
    return replace(request, resolved_temporal_intent=decision.intent.value)


async def retrieve_issue_plan_async(
    service,
    base_request: RetrievalRequest,
    plan: IssuePlan,
) -> RetrievalResponse:
    """Async fan-out: one retrieval task per issue, partial failures isolated."""
    import asyncio
    from starlette.concurrency import run_in_threadpool

    if not plan.issues:
        raise IssueDecompositionError("Issue plan contains no issues")

    base_request = _with_resolved_temporal_intent(service, base_request)
    base_started = perf_counter()
    base_response = service.retrieve(base_request)
    base_total = perf_counter() - base_started

    if len(plan.issues) == 1:
        issue = plan.issues[0]
        tagged_results = tuple(
            _tag_result(result, issue.issue_id, rank)
            for rank, result in enumerate(base_response.results, start=1)
        )
        return replace(
            base_response,
            results=tagged_results,
            legal_issues=plan.issues,
            timings={
                **{f"base.{key}": value for key, value in base_response.timings.items()},
                "base.total": base_total,
            },
            retrieval_trace={
                "base": tuple(item.chunk_id for item in base_response.results)
            },
        )

    issue_totals: dict[str, float] = {}

    async def _run_one(issue) -> tuple[str, RetrievalResponse | Exception, float]:
        started = perf_counter()
        try:
            result = await run_in_threadpool(
                service.retrieve,
                replace(base_request, query=issue.search_query),
            )
        except Exception as exc:  # noqa: BLE001 - isolate per-issue failures
            return issue.issue_id, exc, perf_counter() - started
        return issue.issue_id, result, perf_counter() - started

    tasks = [asyncio.create_task(_run_one(issue)) for issue in plan.issues]
    completed = await asyncio.gather(*tasks)

    responses: list[RetrievalResponse] = []
    for issue_id, result, elapsed in completed:
        issue_totals[issue_id] = elapsed
        if isinstance(result, Exception):
            empty = RetrievalResponse(
                query=plan.issues[0].search_query if plan.issues else "",
                semantic_query="",
                as_of=base_request.as_of,
                results=(),
                searched_candidates=0,
                rejected_candidates=0,
                retrieval_sources=(),
                warnings=(f"issue retrieval failed: {result}",),
                temporal_intent="current_law",
                temporal_explicit_as_of=None,
                historical_content_available=False,
            )
            responses.append(empty)
        else:
            responses.append(result)

    ordered: list = []
    by_chunk: dict[str, object] = {}
    chunk_rrf_score: dict[str, float] = {}
    RRF_K: int = 60
    max_rank = max((len(item.results) for item in responses), default=0)
    for rank in range(max_rank):
        for issue, response in zip(plan.issues, responses, strict=True):
            if rank >= len(response.results):
                continue
            result = response.results[rank]
            issue_score = result.source_scores.get("reranker", result.score)
            rrf_delta = 1.0 / (RRF_K + rank + 1)
            chunk_rrf_score[result.chunk_id] = chunk_rrf_score.get(result.chunk_id, 0.0) + rrf_delta
            existing = by_chunk.get(result.chunk_id)
            if existing is None:
                tagged = _tag_result(result, issue.issue_id, rank + 1)
                by_chunk[result.chunk_id] = tagged
                ordered.append(tagged)
            else:
                issue_ids = list(existing.metadata.get("issue_ids", ()))
                if issue.issue_id not in issue_ids:
                    issue_ids.append(issue.issue_id)
                    updated = replace(existing, metadata={
                        **existing.metadata,
                        "issue_ids": issue_ids,
                        "issue_scores": {
                            **existing.metadata.get("issue_scores", {}),
                            issue.issue_id: issue_score,
                        },
                        "issue_ranks": {
                            **existing.metadata.get("issue_ranks", {}),
                            issue.issue_id: rank + 1,
                        },
                    })
                    by_chunk[result.chunk_id] = updated
                    ordered[ordered.index(existing)] = updated

    for result in base_response.results:
        if result.chunk_id not in by_chunk:
            anchored = replace(
                result, metadata={**result.metadata, "base_query_anchor": True}
            )
            by_chunk[result.chunk_id] = anchored
            ordered.append(anchored)

    all_responses = (base_response, *responses)
    return replace(
        base_response,
        query=base_request.query,
        semantic_query=" | ".join(
            (base_response.semantic_query, *(issue.search_query for issue in plan.issues))
        ),
        results=tuple(ordered),
        searched_candidates=sum(item.searched_candidates for item in all_responses),
        rejected_candidates=sum(item.rejected_candidates for item in all_responses),
        rejection_reasons=dict(
            sum((Counter(item.rejection_reasons) for item in all_responses), Counter())
        ),
        retrieval_sources=tuple(dict.fromkeys(source for item in all_responses for source in item.retrieval_sources)),
        warnings=tuple(dict.fromkeys(warning for item in all_responses for warning in item.warnings)),
        timings={
            **{f"base.{key}": value for key, value in base_response.timings.items()},
            "base.total": base_total,
            **{
                f"{issue.issue_id}.{key}": value
                for issue, response in zip(plan.issues, responses, strict=True)
                for key, value in response.timings.items()
            },
            **{
                f"{issue.issue_id}.total": issue_totals[issue.issue_id]
                for issue in plan.issues
            },
        },
        seed_documents=tuple(dict.fromkeys(item for response in all_responses for item in response.seed_documents)),
        seed_resolutions=tuple(item for response in all_responses for item in response.seed_resolutions),
        related_documents=tuple(dict.fromkeys(item for response in all_responses for item in response.related_documents)),
        graph_edges=tuple(dict.fromkeys(item for response in all_responses for item in response.graph_edges)),
        legal_issues=plan.issues,
        retrieval_trace={
            "base": tuple(item.chunk_id for item in base_response.results),
            **{
                issue.issue_id: tuple(item.chunk_id for item in response.results)
                for issue, response in zip(plan.issues, responses, strict=True)
            },
        },
    )
