from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from time import perf_counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import date
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .context_models import PackedContext
from retrieval import LegalIssue
from .prompts import GENERATED_ANSWER_JSON_SCHEMA, SYSTEM_PROMPT, build_user_prompt

if TYPE_CHECKING:
    from retrieval import RetrievalResponse


class GenerationError(RuntimeError):
    """A provider failed or returned an invalid structured response."""

    telemetry = None


class TruncatedGenerationError(GenerationError):
    """The provider stopped at its output-token limit before completing JSON."""


class GenerationTimeoutError(GenerationError):
    """The provider did not finish within the configured deadline."""


class ProviderUnavailableError(GenerationError):
    """The configured LLM endpoint is unavailable or rejected the request."""


class InvalidStructuredResponseError(GenerationError):
    """The provider response cannot be parsed as the required answer schema."""

    def __init__(
        self,
        message: str,
        *,
        contract_code: str = "INVALID_STRUCTURED_RESPONSE",
        detail: str = "",
    ) -> None:
        super().__init__(message)
        self.contract_code = contract_code
        # What exactly broke the contract, phrased for the repair prompt.
        self.detail = detail


@dataclass(frozen=True, slots=True)
class GenerationTelemetry:
    total_seconds: float
    ttft_seconds: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class SupportingQuote:
    evidence_id: str
    quote: str

    def to_dict(self) -> dict[str, str]:
        return {"evidence_id": self.evidence_id, "quote": self.quote}


@dataclass(frozen=True, slots=True)
class GeneratedClaim:
    text: str
    evidence_ids: tuple[str, ...]
    supporting_quotes: tuple[SupportingQuote, ...] = ()
    issue_ids: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], index: int = 0) -> GeneratedClaim:
        text = value.get("text")
        evidence_ids = value.get("evidence_ids")
        if not isinstance(text, str) or not isinstance(evidence_ids, list):
            raise InvalidStructuredResponseError(
                "Invalid claim in structured LLM response",
                contract_code="CLAIM_SHAPE_INVALID",
            )
        if not all(isinstance(item, str) for item in evidence_ids):
            raise InvalidStructuredResponseError(
                "Invalid evidence IDs in structured LLM response",
                contract_code="CLAIM_EVIDENCE_IDS_INVALID",
            )
        raw_quotes = value.get("supporting_quotes", [])
        issue_ids = value.get("issue_ids", [])
        if not isinstance(issue_ids, list) or not all(isinstance(item, str) for item in issue_ids):
            raise InvalidStructuredResponseError(
                "Invalid claim issue IDs", contract_code="CLAIM_ISSUE_IDS_INVALID"
            )
        if not isinstance(raw_quotes, list) or len(raw_quotes) > 8:
            raise InvalidStructuredResponseError(
                "Invalid supporting quotes", contract_code="SUPPORTING_QUOTES_INVALID"
            )
        quotes = []
        for item in raw_quotes:
            if (not isinstance(item, Mapping) or set(item) != {"evidence_id", "quote"}
                or not isinstance(item["evidence_id"], str)
                or not isinstance(item["quote"], str)
                or not item["quote"].strip()):
                raise InvalidStructuredResponseError(
                    "Invalid supporting quote", contract_code="SUPPORTING_QUOTE_INVALID"
                )
            quote = item["quote"].strip()
            if len(quote) > 1200:
                quote = quote[:1200].rsplit(" ", 1)[0].rstrip()
            quotes.append(SupportingQuote(item["evidence_id"], quote))
        violations = []
        extra_keys = set(value) - {"text", "evidence_ids", "supporting_quotes", "issue_ids"}
        if extra_keys:
            violations.append(f"có khóa thừa {sorted(extra_keys)}")
        if not text.strip():
            violations.append("text rỗng")
        if len(text) > 500:
            violations.append(f"text dài {len(text)} ký tự, tối đa 500")
        if len(evidence_ids) > 8:
            violations.append(f"{len(evidence_ids)} evidence_ids, tối đa 8")
        if len(issue_ids) > 5:
            violations.append(f"{len(issue_ids)} issue_ids, tối đa 5")
        if violations:
            raise InvalidStructuredResponseError(
                "Claim violates output contract",
                contract_code="CLAIM_CONTRACT_INVALID",
                detail=f"claim {index}: " + "; ".join(violations),
            )
        return cls(
            text=text.strip(),
            evidence_ids=tuple(evidence_ids),
            supporting_quotes=tuple(quotes),
            issue_ids=tuple(issue_ids),
        )

    def to_dict(self) -> dict[str, Any]:
        result = {"text": self.text, "evidence_ids": list(self.evidence_ids)}
        if self.supporting_quotes:
            result["supporting_quotes"] = [item.to_dict() for item in self.supporting_quotes]
        if self.issue_ids:
            result["issue_ids"] = list(self.issue_ids)
        return result


@dataclass(frozen=True, slots=True)
class IssueResolution:
    issue_id: str
    status: str
    claim_indexes: tuple[int, ...]
    explanation: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> IssueResolution:
        if not isinstance(value, Mapping):
            raise InvalidStructuredResponseError(
                "Invalid issue resolution", contract_code="ISSUE_RESOLUTION_INVALID"
            )
        issue_id = value.get("issue_id")
        if not isinstance(issue_id, str) or not issue_id.strip():
            raise InvalidStructuredResponseError(
                "Invalid issue resolution", contract_code="ISSUE_RESOLUTION_INVALID"
            )
        status = value.get("status")
        if isinstance(status, str):
            status = status.upper().strip()
        if status not in {"ANSWERED", "INSUFFICIENT_EVIDENCE", "NEEDS_CLARIFICATION"}:
            raise InvalidStructuredResponseError(
                "Invalid issue resolution", contract_code="ISSUE_RESOLUTION_INVALID"
            )
        raw_indexes = value.get("claim_indexes", [])
        if not isinstance(raw_indexes, list):
            raise InvalidStructuredResponseError(
                "Invalid issue resolution", contract_code="ISSUE_RESOLUTION_INVALID"
            )
        claim_indexes: list[int] = []
        for item in raw_indexes:
            if isinstance(item, int) and item >= 0:
                claim_indexes.append(item)
            elif isinstance(item, str) and item.strip().isdigit() and int(item.strip()) >= 0:
                claim_indexes.append(int(item.strip()))
            else:
                raise InvalidStructuredResponseError(
                    "Invalid issue resolution", contract_code="ISSUE_RESOLUTION_INVALID"
                )
        explanation = value.get("explanation", "")
        if not isinstance(explanation, str) or not explanation.strip():
            explanation = "Đã giải quyết theo nội dung trích dẫn."
        explanation = explanation.strip()[:500]
        return cls(
            issue_id.strip(), status, tuple(claim_indexes), explanation
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "status": self.status,
            "claim_indexes": list(self.claim_indexes),
            "explanation": self.explanation,
        }


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    answer: str
    claims: tuple[GeneratedClaim, ...]
    limitations: tuple[str, ...]
    confidence: str
    telemetry: GenerationTelemetry | None = None
    issue_resolutions: tuple[IssueResolution, ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GeneratedAnswer:
        answer = value.get("answer")
        claims = value.get("claims")
        limitations = value.get("limitations")
        confidence = value.get("confidence")
        raw_resolutions = value.get("issue_resolutions", [])
        if not isinstance(answer, str) or not isinstance(claims, list):
            raise InvalidStructuredResponseError(
                "Invalid structured LLM response", contract_code="ROOT_SHAPE_INVALID"
            )
        if not isinstance(limitations, list) or not all(
            isinstance(item, str) for item in limitations
        ):
            raise InvalidStructuredResponseError(
                "Invalid limitations in structured LLM response",
                contract_code="LIMITATIONS_NOT_STRING_ARRAY",
            )
        if confidence not in {"high", "medium", "low"}:
            raise InvalidStructuredResponseError(
                "Invalid confidence in structured LLM response",
                contract_code="CONFIDENCE_INVALID",
            )
        if not all(isinstance(item, Mapping) for item in claims):
            raise InvalidStructuredResponseError(
                "Invalid claims in structured LLM response", contract_code="CLAIMS_INVALID"
            )
        if not isinstance(raw_resolutions, list) or len(raw_resolutions) > 5:
            raise InvalidStructuredResponseError(
                "Invalid issue resolutions", contract_code="ISSUE_RESOLUTIONS_INVALID"
            )
        required_keys = {"answer", "claims", "limitations", "confidence"}
        if not required_keys.issubset(set(value)) or len(claims) > 10:
            raise InvalidStructuredResponseError(
                "Answer violates output contract", contract_code="ROOT_CONTRACT_INVALID"
            )
        trimmed_limitations = tuple(item.strip()[:300] for item in limitations[:5])
        return cls(
            answer=answer.strip(),
            claims=tuple(
                GeneratedClaim.from_dict(item, index) for index, item in enumerate(claims[:5])
            ),
            limitations=trimmed_limitations,
            confidence=confidence,
            issue_resolutions=tuple(IssueResolution.from_dict(item) for item in raw_resolutions[:5]),
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "answer": self.answer,
            "claims": [claim.to_dict() for claim in self.claims],
            "limitations": list(self.limitations),
            "confidence": self.confidence,
        }
        if self.issue_resolutions:
            result["issue_resolutions"] = [item.to_dict() for item in self.issue_resolutions]
        return result


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    question: str
    context: PackedContext
    as_of: date
    temporal_intent: str = "current_law"
    warnings: tuple[str, ...] = ()
    previous_answer: GeneratedAnswer | None = None
    repair_instructions: tuple[str, ...] = ()
    issues: tuple[LegalIssue, ...] = ()
    # The user's verbatim wording when ``question`` is an LLM rewrite of it.
    original_question: str | None = None

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("question must not be empty")
        if self.previous_answer is None and self.repair_instructions:
            raise ValueError("repair_instructions require previous_answer")

    @classmethod
    def from_retrieval(
        cls,
        response: RetrievalResponse,
        context: PackedContext,
    ) -> GenerationRequest:
        return cls(
            question=response.query,
            context=context,
            as_of=response.as_of,
            temporal_intent=response.temporal_intent,
            warnings=response.warnings,
            issues=response.legal_issues,
        )


@runtime_checkable
class LegalAnswerGenerator(Protocol):
    def generate(self, request: GenerationRequest) -> GeneratedAnswer: ...


class NoOpLegalAnswerGenerator:
    """Safe provider for deployments where generation is disabled."""

    def generate(self, request: GenerationRequest) -> GeneratedAnswer:
        return GeneratedAnswer(
            answer="Không đủ căn cứ để tạo câu trả lời pháp lý.",
            claims=(),
            limitations=("Bộ sinh câu trả lời chưa được cấu hình.",),
            confidence="low",
        )


class FakeLegalAnswerGenerator:
    """Deterministic queued provider intended for unit and integration tests."""

    def __init__(self, responses: Iterable[GeneratedAnswer | Exception]) -> None:
        self._responses = list(responses)
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GeneratedAnswer:
        self.requests.append(request)
        if not self._responses:
            raise GenerationError("Fake generator has no response configured")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@dataclass(frozen=True, slots=True)
class OpenAICompatibleSettings:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float = 30.0
    max_tokens: int = 1024
    reasoning_effort: str = "none"
    response_format: str = "json_schema"
    allow_fenced_json: bool = False

    @classmethod
    def from_env(cls, prefix: str = "LAWCHAT_LLM_") -> OpenAICompatibleSettings:
        base_url = os.getenv(f"{prefix}BASE_URL", "https://api.openai.com/v1")
        api_key = os.getenv(f"{prefix}API_KEY", "")
        model = os.getenv(f"{prefix}MODEL", "")
        if not model:
            raise ValueError(f"{prefix}MODEL must be configured")
        timeout = float(os.getenv(f"{prefix}TIMEOUT_SECONDS", "30"))
        if timeout <= 0:
            raise ValueError(f"{prefix}TIMEOUT_SECONDS must be > 0")
        max_tokens = int(os.getenv(f"{prefix}MAX_TOKENS", "1024"))
        if max_tokens <= 0:
            raise ValueError(f"{prefix}MAX_TOKENS must be > 0")
        reasoning_effort = os.getenv(f"{prefix}REASONING_EFFORT", "none")
        if reasoning_effort not in {"none", "low", "medium", "high"}:
            raise ValueError(
                f"{prefix}REASONING_EFFORT must be none, low, medium or high"
            )
        response_format = os.getenv(f"{prefix}RESPONSE_FORMAT", "json_schema")
        if response_format not in {"json_schema", "json_object"}:
            raise ValueError(f"{prefix}RESPONSE_FORMAT must be json_schema or json_object")
        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            response_format=response_format,
            allow_fenced_json=os.getenv(f"{prefix}ALLOW_FENCED_JSON", "false").lower() in {"true", "1", "yes"},
        )


JsonTransport = Callable[[str, Mapping[str, Any], Mapping[str, str], float], Mapping[str, Any]]


class OpenAICompatibleLegalAnswerGenerator:
    """Synchronous chat-completions provider with strict JSON-schema output."""

    def __init__(
        self,
        settings: OpenAICompatibleSettings,
        *,
        transport: JsonTransport | None = None,
        async_transport: Any | None = None,
    ) -> None:
        self.settings = settings
        self._transport = transport or _post_json
        self._async_transport = async_transport

    def generate(self, request: GenerationRequest) -> GeneratedAnswer:
        started = perf_counter()
        payload = self._payload(request, stream=False)
        endpoint = self.settings.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        response = {}
        try:
            response = self._transport(
                endpoint,
                payload,
                headers,
                self.settings.timeout_seconds,
            )
            answer = self._answer_from_response(response)
            usage = response.get("usage", {})
            return replace(
                answer,
                telemetry=GenerationTelemetry(
                    total_seconds=perf_counter() - started,
                    prompt_tokens=_optional_int(usage.get("prompt_tokens")),
                    completion_tokens=_optional_int(usage.get("completion_tokens")),
                ),
            )
        except GenerationError as exc:
            usage = response.get("usage", {})
            exc.telemetry = GenerationTelemetry(perf_counter() - started,
                prompt_tokens=_optional_int(usage.get("prompt_tokens")),
                completion_tokens=_optional_int(usage.get("completion_tokens")),
                error_code=getattr(exc, "contract_code", None))
            raise
        except Exception as exc:
            raise GenerationError("LLM provider request failed") from exc

    async def generate_async(self, request: GenerationRequest) -> GeneratedAnswer:
        import httpx

        started = perf_counter()
        first_token_at: float | None = None
        content_parts: list[str] = []
        finish_reason: str | None = None
        usage: dict[str, Any] = {}
        endpoint = self.settings.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        try:
            timeout = httpx.Timeout(self.settings.timeout_seconds)
            async with httpx.AsyncClient(
                timeout=timeout,
                transport=self._async_transport,
            ) as client:
                async with client.stream(
                    "POST",
                    endpoint,
                    json=self._payload(request, stream=True),
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        data = line[5:].strip() if line.startswith("data:") else line
                        if data == "[DONE]":
                            break
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError as exc:
                            raise InvalidStructuredResponseError(
                                "LLM stream contained invalid JSON"
                            ) from exc
                        usage.update(event.get("usage") or {})
                        choices = event.get("choices") or ()
                        if not choices:
                            continue
                        choice = choices[0]
                        finish_reason = choice.get("finish_reason") or finish_reason
                        delta = choice.get("delta") or choice.get("message") or {}
                        content = delta.get("content")
                        if isinstance(content, str) and content:
                            if first_token_at is None:
                                first_token_at = perf_counter()
                            content_parts.append(content)
        except httpx.TimeoutException as exc:
            raise GenerationTimeoutError("LLM provider request timed out") from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError("LLM provider is unavailable") from exc

        response_payload = {
            "choices": [
                {
                    "message": {"content": "".join(content_parts)},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
        }
        telemetry = GenerationTelemetry(
            total_seconds=perf_counter() - started,
            ttft_seconds=first_token_at - started if first_token_at is not None else None,
            prompt_tokens=_optional_int(usage.get("prompt_tokens")),
            completion_tokens=_optional_int(usage.get("completion_tokens")),
            error_code=None)
        try:
            answer = self._answer_from_response(response_payload)
        except GenerationError as exc:
            exc.telemetry = replace(
                telemetry, error_code=getattr(exc, "contract_code", None)
            )
            raise
        return replace(answer, telemetry=telemetry)

    def _payload(self, request: GenerationRequest, *, stream: bool) -> dict[str, Any]:
        payload = {
            "model": self.settings.model,
            "temperature": 0,
            "reasoning_effort": self.settings.reasoning_effort,
            "max_tokens": self.settings.max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(request)},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": GENERATED_ANSWER_JSON_SCHEMA,
            },
        }
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        if self.settings.response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _answer_from_response(self, response: Mapping[str, Any]) -> GeneratedAnswer:
        if _finish_reason(response) == "length":
            raise TruncatedGenerationError(
                "LLM provider truncated the structured response"
            )
        structured = _extract_structured_answer(response, allow_fenced_json=self.settings.allow_fenced_json)
        return GeneratedAnswer.from_dict(structured)


def _post_json(
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout: float,
) -> Mapping[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (TimeoutError, urllib.error.URLError) as exc:
        if isinstance(exc, TimeoutError):
            raise GenerationTimeoutError("LLM provider request timed out") from exc
        raise ProviderUnavailableError("LLM provider is unavailable") from exc
    except json.JSONDecodeError as exc:
        raise InvalidStructuredResponseError(
            "LLM provider returned invalid JSON", contract_code="JSON_PARSE_ERROR"
        ) from exc
    if not isinstance(value, Mapping):
        raise InvalidStructuredResponseError(
            "LLM provider returned a non-object response",
            contract_code="PROVIDER_RESPONSE_NOT_OBJECT",
        )
    return value


def _extract_structured_answer(response: Mapping[str, Any], *, allow_fenced_json: bool = False) -> Mapping[str, Any]:
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise InvalidStructuredResponseError(
            "LLM provider response has no assistant message",
            contract_code="ASSISTANT_MESSAGE_MISSING",
        ) from exc
    parsed = message.get("parsed")
    if isinstance(parsed, Mapping):
        return parsed
    content = message.get("content")
    if not isinstance(content, str):
        raise InvalidStructuredResponseError(
            "LLM provider response has no structured content",
            contract_code="STRUCTURED_CONTENT_MISSING",
        )
    if allow_fenced_json:
        fenced = re.fullmatch(r"\s*```(?:json)?\s*\n(.*?)\n```\s*", content, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            content = fenced.group(1)
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise InvalidStructuredResponseError(
            "LLM provider returned invalid JSON", contract_code="JSON_PARSE_ERROR"
        ) from exc
    if not isinstance(value, Mapping):
        raise InvalidStructuredResponseError(
            "LLM provider returned a non-object answer",
            contract_code="ANSWER_NOT_OBJECT",
        )
    return value


def _finish_reason(response: Mapping[str, Any]) -> str | None:
    try:
        value = response["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError, AttributeError):
        return None
    return value if isinstance(value, str) else None


def _optional_int(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None
