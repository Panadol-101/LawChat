from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from time import perf_counter

from .generator import (
    GeneratedAnswer,
    GenerationTelemetry,
    GenerationTimeoutError,
    GenerationRequest,
    InvalidStructuredResponseError,
    LegalAnswerGenerator,
    ProviderUnavailableError,
    TruncatedGenerationError,
)
from .semantic import (
    SemanticVerifier, SemanticObservation, SemanticReview,
    repair_supporting_quotes,
)
from .verifier import (
    GroundingVerifier,
    VerificationCode,
    VerificationIssue,
    VerificationResult,
    VerificationStatus,
)


@dataclass(frozen=True, slots=True)
class GenerationStageTiming:
    stage: str
    attempt: int
    seconds: float


@dataclass(frozen=True, slots=True)
class GenerationResult:
    answer: GeneratedAnswer
    verification: VerificationResult
    attempts: int
    telemetry: tuple[GenerationTelemetry, ...] = ()

    semantic_observations: tuple[SemanticObservation, ...] = ()
    semantic_mode: str = "off"
    stage_timings: tuple[GenerationStageTiming, ...] = ()

    @property
    def semantic_status(self) -> str:
        if self.semantic_mode == "off":
            return "DISABLED"
        if not self.semantic_observations:
            return "NOT_CHECKED"
        latest = self.semantic_observations[-1].review
        if latest.error:
            return "ERROR"
        return "PASSED" if latest.passed else "FLAGGED"

    @property
    def coverage_status(self) -> str:
        if not self.semantic_observations:
            return "NOT_CHECKED"
        latest = self.semantic_observations[-1].review
        if latest.error:
            return "ERROR"
        if not latest.coverage_required:
            return "NOT_CHECKED"
        if not latest.plan_complete or any(
            item.verdict == "MISSING_OR_PARTIAL" for item in latest.coverage_checks
        ):
            return "INCOMPLETE"
        return "COMPLETE"

    @property
    def status(self) -> VerificationStatus:
        return self.verification.status


class GroundedRAGService:
    """Generate, verify, optionally repair once, then fail closed."""

    def __init__(
        self,
        generator: LegalAnswerGenerator,
        *,
        verifier: GroundingVerifier | None = None,
        semantic_verifier: SemanticVerifier | None = None,
        shadow_semantic_verifier: SemanticVerifier | None = None,
        semantic_mode: str = "off",
    ) -> None:
        self.generator = generator
        self.verifier = verifier or GroundingVerifier()
        if semantic_mode not in {"off", "shadow", "enforce"}:
            raise ValueError("semantic_mode must be off, shadow or enforce")
        if semantic_mode != "off" and semantic_verifier is None:
            raise ValueError("semantic_verifier required when semantic review is enabled")
        self.semantic_verifier = semantic_verifier
        self.shadow_semantic_verifier = shadow_semantic_verifier
        self.semantic_mode = semantic_mode

    def answer(self, request: GenerationRequest) -> GenerationResult:
        observations: list[SemanticObservation] = []
        stages: list[GenerationStageTiming] = []
        result = self._answer(request, observations, stages)
        return replace(result, semantic_observations=tuple(observations),
            semantic_mode=self.semantic_mode, stage_timings=tuple(stages))

    def _answer(self, request: GenerationRequest, observations, stages) -> GenerationResult:
        stage_started = perf_counter()
        preflight = self.verifier.verify(request, _safe_refusal(request))
        stages.append(GenerationStageTiming("structural_preflight", 0, perf_counter() - stage_started))
        if preflight.status is VerificationStatus.REFUSED and preflight.issues:
            return GenerationResult(_safe_refusal(request), preflight, attempts=0)
        current = request
        telemetry = []
        prior_issues = ()
        for attempt in (1, 2):
            stage_started = perf_counter()
            try:
                answer = self.generator.generate(current)
            except Exception as exc:
                stages.append(GenerationStageTiming(
                    "generation" if attempt == 1 else "repair_generation",
                    attempt, perf_counter() - stage_started,
                ))
                if isinstance(exc, InvalidStructuredResponseError) and attempt == 1:
                    if getattr(exc, "telemetry", None):
                        telemetry.append(exc.telemetry)
                    current = self._format_repair(request, exc)
                    continue
                return _provider_failure_result(request, attempts=attempt,
                    prior_issues=prior_issues, error=exc, telemetry=tuple(telemetry))
            stages.append(GenerationStageTiming(
                "generation" if attempt == 1 else "repair_generation",
                attempt, perf_counter() - stage_started,
            ))
            if answer.telemetry:
                telemetry.append(answer.telemetry)
            if self.semantic_mode != "off":
                answer = repair_supporting_quotes(request, answer)
            answer = _normalize_issue_limitations(request, answer)
            verification = self._verify(request, answer, observations, stages, attempt)
            final = self._finish_attempt(request, answer, verification, attempt, tuple(telemetry))
            if final is not None:
                return final
            prior_issues = verification.issues
            current = replace(request, previous_answer=answer,
                repair_instructions=tuple(item.message for item in verification.issues))
        raise AssertionError("unreachable")

    async def answer_async(self, request: GenerationRequest) -> GenerationResult:
        observations: list[SemanticObservation] = []
        stages: list[GenerationStageTiming] = []
        result = await self._answer_async(request, observations, stages)
        return replace(result, semantic_observations=tuple(observations),
            semantic_mode=self.semantic_mode, stage_timings=tuple(stages))

    async def _answer_async(self, request: GenerationRequest, observations, stages) -> GenerationResult:
        stage_started = perf_counter()
        preflight = self.verifier.verify(request, _safe_refusal(request))
        stages.append(GenerationStageTiming("structural_preflight", 0, perf_counter() - stage_started))
        if preflight.status is VerificationStatus.REFUSED and preflight.issues:
            return GenerationResult(_safe_refusal(request), preflight, attempts=0)
        current = request
        telemetry = []
        prior_issues = ()
        for attempt in (1, 2):
            stage_started = perf_counter()
            try:
                answer = await _generate_async(self.generator, current)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                stages.append(GenerationStageTiming(
                    "generation" if attempt == 1 else "repair_generation",
                    attempt, perf_counter() - stage_started,
                ))
                if isinstance(exc, InvalidStructuredResponseError) and attempt == 1:
                    if getattr(exc, "telemetry", None):
                        telemetry.append(exc.telemetry)
                    current = self._format_repair(request, exc)
                    continue
                return _provider_failure_result(request, attempts=attempt,
                    prior_issues=prior_issues, error=exc, telemetry=tuple(telemetry))
            stages.append(GenerationStageTiming(
                "generation" if attempt == 1 else "repair_generation",
                attempt, perf_counter() - stage_started,
            ))
            if answer.telemetry:
                telemetry.append(answer.telemetry)
            if self.semantic_mode != "off":
                answer = repair_supporting_quotes(request, answer)
            answer = _normalize_issue_limitations(request, answer)
            verification = await self._verify_async(request, answer, observations, stages, attempt)
            final = self._finish_attempt(request, answer, verification, attempt, tuple(telemetry))
            if final is not None:
                return final
            prior_issues = verification.issues
            current = replace(request, previous_answer=answer,
                repair_instructions=tuple(item.message for item in verification.issues))
        raise AssertionError("unreachable")

    @staticmethod
    def _format_repair(request, error=None):
        error_code = getattr(error, "contract_code", "INVALID_STRUCTURED_RESPONSE")
        return replace(request, previous_answer=_safe_refusal(request), repair_instructions=(
            f"Phản hồi trước sai định dạng JSON ({error_code}) và đã bị loại bỏ. "
            "Sinh lại từ evidence theo đúng schema: "
            "answer là chuỗi rỗng; claims gồm text, evidence_ids, supporting_quotes, issue_ids; "
            "issue_resolutions có đúng một kết quả cho mỗi issue; limitations là mảng CHUỖI; "
            "confidence là high/medium/low. "
            "Không có Markdown hoặc trường thừa. Đây là lần sinh cuối cùng.",
        ))

    @staticmethod
    def _finish_attempt(request, answer, verification, attempt, telemetry):
        if verification.status is VerificationStatus.VERIFIED:
            if answer.issue_resolutions and all(
                item.status != "ANSWERED" for item in answer.issue_resolutions
            ):
                refusal = replace(
                    answer,
                    answer="Không đủ căn cứ để trả lời các vấn đề pháp lý đã xác định.",
                )
                return GenerationResult(
                    refusal,
                    VerificationResult(VerificationStatus.REFUSED),
                    attempt,
                    telemetry,
                )
            return GenerationResult(render_verified_answer(answer), verification, attempt, telemetry)
        if verification.status is VerificationStatus.REFUSED:
            return GenerationResult(_safe_refusal(request) if verification.issues else answer,
                verification, attempt, telemetry)
        if attempt == 2:
            return GenerationResult(_safe_refusal(request),
                VerificationResult(VerificationStatus.REFUSED, verification.issues), attempt, telemetry)
        return None

    def _verify(self, request, answer, observations, stages, attempt):
        started = perf_counter()
        structural = self.verifier.verify(request, answer)
        stages.append(GenerationStageTiming("structural_verification", attempt, perf_counter() - started))
        if structural.status is not VerificationStatus.VERIFIED or self.semantic_mode == "off":
            return structural
        started = perf_counter()
        review = self.semantic_verifier.review(request, answer)
        stages.append(GenerationStageTiming("semantic_judge", attempt, perf_counter() - started))
        if self.shadow_semantic_verifier is not None:
            started = perf_counter()
            shadow = self.shadow_semantic_verifier.review(request, answer)
            stages.append(GenerationStageTiming(
                "semantic_judge_candidate_shadow", attempt, perf_counter() - started
            ))
            observations.append(SemanticObservation(attempt, "candidate_shadow", shadow))
        return self._combine_review(structural, review, observations, attempt)

    async def _verify_async(self, request, answer, observations, stages, attempt):
        started = perf_counter()
        structural = self.verifier.verify(request, answer)
        stages.append(GenerationStageTiming("structural_verification", attempt, perf_counter() - started))
        if structural.status is not VerificationStatus.VERIFIED or self.semantic_mode == "off":
            return structural
        started = perf_counter()
        if self.shadow_semantic_verifier is None:
            review = await self.semantic_verifier.review_async(request, answer)
            shadow = None
        else:
            review, shadow = await asyncio.gather(
                self.semantic_verifier.review_async(request, answer),
                self.shadow_semantic_verifier.review_async(request, answer),
            )
        judge_seconds = perf_counter() - started
        stages.append(GenerationStageTiming("semantic_judge", attempt, judge_seconds))
        if shadow is not None:
            stages.append(GenerationStageTiming(
                "semantic_judge_candidate_shadow", attempt, judge_seconds
            ))
            observations.append(SemanticObservation(attempt, "candidate_shadow", shadow))
        return self._combine_review(structural, review, observations, attempt)

    def _combine_review(self, structural, review: SemanticReview, observations, attempt):
        observations.append(SemanticObservation(attempt, self.semantic_mode, review))
        if self.semantic_mode == "shadow" or review.passed:
            return structural
        if review.error:
            # An unavailable judge is not evidence of a legal contradiction.
            # Fail closed without spending another generator call on infrastructure failure.
            return VerificationResult(VerificationStatus.REFUSED, (
                VerificationIssue(VerificationCode.SEMANTIC_REVIEW_UNAVAILABLE,
                    "Chưa hoàn tất kiểm tra nội dung kết luận; cần thử lại khi bộ kiểm tra khả dụng."),
            ))
        if not review.plan_complete:
            return VerificationResult(VerificationStatus.REFUSED, (
                VerificationIssue(
                    VerificationCode.ISSUE_PLAN_INCOMPLETE,
                    "Kế hoạch vấn đề chưa bao phủ toàn bộ câu hỏi: " + review.plan_reason,
                ),
            ))
        issues = [VerificationIssue(VerificationCode.SUPPORTING_QUOTE_INVALID,
            f"Claim {item.claim_index}: {item.code} ({item.evidence_id}); cần đoạn trích nguyên văn từ evidence đã dẫn.",
            item.claim_index) for item in review.quote_issues]
        issues.extend(VerificationIssue(VerificationCode.CLAIM_NOT_SUPPORTED,
            f"Claim {item.claim_index}: {item.verdict}. {item.reason}", item.claim_index)
            for item in review.checks if item.verdict != "SUPPORTED")
        issues.extend(
            VerificationIssue(
                VerificationCode.ISSUE_NOT_COVERED,
                f"{item.issue_id}: {item.reason}",
            )
            for item in review.coverage_checks
            if item.verdict == "MISSING_OR_PARTIAL"
        )
        return VerificationResult(VerificationStatus.REPAIR_REQUIRED, tuple(issues))

    generate = answer


def render_verified_answer(answer: GeneratedAnswer) -> GeneratedAnswer:
    """Replace untrusted prose with a deterministic rendering of verified claims."""
    paragraphs = []
    for claim in answer.claims:
        text = claim.text.strip().rstrip(".")
        citations = " ".join(f"[{item}]" for item in claim.evidence_ids)
        paragraphs.append(f"{text}. {citations}".strip())
    return replace(answer, answer="\n\n".join(paragraphs))


LegalRAGService = GroundedRAGService


def _safe_refusal(request: GenerationRequest) -> GeneratedAnswer:
    if (
        request.temporal_intent == "historical"
        and not request.historical_content_available
    ):
        return GeneratedAnswer(
            answer=(
                "Không đủ căn cứ lịch sử đã được xác minh để trả lời tại thời điểm "
                f"{request.as_of.isoformat()}."
            ),
            claims=(),
            limitations=(
                "Nguồn hiện có là nội dung phiên bản hiện tại, không chứng minh nội dung lịch sử tại thời điểm được hỏi.",
            ),
            confidence="low",
        )
    return GeneratedAnswer(
        answer="Không đủ căn cứ trong evidence đã được xác minh để trả lời câu hỏi này.",
        claims=(),
        limitations=("Cần bổ sung nguồn pháp lý phù hợp và đã được xác minh.",),
        confidence="low",
    )


def _provider_failure_result(
    request: GenerationRequest,
    *,
    attempts: int,
    prior_issues: tuple[VerificationIssue, ...] = (),
    error: Exception | None = None,
    telemetry: tuple[GenerationTelemetry, ...] = (),
) -> GenerationResult:
    if getattr(error, "telemetry", None):
        telemetry = (*telemetry, error.telemetry)
    truncated = isinstance(error, TruncatedGenerationError)
    if truncated:
        code = VerificationCode.GENERATION_TRUNCATED
        message = "LLM provider đã chạm giới hạn output trước khi hoàn tất JSON."
    elif isinstance(error, GenerationTimeoutError):
        code = VerificationCode.GENERATION_TIMEOUT
        message = "LLM provider không hoàn tất trong thời hạn cho phép."
    elif isinstance(error, ProviderUnavailableError):
        code = VerificationCode.PROVIDER_UNAVAILABLE
        message = "LLM provider hiện không khả dụng."
    elif isinstance(error, InvalidStructuredResponseError):
        code = VerificationCode.INVALID_STRUCTURED_RESPONSE
        message = "LLM provider trả về structured response không hợp lệ."
    else:
        code = VerificationCode.GENERATION_FAILED
        message = (
            "LLM provider không tạo được structured response an toàn "
            f"ở lần thử {attempts}."
        )
    verification = VerificationResult(
        VerificationStatus.REFUSED,
        (*prior_issues,
            VerificationIssue(
                code,
                message,
            ),
        ),
    )
    return GenerationResult(
        _safe_refusal(request), verification, attempts, telemetry=telemetry
    )


async def _generate_async(
    generator: LegalAnswerGenerator,
    request: GenerationRequest,
) -> GeneratedAnswer:
    method = getattr(generator, "generate_async", None)
    if callable(method):
        return await method(request)
    return await _run_sync(generator.generate, request)


async def _run_sync(function, *args):
    """Run one sync provider call without reusing a poisoned worker thread."""
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        return await asyncio.get_running_loop().run_in_executor(
            executor, function, *args
        )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _telemetries(*answers: GeneratedAnswer) -> tuple[GenerationTelemetry, ...]:
    return tuple(
        answer.telemetry for answer in answers if answer.telemetry is not None
    )


def _normalize_issue_limitations(
    request: GenerationRequest, answer: GeneratedAnswer
) -> GeneratedAnswer:
    """Make every unresolved issue visible without asking the model to duplicate text."""
    unresolved = (
        item.explanation
        for item in answer.issue_resolutions
        if item.status != "ANSWERED"
    )
    evidence = {
        item.evidence_id: item for item in request.context.evidence
    }
    status_parts = []
    for claim in answer.claims:
        for evidence_id in claim.evidence_ids:
            item = evidence.get(evidence_id)
            if item is None:
                continue
            pairs = list(zip(
                item.related_document_numbers, item.related_statuses
            ))
            if item.citation.document_number:
                pairs.append((item.citation.document_number, item.citation.status))
            for number, status in pairs:
                normalized = status.upper()
                if normalized in {"EXPIRED", "REPEALED"}:
                    status_parts.append(
                        f"{number}: đã hết hiệu lực hoặc bị bãi bỏ"
                    )
                elif normalized == "SUSPENDED":
                    status_parts.append(f"{number}: đang bị đình chỉ")
                elif normalized == "PARTIALLY_EFFECTIVE":
                    status_parts.append(f"{number}: chỉ còn hiệu lực một phần")
    unique_status_parts = tuple(dict.fromkeys(status_parts))
    status_disclosures = (
        (
            f"Trạng thái tại ngày {request.as_of.isoformat()}: "
            + "; ".join(unique_status_parts)
            + "."
        )[:300],
    ) if unique_status_parts else ()
    limitations = tuple(dict.fromkeys(
        (*answer.limitations, *unresolved, *status_disclosures)
    ))
    return replace(answer, limitations=limitations[:5])
