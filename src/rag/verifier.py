from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlsplit, urlunsplit

from .generator import GeneratedAnswer, GeneratedClaim, GenerationRequest
from .context_models import Evidence
from config import FeatureFlags, get_feature_flags


class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    REPAIR_REQUIRED = "REPAIR_REQUIRED"
    REFUSED = "REFUSED"


class VerificationCode(str, Enum):
    CLAIM_NOT_SUPPORTED = "CLAIM_NOT_SUPPORTED"
    SUPPORTING_QUOTE_INVALID = "SUPPORTING_QUOTE_INVALID"
    SEMANTIC_REVIEW_UNAVAILABLE = "SEMANTIC_REVIEW_UNAVAILABLE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    GENERATION_FAILED = "GENERATION_FAILED"
    GENERATION_TRUNCATED = "GENERATION_TRUNCATED"
    GENERATION_TIMEOUT = "GENERATION_TIMEOUT"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    INVALID_STRUCTURED_RESPONSE = "INVALID_STRUCTURED_RESPONSE"
    EMPTY_CLAIM = "EMPTY_CLAIM"
    MISSING_CLAIMS = "MISSING_CLAIMS"
    MISSING_CITATION = "MISSING_CITATION"
    UNKNOWN_EVIDENCE_ID = "UNKNOWN_EVIDENCE_ID"
    DOCUMENT_NUMBER_NOT_IN_EVIDENCE = "DOCUMENT_NUMBER_NOT_IN_EVIDENCE"
    DOCUMENT_NUMBER_CITATION_MISMATCH = "DOCUMENT_NUMBER_CITATION_MISMATCH"
    PROVISION_CITATION_MISMATCH = "PROVISION_CITATION_MISMATCH"
    STATUS_CITATION_MISMATCH = "STATUS_CITATION_MISMATCH"
    STATUS_DISCLOSURE_MISSING = "STATUS_DISCLOSURE_MISSING"
    PROVISION_STATUS_LIMITATION_MISSING = "PROVISION_STATUS_LIMITATION_MISSING"
    UNRESOLVED_PROVISION_STATUS = "UNRESOLVED_PROVISION_STATUS"
    AS_OF_MISMATCH = "AS_OF_MISMATCH"
    UNTRUSTED_URL = "UNTRUSTED_URL"
    INVALID_CONFIDENCE = "INVALID_CONFIDENCE"
    EMPTY_ANSWER = "EMPTY_ANSWER"
    UNKNOWN_ISSUE_ID = "UNKNOWN_ISSUE_ID"
    MISSING_ISSUE_RESOLUTION = "MISSING_ISSUE_RESOLUTION"
    DUPLICATE_ISSUE_RESOLUTION = "DUPLICATE_ISSUE_RESOLUTION"
    INVALID_ISSUE_RESOLUTION = "INVALID_ISSUE_RESOLUTION"
    ISSUE_PLAN_INCOMPLETE = "ISSUE_PLAN_INCOMPLETE"
    ISSUE_NOT_COVERED = "ISSUE_NOT_COVERED"
    CASE_SPECIFIC_PREDICTION_UNSUPPORTED = "CASE_SPECIFIC_PREDICTION_UNSUPPORTED"


@dataclass(frozen=True, slots=True)
class VerificationIssue:
    code: VerificationCode
    message: str
    claim_index: int | None = None


@dataclass(frozen=True, slots=True)
class VerificationResult:
    status: VerificationStatus
    issues: tuple[VerificationIssue, ...] = ()

    @property
    def verified(self) -> bool:
        return self.status is VerificationStatus.VERIFIED


_DOCUMENT_NUMBER_RE = re.compile(
    r"(?<!\w)\d{1,4}\s*/\s*\d{4}\s*/\s*[A-ZÀ-ỸĐ0-9]+"
    r"(?:\s*-\s*[A-ZÀ-ỸĐ0-9]+)*(?!\w)",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>()\[\]{}\"']+", re.IGNORECASE)
_EVIDENCE_ID_RE = re.compile(r"\b[EG][1-9][0-9]*\b")
_PROVISION_PATTERNS = {
    "article": re.compile(r"\bĐiều\s+([0-9]+[A-Za-zĐđ]?)\b", re.IGNORECASE),
    "clause": re.compile(r"\bKhoản\s+([0-9]+[A-Za-zĐđ]?)\b", re.IGNORECASE),
    "point": re.compile(
        r"(?<!thời )(?<!địa )\bĐiểm\s+([0-9]+|[A-Za-zĐđ])\b",
        re.IGNORECASE,
    ),
}
_REFUSAL_RE = re.compile(
    r"\b(?:không đủ|thiếu)\s+(?:căn cứ|bằng chứng|dữ liệu)|"
    r"\bkhông thể (?:kết luận|trả lời|xác định)\b|\btừ chối\b",
    re.IGNORECASE,
)
_STATUS_PATTERNS: tuple[tuple[re.Pattern[str], frozenset[str]], ...] = (
    (
        re.compile(r"\b(?:hết hiệu lực một phần|còn hiệu lực một phần)\b", re.IGNORECASE),
        frozenset({"PARTIALLY_EFFECTIVE"}),
    ),
    (
        re.compile(r"\b(?:chưa|chưa bắt đầu)\s+có hiệu lực\b", re.IGNORECASE),
        frozenset({"NOT_YET_EFFECTIVE", "DRAFT"}),
    ),
    (
        re.compile(r"\b(?:không còn|đã hết|hết)\s+hiệu lực\b|\b(?:bị|đã)\s+bãi bỏ\b", re.IGNORECASE),
        frozenset({"EXPIRED", "REPEALED"}),
    ),
    (
        re.compile(r"\b(?:bị\s+)?(?:tạm đình chỉ|đình chỉ|tạm ngưng)\b", re.IGNORECASE),
        frozenset({"SUSPENDED"}),
    ),
    (
        re.compile(
            r"(?<!không )(?<!chưa )\b(?:vẫn\s+)?(?:còn|đang)\s+(?:có\s+)?hiệu lực\b|"
            r"(?<!không )(?<!chưa )\bcó hiệu lực\b|"
            r"\bhiện hành\b",
            re.IGNORECASE,
        ),
        frozenset({"EFFECTIVE", "PARTIALLY_EFFECTIVE"}),
    ),
)


class GroundingVerifier:
    """Deterministic structural and legal-citation verifier."""

    def verify(
        self,
        request: GenerationRequest,
        answer: GeneratedAnswer,
        *,
        feature_flags: FeatureFlags | None = None,
    ) -> VerificationResult:
        flags = feature_flags or get_feature_flags()
        evidence_by_id = {item.evidence_id: item for item in request.context.evidence}
        combined = _combined_text(answer)
        global_status_expectations = _status_expectations(
            "\n".join((answer.answer, *answer.limitations))
        )

        if requires_case_specific_prediction_refusal(request.question):
            return VerificationResult(
                VerificationStatus.REFUSED,
                (
                    VerificationIssue(
                        VerificationCode.CASE_SPECIFIC_PREDICTION_UNSUPPORTED,
                        "Câu hỏi yêu cầu dự đoán hoặc kết luận chắc chắn cho vụ việc cụ thể nhưng chưa có đủ dữ kiện kiểm chứng.",
                    ),
                ),
            )

        document_partial = (
            request.temporal_intent == "current_law"
            and any(
                item.citation.status_scope == "document"
                and item.citation.status.upper() == "PARTIALLY_EFFECTIVE"
                for item in evidence_by_id.values()
            )
        )
        if request.temporal_intent == "current_law" and any(
            item.citation.status_scope == "provision"
            and item.citation.status.upper() in {"REPEALED", "PARTIALLY_EFFECTIVE"}
            for item in evidence_by_id.values()
        ):
            return VerificationResult(
                VerificationStatus.REFUSED,
                (
                    VerificationIssue(
                        VerificationCode.UNRESOLVED_PROVISION_STATUS,
                        (
                            "Hiệu lực của Điều/Khoản/Điểm trong văn bản đã được xác minh "
                            "là đã hết hiệu lực một phần/toàn phần; evidence bị từ chối."
                        ),
                    ),
                ),
            )

        issues: list[VerificationIssue] = []
        if document_partial and flags.fail_closed_partial:
            has_provision_evidence = any(
                item.citation.status_scope == "provision"
                for item in evidence_by_id.values()
            )
            if not has_provision_evidence:
                return VerificationResult(
                    VerificationStatus.REFUSED,
                    (
                        VerificationIssue(
                            VerificationCode.UNRESOLVED_PROVISION_STATUS,
                            (
                                "Văn bản có trạng thái PARTIALLY_EFFECTIVE ở cấp văn bản "
                                "và evidence không chứa trạng thái hiệu lực cấp điều khoản. "
                                "Từ chối để tránh trả lời dựa trên điều khoản đã hết hiệu lực."
                            ),
                        ),
                    ),
                )

        if not evidence_by_id or not any(item.text.strip() for item in evidence_by_id.values()):
            issues = [
                VerificationIssue(
                    VerificationCode.INSUFFICIENT_EVIDENCE,
                    "Không có evidence để hỗ trợ câu trả lời.",
                )
            ]
            return VerificationResult(VerificationStatus.REFUSED, tuple(issues))

        if _is_safe_refusal(answer):
            return VerificationResult(VerificationStatus.REFUSED)

        if not answer.claims and not answer.issue_resolutions:
            issues.append(
                VerificationIssue(
                    VerificationCode.MISSING_CLAIMS,
                    "Câu trả lời có kết luận nhưng không khai báo claim pháp lý có citation.",
                )
            )
        if answer.confidence not in {"high", "medium", "low"}:
            issues.append(
                VerificationIssue(
                    VerificationCode.INVALID_CONFIDENCE,
                    "Confidence không thuộc high, medium hoặc low.",
                )
            )

        known_numbers = {
            _normalize_identifier(item.citation.document_number)
            for item in evidence_by_id.values()
            if item.citation.document_number
        }
        known_numbers.update(
            _normalize_identifier(number)
            for item in evidence_by_id.values()
            for number in item.related_document_numbers
        )
        known_numbers.update(
            _normalize_identifier(number)
            for item in evidence_by_id.values()
            for number in _document_numbers(item.text)
        )
        for document_number in _document_numbers(combined):
            if _normalize_identifier(document_number) not in known_numbers:
                issues.append(
                    VerificationIssue(
                        VerificationCode.DOCUMENT_NUMBER_NOT_IN_EVIDENCE,
                        f"Số hiệu {document_number!r} không có trong evidence.",
                    )
                )

        for evidence_id in dict.fromkeys(_EVIDENCE_ID_RE.findall(combined)):
            if evidence_id not in evidence_by_id:
                issues.append(
                    VerificationIssue(
                        VerificationCode.UNKNOWN_EVIDENCE_ID,
                        f"Evidence ID {evidence_id!r} không tồn tại.",
                    )
                )

        for field, pattern in _PROVISION_PATTERNS.items():
            evidence_values = _evidence_provision_values(
                evidence_by_id.values(),
                field,
            )
            for match in pattern.finditer(answer.answer):
                value = match.group(1)
                if _normalize_provision(value) not in evidence_values:
                    issues.append(
                        VerificationIssue(
                            VerificationCode.PROVISION_CITATION_MISMATCH,
                            f"{field} {value!r} trong answer không có trong evidence.",
                        )
                    )

        evidence_statuses = {
            item.citation.status.upper() for item in evidence_by_id.values()
        }
        for expected in _status_expectations(answer.answer):
            if evidence_statuses.isdisjoint(expected):
                issues.append(
                    VerificationIssue(
                        VerificationCode.STATUS_CITATION_MISMATCH,
                        "Trạng thái hiệu lực trong answer không khớp evidence tại as_of.",
                    )
                )

        allowed_urls = {
            _normalize_url(item.citation.source_url)
            for item in evidence_by_id.values()
            if item.citation.source_url
        }
        for url in _urls(combined):
            if _normalize_url(url) not in allowed_urls:
                issues.append(
                    VerificationIssue(
                        VerificationCode.UNTRUSTED_URL,
                        "Câu trả lời chứa URL không có trong evidence.",
                    )
                )

        for index, claim in enumerate(answer.claims):
            issues.extend(
                self._verify_claim(
                    index,
                    claim,
                    evidence_by_id,
                    request,
                    global_status_expectations,
                )
            )

        issues.extend(self._verify_issue_coverage_contract(request, answer))
        return VerificationResult(
            VerificationStatus.REPAIR_REQUIRED if issues else VerificationStatus.VERIFIED,
            tuple(_deduplicate_issues(issues)),
        )

    def _verify_issue_coverage_contract(
        self, request: GenerationRequest, answer: GeneratedAnswer
    ) -> list[VerificationIssue]:
        if not request.issues:
            return []
        result: list[VerificationIssue] = []
        known = {item.issue_id for item in request.issues}
        resolution_ids = [item.issue_id for item in answer.issue_resolutions]
        for issue_id in sorted(known - set(resolution_ids)):
            result.append(VerificationIssue(
                VerificationCode.MISSING_ISSUE_RESOLUTION,
                f"Chưa khai báo kết quả cho vấn đề {issue_id}.",
            ))
        for issue_id in sorted({item for item in resolution_ids if resolution_ids.count(item) > 1}):
            result.append(VerificationIssue(
                VerificationCode.DUPLICATE_ISSUE_RESOLUTION,
                f"Vấn đề {issue_id} có nhiều hơn một kết quả.",
            ))
        for index, claim in enumerate(answer.claims):
            if not claim.issue_ids:
                result.append(VerificationIssue(
                    VerificationCode.UNKNOWN_ISSUE_ID,
                    "Claim chưa chỉ rõ vấn đề pháp lý mà nó trả lời.",
                    index,
                ))
            for issue_id in claim.issue_ids:
                if issue_id not in known:
                    result.append(VerificationIssue(
                        VerificationCode.UNKNOWN_ISSUE_ID,
                        f"Claim tham chiếu vấn đề không tồn tại: {issue_id}.",
                        index,
                    ))
        for resolution in answer.issue_resolutions:
            if resolution.issue_id not in known:
                result.append(VerificationIssue(
                    VerificationCode.UNKNOWN_ISSUE_ID,
                    f"Kết quả tham chiếu vấn đề không tồn tại: {resolution.issue_id}.",
                ))
                continue
            valid_indexes = all(0 <= item < len(answer.claims) for item in resolution.claim_indexes)
            linked = valid_indexes and all(
                resolution.issue_id in answer.claims[item].issue_ids
                for item in resolution.claim_indexes
            )
            if resolution.status == "ANSWERED":
                if not resolution.claim_indexes or not linked:
                    result.append(VerificationIssue(
                        VerificationCode.INVALID_ISSUE_RESOLUTION,
                        f"{resolution.issue_id} được đánh dấu ANSWERED nhưng không dẫn claim hợp lệ.",
                    ))
            elif resolution.claim_indexes:
                result.append(VerificationIssue(
                    VerificationCode.INVALID_ISSUE_RESOLUTION,
                    f"{resolution.issue_id} chưa được trả lời nhưng vẫn dẫn claim.",
                ))
            elif resolution.explanation not in answer.limitations:
                result.append(VerificationIssue(
                    VerificationCode.INVALID_ISSUE_RESOLUTION,
                    f"Giới hạn của {resolution.issue_id} chưa được công bố trong limitations.",
                ))
        referenced_claims = {
            index
            for resolution in answer.issue_resolutions
            if resolution.status == "ANSWERED"
            for index in resolution.claim_indexes
            if 0 <= index < len(answer.claims)
        }
        for index in sorted(set(range(len(answer.claims))) - referenced_claims):
            result.append(VerificationIssue(
                VerificationCode.INVALID_ISSUE_RESOLUTION,
                "Claim chưa được một issue_resolution ANSWERED tham chiếu.",
                index,
            ))
        return result

    def _verify_claim(
        self,
        index: int,
        claim: GeneratedClaim,
        evidence_by_id: dict[str, Evidence],
        request: GenerationRequest,
        global_status_expectations: tuple[frozenset[str], ...],
    ) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []
        if not claim.text.strip():
            issues.append(
                VerificationIssue(
                    VerificationCode.EMPTY_CLAIM,
                    "Claim pháp lý đang để trống.",
                    index,
                )
            )
        if not claim.evidence_ids:
            issues.append(
                VerificationIssue(
                    VerificationCode.MISSING_CITATION,
                    "Claim pháp lý không có evidence ID.",
                    index,
                )
            )
            return issues

        unknown_ids = tuple(
            evidence_id
            for evidence_id in claim.evidence_ids
            if evidence_id not in evidence_by_id
        )
        for evidence_id in unknown_ids:
            issues.append(
                VerificationIssue(
                    VerificationCode.UNKNOWN_EVIDENCE_ID,
                    f"Evidence ID {evidence_id!r} không tồn tại.",
                    index,
                )
            )
        cited = [
            evidence_by_id[evidence_id]
            for evidence_id in dict.fromkeys(claim.evidence_ids)
            if evidence_id in evidence_by_id
        ]
        if not cited:
            return issues

        for evidence in cited:
            if evidence.citation.as_of != request.as_of:
                issues.append(
                    VerificationIssue(
                        VerificationCode.AS_OF_MISMATCH,
                        f"{evidence.evidence_id} không có trạng thái tại as_of yêu cầu.",
                        index,
                    )
                )

        cited_numbers = {
            _normalize_identifier(item.citation.document_number)
            for item in cited
            if item.citation.document_number
        }
        cited_numbers.update(
            _normalize_identifier(number)
            for item in cited
            for number in item.related_document_numbers
        )
        cited_numbers.update(
            _normalize_identifier(number)
            for item in cited
            for number in _document_numbers(item.text)
        )
        for document_number in _document_numbers(claim.text):
            if _normalize_identifier(document_number) not in cited_numbers:
                issues.append(
                    VerificationIssue(
                        VerificationCode.DOCUMENT_NUMBER_CITATION_MISMATCH,
                        f"Số hiệu {document_number!r} không khớp citation của claim.",
                        index,
                    )
                )

        for field, pattern in _PROVISION_PATTERNS.items():
            cited_values = _evidence_provision_values(cited, field)
            for match in pattern.finditer(claim.text):
                value = match.group(1)
                if _normalize_provision(value) not in cited_values:
                    issues.append(
                        VerificationIssue(
                            VerificationCode.PROVISION_CITATION_MISMATCH,
                            f"{field} {value!r} không khớp citation của claim.",
                            index,
                        )
                    )

        cited_statuses = {item.citation.status.upper() for item in cited}
        cited_statuses.update(
            status.upper()
            for item in cited
            for status in item.related_statuses
        )
        status_expectations = _status_expectations(claim.text)
        for expected in status_expectations:
            if cited_statuses.isdisjoint(expected):
                issues.append(
                    VerificationIssue(
                        VerificationCode.STATUS_CITATION_MISMATCH,
                        "Trạng thái hiệu lực trong claim không khớp evidence tại as_of.",
                        index,
                    )
                )
        for status in cited_statuses:
            required_disclosure = _required_status_disclosure(status)
            if required_disclosure and not any(
                not expected.isdisjoint(required_disclosure)
                for expected in (
                    *status_expectations,
                    *global_status_expectations,
                )
            ):
                issues.append(
                    VerificationIssue(
                        VerificationCode.STATUS_DISCLOSURE_MISSING,
                        f"Claim chưa công bố trạng thái {status} của evidence được trích dẫn.",
                        index,
                    )
                )
        return issues



def _combined_text(answer: GeneratedAnswer) -> str:
    return "\n".join(
        (answer.answer, *(claim.text for claim in answer.claims), *answer.limitations)
    )


def _is_safe_refusal(answer: GeneratedAnswer) -> bool:
    return not answer.claims and answer.confidence == "low" and bool(
        _REFUSAL_RE.search(answer.answer)
    )


_UNSUPPORTED_PREDICTION_PATTERNS = (
    re.compile(r"\bdự\s+đoán\s+chính\s+xác\b", re.IGNORECASE),
    re.compile(
        r"\b(?:thẩm\s+phán|tòa\s+án)\b.{0,100}\b(?:chính\s+xác\s+)?"
        r"(?:bao\s+nhiêu|mức\s+bồi\s+thường)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bchắc\s+chắn\b.{0,80}\b(?:thắng|thua|phạm\s+tội|vô\s+hiệu|"
        r"trái\s+pháp\s+luật|bồi\s+thường|trả\s+lại)\b|"
        r"\b(?:thắng|thua|phạm\s+tội|vô\s+hiệu|trái\s+pháp\s+luật|"
        r"bồi\s+thường|trả\s+lại)\b.{0,80}\bchắc\s+chắn\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:không\s+có|không\s+cung\s+cấp|chưa\s+cung\s+cấp|"
        r"chưa\s+được\s+cung\s+cấp)\b.{0,120}\b(?:chính\s+xác|"
        r"kết\s+luận|khẳng\s+định|xác\s+nhận|tính)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:kết\s+luận|khẳng\s+định|xác\s+nhận|tính)\b.{0,120}"
        r"\b(?:không\s+có|không\s+cung\s+cấp|chưa\s+cung\s+cấp|"
        r"chưa\s+được\s+cung\s+cấp)\b",
        re.IGNORECASE,
    ),
)


def requires_case_specific_prediction_refusal(question: str) -> bool:
    normalized = " ".join(question.split())
    return any(pattern.search(normalized) for pattern in _UNSUPPORTED_PREDICTION_PATTERNS)




def _document_numbers(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(match.group(0) for match in _DOCUMENT_NUMBER_RE.finditer(text)))


def _urls(text: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(match.group(0).rstrip(".,;:!?)]}") for match in _URL_RE.finditer(text))
    )


def _status_expectations(text: str) -> tuple[frozenset[str], ...]:
    return tuple(expected for pattern, expected in _STATUS_PATTERNS if pattern.search(text))


def _evidence_provision_values(
    evidence: Iterable[Evidence],
    field: str,
) -> set[str]:
    items = tuple(evidence)
    values = {
        _normalize_provision(getattr(item.citation, field))
        for item in items
        if getattr(item.citation, field)
    }
    pattern = _PROVISION_PATTERNS[field]
    values.update(
        _normalize_provision(match.group(1))
        for item in items
        for match in pattern.finditer(item.text)
    )
    return values


def _required_status_disclosure(status: str) -> frozenset[str]:
    groups = {
        "PARTIALLY_EFFECTIVE": frozenset({"PARTIALLY_EFFECTIVE"}),
        "SUSPENDED": frozenset({"SUSPENDED"}),
        "EXPIRED": frozenset({"EXPIRED", "REPEALED"}),
        "REPEALED": frozenset({"EXPIRED", "REPEALED"}),
        "NOT_YET_EFFECTIVE": frozenset({"NOT_YET_EFFECTIVE", "DRAFT"}),
        "DRAFT": frozenset({"NOT_YET_EFFECTIVE", "DRAFT"}),
    }
    return groups.get(status, frozenset())


def _normalize_identifier(value: str | None) -> str:
    return re.sub(r"\s+", "", value or "").casefold()


def _normalize_provision(value: str | None) -> str:
    return (value or "").strip().casefold()


def _normalize_url(value: str) -> str:
    candidate = value.strip().rstrip("/")
    if candidate.casefold().startswith("www."):
        candidate = "https://" + candidate
    parts = urlsplit(candidate)
    scheme = parts.scheme.casefold()
    host = (parts.hostname or "").casefold()
    try:
        port_value = parts.port
    except ValueError:
        return candidate.casefold()
    port = f":{port_value}" if port_value else ""
    path = parts.path.rstrip("/")
    return urlunsplit((scheme, host + port, path, parts.query, ""))


def _deduplicate_issues(issues: list[VerificationIssue]) -> list[VerificationIssue]:
    return list(dict.fromkeys(issues))
