from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from database.models import LegalStatus

from .models import (
    ACTIVE_LEGAL_STATUSES,
    ParsedLegalQuery,
    RetrievalRequest,
    RetrievalResponse,
)
from .query_parser import LegalQueryParser


class TemporalIntent(str, Enum):
    CURRENT_LAW = "current_law"
    STATUS_LOOKUP = "status_lookup"
    HISTORICAL = "historical"


ALL_LEGAL_STATUSES = tuple(item.value for item in LegalStatus)

_STATUS_LOOKUP_RE = re.compile(
    r"\b(?:hiện\s+)?(?:còn|đã\s+hết|hết)\s+hiệu\s+lực\b|"
    r"\btrạng\s+thái\s+hiệu\s+lực\b",
    re.IGNORECASE,
)
_CURRENT_LAW_RE = re.compile(
    r"\b(?:hiện\s+nay|hiện\s+hành|đang\s+áp\s+dụng|mới\s+nhất)\b",
    re.IGNORECASE,
)
_HISTORICAL_RE = re.compile(
    r"\b(?:vào|tại)\s+(?:ngày|thời\s+điểm)\b|\btrước\s+ngày\b|\btừng\s+quy\s+định\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class TemporalDecision:
    intent: TemporalIntent
    allowed_statuses: tuple[str, ...]
    preserve_explicit_documents: bool
    expand_replacements: bool
    historical_content_required: bool
    explicit_as_of: bool
    warnings: tuple[str, ...] = ()


class TemporalPolicy:
    """Turns query intent into one deterministic legal-status policy."""

    def decide(
        self,
        parsed: ParsedLegalQuery,
        request: RetrievalRequest,
    ) -> TemporalDecision:
        query = parsed.original_query
        if request.resolved_temporal_intent is not None:
            intent = TemporalIntent(request.resolved_temporal_intent)
        elif _STATUS_LOOKUP_RE.search(query):
            intent = TemporalIntent.STATUS_LOOKUP
        elif _CURRENT_LAW_RE.search(query):
            intent = TemporalIntent.CURRENT_LAW
        elif parsed.has_explicit_date or _HISTORICAL_RE.search(query):
            intent = TemporalIntent.HISTORICAL
        else:
            intent = TemporalIntent.CURRENT_LAW

        preserve_explicit = bool(parsed.document_numbers or request.document_numbers)
        if request.statuses is not None:
            statuses = request.statuses
        elif intent is TemporalIntent.STATUS_LOOKUP or preserve_explicit:
            statuses = ALL_LEGAL_STATUSES
        else:
            statuses = ACTIVE_LEGAL_STATUSES

        historical = intent is TemporalIntent.HISTORICAL
        # Audit fix E11 (Phase 5): the warning below is correct for
        # ``historical_content_available=False`` (the current production
        # state where the historical index is not built yet), but becomes
        # actively misleading when historical corpus is online. The flag
        # ``historical_index_enabled`` reflects the rollout state.
        try:
            from config import get_feature_flags
        except (ImportError, ValueError):
            from ..config import get_feature_flags

        historical_content_ready = get_feature_flags().historical_index_enabled
        if historical and not historical_content_ready:
            warnings = (
                (
                    "historical status is enforced at as_of, but retrieval content "
                    "comes from the current indexed document version"
                ),
            )
        else:
            warnings = ()
        return TemporalDecision(
            intent=intent,
            allowed_statuses=statuses,
            preserve_explicit_documents=preserve_explicit,
            expand_replacements=(
                intent in {TemporalIntent.CURRENT_LAW, TemporalIntent.STATUS_LOOKUP}
                and preserve_explicit
            ),
            historical_content_required=historical,
            explicit_as_of=parsed.has_explicit_date or request.as_of is not None,
            warnings=warnings,
        )


class RetrievalTarget(Protocol):
    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse: ...


class TemporalRetrievalRouter:
    """Route content questions without ever falling back across time domains."""

    def __init__(
        self,
        current: RetrievalTarget,
        historical: RetrievalTarget | None = None,
        *,
        query_parser: LegalQueryParser | None = None,
        temporal_policy: TemporalPolicy | None = None,
    ) -> None:
        self.current = current
        self.historical = historical
        self.query_parser = query_parser or LegalQueryParser()
        self.temporal_policy = temporal_policy or TemporalPolicy()

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        parsed = self.query_parser.parse(request.query, as_of=request.as_of)
        decision = self.temporal_policy.decide(parsed, request)
        if decision.intent is not TemporalIntent.HISTORICAL:
            return self.current.retrieve(request)
        if self.historical is not None:
            return self.historical.retrieve(request)
        return RetrievalResponse(
            query=parsed.original_query,
            semantic_query=parsed.semantic_query,
            as_of=parsed.as_of,
            results=(),
            searched_candidates=0,
            rejected_candidates=0,
            warnings=("historical content collection is unavailable",),
            temporal_intent=decision.intent.value,
            temporal_explicit_as_of=decision.explicit_as_of,
            historical_content_available=False,
        )
