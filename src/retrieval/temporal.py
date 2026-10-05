from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from database.models import LegalStatus

from .models import (
    ACTIVE_LEGAL_STATUSES,
    ParsedLegalQuery,
    RetrievalRequest,
)


class TemporalIntent(str, Enum):
    CURRENT_LAW = "current_law"
    STATUS_LOOKUP = "status_lookup"


ALL_LEGAL_STATUSES = tuple(item.value for item in LegalStatus)

_STATUS_LOOKUP_RE = re.compile(
    r"\b(?:hiện\s+)?(?:còn|đã\s+hết|hết)\s+hiệu\s+lực\b|"
    r"\btrạng\s+thái\s+hiệu\s+lực\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class TemporalDecision:
    intent: TemporalIntent
    allowed_statuses: tuple[str, ...]
    preserve_explicit_documents: bool
    expand_replacements: bool
    explicit_as_of: bool
    warnings: tuple[str, ...] = ()


class TemporalPolicy:
    """Turns query intent into one deterministic legal-status policy.

    LawChat only answers from law in force at the reference date. Dates in a
    question are case facts, never a request to read historical law.
    """

    def decide(
        self,
        parsed: ParsedLegalQuery,
        request: RetrievalRequest,
    ) -> TemporalDecision:
        if request.resolved_temporal_intent is not None:
            intent = TemporalIntent(request.resolved_temporal_intent)
        elif _STATUS_LOOKUP_RE.search(parsed.original_query):
            intent = TemporalIntent.STATUS_LOOKUP
        else:
            intent = TemporalIntent.CURRENT_LAW

        preserve_explicit = bool(parsed.document_numbers or request.document_numbers)
        if request.statuses is not None:
            statuses = request.statuses
        elif intent is TemporalIntent.STATUS_LOOKUP or preserve_explicit:
            statuses = ALL_LEGAL_STATUSES
        else:
            statuses = ACTIVE_LEGAL_STATUSES

        return TemporalDecision(
            intent=intent,
            allowed_statuses=statuses,
            preserve_explicit_documents=preserve_explicit,
            expand_replacements=preserve_explicit,
            explicit_as_of=request.as_of is not None,
        )
