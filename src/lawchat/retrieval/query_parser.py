from __future__ import annotations

import re
import os
from datetime import date, datetime
from typing import Callable
from zoneinfo import ZoneInfo

from .models import ParsedLegalQuery


_ISO_DATE_RE = re.compile(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)")
_VI_NUMERIC_DATE_RE = re.compile(
    r"(?<!\d)(\d{1,2})[/-](\d{1,2})[/-](\d{4})(?!\d)"
)
_VI_TEXT_DATE_RE = re.compile(
    r"\bngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})\b",
    re.IGNORECASE,
)
_DOCUMENT_NUMBER_RE = re.compile(
    r"(?<![\w/])\d{1,4}\s*/\s*(?:\d{4}\s*/\s*)?"
    r"[A-ZÀ-ỸĐ][A-ZÀ-ỸĐ0-9-]*",
    re.IGNORECASE,
)
_ARTICLE_RE = re.compile(r"\bđiều\s+(\d+[a-z]?)\b", re.IGNORECASE)
_CLAUSE_RE = re.compile(r"\bkhoản\s+(\d+[a-z]?)\b", re.IGNORECASE)
_POINT_RE = re.compile(r"\bđiểm\s+([a-zđ])\b", re.IGNORECASE)
_RELATIONSHIP_PATTERNS = {
    "AMENDS": re.compile(r"\b(?:sửa đổi|điều chỉnh)\b", re.IGNORECASE),
    "SUPPLEMENTS": re.compile(r"\bbổ sung\b", re.IGNORECASE),
    "REPEALS": re.compile(r"\b(?:bãi bỏ|hủy bỏ)\b", re.IGNORECASE),
    "REPLACES": re.compile(r"\bthay thế\b", re.IGNORECASE),
}


class AmbiguousTemporalQuery(ValueError):
    """Raised when one query contains multiple distinct dates without an override."""


class LegalDataCutoffExceeded(ValueError):
    """Raised when a request asks for law after the supported data cutoff."""


def _vietnam_today() -> date:
    cutoff = os.getenv("LAWCHAT_LEGAL_CUTOFF_DATE")
    return date.fromisoformat(cutoff) if cutoff else datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).date()


class LegalQueryParser:
    def __init__(
        self,
        *,
        today: Callable[[], date] = _vietnam_today,
        clean_semantic_query: bool = True,
        enforce_configured_cutoff: bool = True,
    ) -> None:
        self._today = today
        self.clean_semantic_query = clean_semantic_query
        self.enforce_configured_cutoff = enforce_configured_cutoff

    def parse(
        self,
        query: str,
        *,
        as_of: date | None = None,
    ) -> ParsedLegalQuery:
        normalized = " ".join(query.split()).strip()
        if not normalized:
            raise ValueError("query must not be empty")

        dates = self._extract_dates(normalized)
        if as_of is None and len(dates) > 1:
            rendered = ", ".join(item.isoformat() for item in dates)
            raise AmbiguousTemporalQuery(
                "query contains multiple dates; pass as_of explicitly "
                f"({rendered})"
            )
        configured_cutoff = (
            os.getenv("LAWCHAT_LEGAL_CUTOFF_DATE")
            if self.enforce_configured_cutoff
            else None
        )
        cutoff = date.fromisoformat(configured_cutoff) if configured_cutoff else None
        effective_as_of = as_of or (dates[0] if dates else cutoff or self._today())
        if configured_cutoff:
            assert cutoff is not None
            if effective_as_of > cutoff:
                raise LegalDataCutoffExceeded(
                    "Hệ thống chỉ hỗ trợ dữ liệu pháp luật đến ngày "
                    f"{cutoff.strftime('%d/%m/%Y')}; ngày yêu cầu là "
                    f"{effective_as_of.strftime('%d/%m/%Y')}."
                )

        document_numbers = tuple(
            value.replace(" ", "")
            for value in _unique_matches(_DOCUMENT_NUMBER_RE, normalized, upper=True)
        )
        articles = _unique_matches(_ARTICLE_RE, normalized)
        clauses = _unique_matches(_CLAUSE_RE, normalized)
        points = _unique_matches(_POINT_RE, normalized)
        relationship_types = tuple(
            relationship_type
            for relationship_type, pattern in _RELATIONSHIP_PATTERNS.items()
            if pattern.search(normalized)
        )
        relationship_direction = _relationship_direction(
            normalized,
            relationship_types,
        )
        has_retrieval_syntax = bool(
            dates or document_numbers or articles or clauses or points
        )
        semantic_query = (
            _build_semantic_query(normalized)
            if self.clean_semantic_query and has_retrieval_syntax
            else normalized
        )
        if not semantic_query:
            semantic_query = (
                "nội dung quy định pháp luật"
                if document_numbers or articles or clauses or points
                else normalized
            )

        return ParsedLegalQuery(
            original_query=normalized,
            semantic_query=semantic_query,
            as_of=effective_as_of,
            document_numbers=document_numbers,
            referenced_articles=articles,
            referenced_clauses=clauses,
            referenced_points=points,
            relationship_types=relationship_types,
            relationship_direction=relationship_direction,
            has_explicit_date=bool(dates),
        )

    @staticmethod
    def _extract_dates(query: str) -> tuple[date, ...]:
        found: list[date] = []
        for match in _ISO_DATE_RE.finditer(query):
            found.append(_safe_date(int(match[1]), int(match[2]), int(match[3])))
        for match in _VI_NUMERIC_DATE_RE.finditer(query):
            found.append(_safe_date(int(match[3]), int(match[2]), int(match[1])))
        for match in _VI_TEXT_DATE_RE.finditer(query):
            found.append(_safe_date(int(match[3]), int(match[2]), int(match[1])))
        return tuple(dict.fromkeys(found))


def _safe_date(year: int, month: int, day: int) -> date:
    try:
        return date(year, month, day)
    except ValueError as exc:
        raise ValueError(f"invalid date in query: {day:02d}/{month:02d}/{year}") from exc


def _unique_matches(
    pattern: re.Pattern[str],
    value: str,
    *,
    upper: bool = False,
) -> tuple[str, ...]:
    matches = [
        match.group(1) if match.lastindex else match.group(0)
        for match in pattern.finditer(value)
    ]
    normalized = [item.upper() if upper else item.casefold() for item in matches]
    return tuple(dict.fromkeys(normalized))


def _build_semantic_query(query: str) -> str:
    value = query
    for pattern in (
        _ISO_DATE_RE,
        _VI_NUMERIC_DATE_RE,
        _VI_TEXT_DATE_RE,
        _DOCUMENT_NUMBER_RE,
        _ARTICLE_RE,
        _CLAUSE_RE,
        _POINT_RE,
    ):
        value = pattern.sub(" ", value)
    value = re.sub(
        r"^\s*(?:tại|vào)\s+(?:ngày\s*)?[,;:]?\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"^\s*ngày\s*[,;:]?\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"^\s*(?:pháp luật\s+)?quy định\s+(?:đang\s+áp\s+dụng\s+)?"
        r"(?:(?:như\s+thế\s+nào|liên\s+quan\s+đến)\s+)?(?:về\s+)?",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"^\s*(?:tôi\s+cần\s+tìm\s+)?quy định\s+pháp\s+luật\s+về\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"\b(?:của\s+)?văn\s+bản\s*[,;:]?",
        " ",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"\b(?:nghị\s+định|bộ\s+luật|luật)\b\s*[,;:]?",
        " ",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"^\s*(?:từng\s+)?quy\s+định\s+(?:nội\s+dung\s+)?về\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"\s+(?:hiện\s+)?(?:còn|đã\s+hết|hết)\s+hiệu\s+lực\s+không\s*[?.!]*$",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"(?:được\s+pháp\s+luật\s+điều\s+chỉnh\s+ra\s+sao|"
        r"quy\s+định\s+(?:nội\s+dung\s+)?gì|là\s+gì)\s*[?.!]*$",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\s+", " ", value).strip(" ,;:.?!-")
    return value


def _relationship_direction(
    query: str,
    relationship_types: tuple[str, ...],
) -> str | None:
    if not relationship_types:
        return None
    document = _DOCUMENT_NUMBER_RE.search(query)
    relationship_positions = [
        match.start()
        for relationship_type in relationship_types
        if (match := _RELATIONSHIP_PATTERNS[relationship_type].search(query))
    ]
    if not document or not relationship_positions:
        return None
    return "outgoing" if document.start() < min(relationship_positions) else "incoming"
