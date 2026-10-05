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
# A document number either carries a year ("45/2019/QH14", any case) or an
# upper-case symbol of at least two letters ("88/CP", "15/NQ-TW"). Requiring
# one of the two keeps rates such as "200/ngày" from becoming hard filters.
_DOCUMENT_NUMBER_RE = re.compile(
    r"(?<![\w/])\d{1,4}\s*/\s*(?:"
    r"\d{4}\s*/\s*(?i:[A-ZĐÐ][A-ZĐÐ0-9-]*)"
    r"|[A-ZĐÐ]{2,}(?:-[A-Za-zĐđÐ0-9]+)*(?!\w)"
    r")"
)
# A year-specific name of a replaced law also resolves to the law in force,
# because LawChat answers from current law and discloses the old one's status.
_COMMON_LAW_ALIASES: tuple[tuple[re.Pattern[str], str | tuple[str, ...]], ...] = (
    (re.compile(r"\b(?:bộ\s+luật\s+lao\s+động\s*(?:năm\s*)?2019|bllđ\s*2019)\b", re.IGNORECASE), "45/2019/QH14"),
    (re.compile(r"\b(?:bộ\s+luật\s+lao\s+động\s*(?:năm\s*)?2012|bllđ\s*2012)\b", re.IGNORECASE), ("10/2012/QH13", "45/2019/QH14")),
    (re.compile(r"\b(?:bộ\s+luật\s+lao\s+động|bllđ)\b", re.IGNORECASE), "45/2019/QH14"),
    (re.compile(r"\b(?:bộ\s+luật\s+hình\s+sự\s*(?:năm\s*)?2015|blhs\s*2015)\b", re.IGNORECASE), "100/2015/QH13"),
    (re.compile(r"\b(?:bộ\s+luật\s+hình\s+sự|blhs)\b", re.IGNORECASE), "100/2015/QH13"),
    (re.compile(r"\b(?:bộ\s+luật\s+dân\s+sự\s*(?:năm\s*)?2015|blds\s*2015)\b", re.IGNORECASE), "91/2015/QH13"),
    (re.compile(r"\b(?:bộ\s+luật\s+dân\s+sự|blds)\b", re.IGNORECASE), "91/2015/QH13"),
    (re.compile(r"\b(?:luật\s+doanh\s+nghiệp\s*(?:năm\s*)?2020|ldn\s*2020)\b", re.IGNORECASE), "59/2020/QH14"),
    (re.compile(r"\b(?:luật\s+doanh\s+nghiệp\s*(?:năm\s*)?2014|ldn\s*2014)\b", re.IGNORECASE), ("68/2014/QH13", "59/2020/QH14")),
    (re.compile(r"\b(?:luật\s+doanh\s+nghiệp|ldn)\b", re.IGNORECASE), "59/2020/QH14"),
    (re.compile(r"\b(?:luật\s+đầu\s+tư\s*(?:năm\s*)?2025)\b", re.IGNORECASE), "143/2025/QH15"),
    (re.compile(r"\b(?:luật\s+đầu\s+tư\s*(?:năm\s*)?2020)\b", re.IGNORECASE), ("61/2020/QH14", "143/2025/QH15")),
    (re.compile(r"\bluật\s+đầu\s+tư\b(?!\s+công)", re.IGNORECASE), "143/2025/QH15"),
    (re.compile(r"\b(?:luật\s+đất\s+đai\s*(?:năm\s*)?2024)\b", re.IGNORECASE), "31/2024/QH15"),
    (re.compile(r"\b(?:luật\s+đất\s+đai\s*(?:năm\s*)?2013)\b", re.IGNORECASE), "45/2013/QH13"),
    # No generic "luật đất đai" alias: the in-force 31/2024/QH15 has no
    # chunks in the corpus yet, and filtering on it (or on the expired
    # 45/2013/QH13) would return nothing or outdated law. Add it back once
    # the 2024 text is ingested.
    (re.compile(r"\b(?:bộ\s+luật\s+tố\s+tụng\s+hình\s+sự\s*(?:năm\s*)?2015|bltths\s*2015|bộ\s+luật\s+tố\s+tụng\s+hình\s+sự)\b", re.IGNORECASE), "101/2015/QH13"),
    (re.compile(r"\b(?:bộ\s+luật\s+tố\s+tụng\s+dân\s+sự\s*(?:năm\s*)?2015|blttds\s*2015|bộ\s+luật\s+tố\s+tụng\s+dân\s+sự)\b", re.IGNORECASE), "92/2015/QH13"),
    (re.compile(r"\b(?:luật\s+hôn\s+nhân\s+và\s+gia\s+đình\s*(?:năm\s*)?2014|luật\s+hôn\s+nhân\s+và\s+gia\s+đình)\b", re.IGNORECASE), "52/2014/QH13"),
    (re.compile(r"\b(?:luật\s+thương\s+mại\s*(?:năm\s*)?2005|luật\s+thương\s+mại)\b", re.IGNORECASE), "36/2005/QH11"),
    (re.compile(r"\b(?:luật\s+xử\s+lý\s+vi\s+phạm\s+hành\s+chính)\b", re.IGNORECASE), "15/2012/QH13"),
    (re.compile(r"\b(?:luật\s+ban\s+hành\s+văn\s+bản\s+quy\s+phạm\s+pháp\s+luật\s*(?:năm\s*)?2025)\b", re.IGNORECASE), "64/2025/QH15"),
    (re.compile(r"\b(?:luật\s+ban\s+hành\s+văn\s+bản\s+quy\s+phạm\s+pháp\s+luật\s*(?:năm\s*)?2015)\b", re.IGNORECASE), ("80/2015/QH13", "64/2025/QH15")),
    (re.compile(r"\bluật\s+ban\s+hành\s+văn\s+bản\s+quy\s+phạm\s+pháp\s+luật\b", re.IGNORECASE), "64/2025/QH15"),
    (re.compile(r"\b(?:luật\s+nhà\s+ở\s*(?:năm\s*)?2023)\b", re.IGNORECASE), "27/2023/QH15"),
    (re.compile(r"\b(?:luật\s+nhà\s+ở\s*(?:năm\s*)?2014)\b", re.IGNORECASE), ("65/2014/QH13", "27/2023/QH15")),
    (re.compile(r"\bluật\s+nhà\s+ở\b", re.IGNORECASE), "27/2023/QH15"),
    (re.compile(r"\b(?:luật\s+xây\s+dựng\s*(?:năm\s*)?2014|luật\s+xây\s+dựng)\b", re.IGNORECASE), "50/2014/QH13"),
    (re.compile(r"\b(?:luật\s+bảo\s+hiểm\s+xã\s+hội\s*(?:năm\s*)?2024)\b", re.IGNORECASE), "41/2024/QH15"),
    (re.compile(r"\b(?:luật\s+bảo\s+hiểm\s+xã\s+hội\s*(?:năm\s*)?2014)\b", re.IGNORECASE), ("58/2014/QH13", "41/2024/QH15")),
    (re.compile(r"\bluật\s+bảo\s+hiểm\s+xã\s+hội\b", re.IGNORECASE), "41/2024/QH15"),
    (re.compile(r"\b(?:luật\s+căn\s+cước\s*(?:năm\s*)?2023)\b", re.IGNORECASE), "26/2023/QH15"),
    (re.compile(r"\b(?:luật\s+căn\s+cước\s+công\s+dân\s*(?:năm\s*)?2014)\b", re.IGNORECASE), ("59/2014/QH13", "26/2023/QH15")),
    (re.compile(r"\bluật\s+căn\s+cước(?:\s+công\s+dân)?\b", re.IGNORECASE), "26/2023/QH15"),
    (re.compile(r"\b(?:luật\s+an\s+ninh\s+mạng\s*(?:năm\s*)?2018|luật\s+an\s+ninh\s+mạng)\b", re.IGNORECASE), "24/2018/QH14"),
)

_CONVERSATIONAL_PREFIX_RE = re.compile(
    r"^\s*(?:(?:xin|cho)\s+(?:tôi\s+)?hỏi|hãy\s+cho\s+(?:tôi\s+)?biết|tôi\s+muốn\s+(?:hỏi|biết)|"
    r"(?:hãy\s+)?trích(?:\s+dẫn)?(?:\s+cho\s+tôi)?(?:\s+nội\s+dung)?|"
    r"(?:hãy\s+)?nêu(?:\s+cho\s+tôi)?)\b[\s,:]*",
    re.IGNORECASE,
)
_ARTICLE_RE = re.compile(r"\bđiều\s+(\d+[a-z]?)\b", re.IGNORECASE)
_CLAUSE_RE = re.compile(
    r"\bkhoản\s+(\d+[a-z]?)\b"
    r"(?!\s*(?:triệu|tr\b|nghìn|ngàn|tỷ|tỉ|đồng|đ\b|%|phần\s+trăm|usd|vnd))",
    re.IGNORECASE,
)
_POINT_RE = re.compile(r"\bđiểm\s+([a-zđ])\b", re.IGNORECASE)
_RELATIONSHIP_PATTERNS = {
    "AMENDS": re.compile(r"\b(?:sửa đổi|điều chỉnh)\b", re.IGNORECASE),
    "SUPPLEMENTS": re.compile(r"\bbổ sung\b", re.IGNORECASE),
    "REPEALS": re.compile(r"\b(?:bãi bỏ|hủy bỏ)\b", re.IGNORECASE),
    "REPLACES": re.compile(r"\bthay thế\b", re.IGNORECASE),
}


class UnsupportedAsOfDate(ValueError):
    """Raised when a caller asks for law at a date other than the reference date."""


def _vietnam_today() -> date:
    """Reference date: the configured data snapshot date, else today in Vietnam."""
    cutoff = os.getenv("LAWCHAT_LEGAL_CUTOFF_DATE")
    return date.fromisoformat(cutoff) if cutoff else datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).date()


def ensure_supported_as_of(as_of: date | None) -> None:
    """Reject requests for law at any date other than the reference date."""
    reference_date = _vietnam_today()
    if as_of is not None and as_of != reference_date:
        raise UnsupportedAsOfDate(
            "Hệ thống chỉ hỗ trợ pháp luật đang có hiệu lực tại ngày "
            f"{reference_date.strftime('%d/%m/%Y')}; không hỗ trợ tra cứu "
            f"tại ngày {as_of.strftime('%d/%m/%Y')}."
        )


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

        # Dates in the question are case facts ("nghỉ việc ngày 1/3/2024"),
        # not a request to read the law in force at that date. LawChat only
        # answers from law in force at the reference date.
        # Callers validate a user-supplied as_of with ensure_supported_as_of().
        dates = self._extract_dates(normalized)
        effective_as_of = as_of or self._today()

        document_numbers = _extract_document_numbers(normalized)
        articles = _unique_matches(_ARTICLE_RE, normalized)
        # "khoản"/"điểm" are everyday words ("khoản vay", "điểm thi"); only
        # treat them as structural filters next to an explicit "Điều".
        clauses = _unique_matches(_CLAUSE_RE, normalized) if articles else ()
        points = _unique_matches(_POINT_RE, normalized) if articles else ()
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

        from .reranker import SemanticQueryRewriter
        flags = _flags_or_default()
        rewriter = SemanticQueryRewriter()
        if flags.semantic_rewriter_v2:
            # Audit fix W1 (Phase 3): keep paraphrase variants beside the
            # canonical string; dense/sparse search always receive a str.
            semantic_variants = rewriter.rewrite_variants(semantic_query)
            semantic_query = semantic_variants[0]
        else:
            semantic_query = rewriter.rewrite(semantic_query)
            semantic_variants = (semantic_query,)

        return ParsedLegalQuery(
            original_query=normalized,
            semantic_query=semantic_query,
            semantic_variants=semantic_variants,
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
        return tuple(dict.fromkeys(item for item in found if item is not None))


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


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


def _extract_document_numbers(text: str) -> tuple[str, ...]:
    numbers = list(
        value.replace(" ", "")
        for value in _unique_matches(_DOCUMENT_NUMBER_RE, text, upper=True)
    )
    # Aliases are ordered year-specific first; a generic name ("BLLĐ") must
    # not also fire on text already matched as "BLLĐ 2012".
    matched_spans: list[tuple[int, int]] = []
    for pattern, doc_no in _COMMON_LAW_ALIASES:
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(start < other_end and other_start < end for other_start, other_end in matched_spans):
                continue
            matched_spans.append((start, end))
            for number in (doc_no,) if isinstance(doc_no, str) else doc_no:
                if number not in numbers:
                    numbers.append(number)
    return tuple(dict.fromkeys(numbers))


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
    # A recognised law name is already a document filter; leaving half of it
    # ("hình sự" from "bộ luật hình sự") only adds noise to the dense query.
    for pattern, _document_number in _COMMON_LAW_ALIASES:
        value = pattern.sub(" ", value)
    value = _CONVERSATIONAL_PREFIX_RE.sub("", value)
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
    # One leftover word ("trích", "nội dung") carries no meaning to search on;
    # the caller then falls back to the structural lookup query.
    return value if len(value.split()) >= 2 else ""


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


def _flags_or_default():
    """Resolve feature flags without hard-failing when config is unavailable."""
    try:
        try:
            from config import get_feature_flags
        except (ImportError, ValueError):
            from ..config import get_feature_flags  # type: ignore[import-not-found]

        return get_feature_flags()
    except Exception:
        from dataclasses import replace

        try:
            from config import load_feature_flags
        except (ImportError, ValueError):
            from ..config import load_feature_flags

        return replace(load_feature_flags(), semantic_rewriter_v2=False)
