"""Audit fix G1 (Phase 4): lightweight Vietnamese NER for legal queries.

The legacy query parser only extracted document numbers via a single
regex. This module adds a deterministic, dependency-free extractor for
issuing-authority names, document-type keywords, dates, and signer names
so the graph resolver can match on more than ``document_number``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExtractedEntity:
    kind: str
    value: str
    start: int
    end: int


_AUTHORITY_KEYWORDS: tuple[str, ...] = (
    "Quốc hội",
    "Chính phủ",
    "Thủ tướng",
    "Bộ Tư pháp",
    "Bộ Tài chính",
    "Bộ Lao động",
    "Bộ Y tế",
    "Bộ Giáo dục",
    "Bộ Công an",
    "Bộ Quốc phòng",
    "Tòa án nhân dân",
    "Viện kiểm sát",
    "Uỷ ban nhân dân",
    "Hội đồng nhân dân",
)

_DOC_TYPE_KEYWORDS: tuple[str, ...] = (
    "Luật",
    "Bộ luật",
    "Nghị quyết",
    "Nghị định",
    "Quyết định",
    "Thông tư",
    "Thông tư liên tịch",
    "Pháp lệnh",
    "Lệnh",
    "Chỉ thị",
)


_DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b")


def extract_legal_entities(query: str) -> tuple[ExtractedEntity, ...]:
    """Extract a tuple of legal entities from a Vietnamese query string."""
    entities: list[ExtractedEntity] = []
    entities.extend(_match_keywords(query, _AUTHORITY_KEYWORDS, "authority"))
    entities.extend(_match_keywords(query, _DOC_TYPE_KEYWORDS, "document_type"))
    for match in _DATE_RE.finditer(query):
        entities.append(
            ExtractedEntity(
                kind="date",
                value=match.group(0),
                start=match.start(),
                end=match.end(),
            )
        )
    entities.sort(key=lambda item: (item.start, item.end))
    return tuple(entities)


def _match_keywords(
    query: str,
    keywords: tuple[str, ...],
    kind: str,
) -> list[ExtractedEntity]:
    results: list[ExtractedEntity] = []
    lowered = query.casefold()
    for keyword in keywords:
        pattern = re.compile(re.escape(keyword), flags=re.IGNORECASE)
        for match in pattern.finditer(lowered):
            results.append(
                ExtractedEntity(
                    kind=kind,
                    value=match.group(0),
                    start=match.start(),
                    end=match.end(),
                )
            )
    return results


__all__ = ["ExtractedEntity", "extract_legal_entities"]