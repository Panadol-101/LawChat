from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import tantivy

from indexing.bm25_index import (
    BM25_SCHEMA_VERSION,
    BM25Manifest,
    BM25Settings,
    normalize_keyword,
    normalize_legal_identifier,
    _segment_legal_text,
)

from .fusion import RetrievalCandidate


DEFAULT_SEARCH_FIELDS = (
    "document_number",
    "title",
    "retrieval_text",
)
DEFAULT_FIELD_BOOSTS = {
    "document_number": 10.0,
    "title": 3.0,
    "retrieval_text": 1.0,
}

_LEXICAL_TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+(?:[/.-][0-9A-Za-zÀ-ỹĐđ]+)*")
_YES_NO_QUESTION_RE = re.compile(
    r"\bcó\s+được\b(?P<body>.+?)\bkhông\s*[?.!]*$",
    flags=re.IGNORECASE,
)
_LEGAL_PHRASE_EXPANSIONS = (
    (
        re.compile(
            r"\bđơn\s+phương\s+chấm\s+dứt\s+hợp\s+đồng(?:\s+lao\s+động)?\b",
            flags=re.IGNORECASE,
        ),
        "đơn phương chấm dứt hợp đồng lao động",
        20,
    ),
)


class SparseIndexUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SparseSearchFilter:
    doc_ids: tuple[str, ...] = ()
    document_numbers: tuple[str, ...] = ()
    document_types: tuple[str, ...] = ()
    authorities: tuple[str, ...] = ()
    legal_fields: tuple[str, ...] = ()
    article_hints: tuple[str, ...] = ()
    clause_hints: tuple[str, ...] = ()
    point_hints: tuple[str, ...] = ()
    require_structure: bool = False


class TantivySparseSearcher:
    def __init__(
        self,
        index: tantivy.Index,
        *,
        index_name: str,
    ) -> None:
        self.index = index
        self.index_name = index_name

    @classmethod
    def from_settings(
        cls,
        settings: BM25Settings | None = None,
    ) -> "TantivySparseSearcher":
        settings = settings or BM25Settings.from_env()
        alias = _read_json(settings.alias_path, description="BM25 alias")
        if alias.get("schema_version") != BM25_SCHEMA_VERSION:
            raise SparseIndexUnavailable("BM25 alias uses an unsupported schema")
        index_name = alias.get("index_name")
        if not index_name:
            raise SparseIndexUnavailable("BM25 alias does not contain index_name")
        index_path = settings.root / "indexes" / index_name
        manifest_value = _read_json(
            index_path / "lawchat_manifest.json",
            description="BM25 manifest",
        )
        manifest = BM25Manifest(**manifest_value)
        if not manifest.complete:
            raise SparseIndexUnavailable(f"BM25 index {index_name} is incomplete")
        if manifest.source_collection != settings.source_collection:
            raise SparseIndexUnavailable(
                "BM25 and Qdrant collections belong to different releases"
            )
        return cls(tantivy.Index.open(str(index_path)), index_name=index_name)

    def search(
        self,
        query: str,
        *,
        limit: int = 100,
        filters: SparseSearchFilter | None = None,
    ) -> list[RetrievalCandidate]:
        normalized_query = " ".join(query.split()).strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        if limit <= 0:
            raise ValueError("limit must be > 0")

        filters = filters or SparseSearchFilter()
        # Audit fix (Phase 1 hotfix 1.4): mirror the Vietnamese segmentation
        # used at indexing time so multi-word phrases match correctly. The
        # legacy code passed the raw Vietnamese string to Tantivy, which
        # only tokenised on whitespace.
        segmented_query = _segment_legal_text(normalized_query)
        query_text = _build_query(segmented_query, filters)
        parsed_query, _errors = self.index.parse_query_lenient(
            query_text,
            default_field_names=list(DEFAULT_SEARCH_FIELDS),
            field_boosts=DEFAULT_FIELD_BOOSTS,
        )
        searcher = self.index.searcher()
        result = searcher.search(parsed_query, limit=limit, count=False)
        candidates: list[RetrievalCandidate] = []
        for rank, (score, address) in enumerate(result.hits, start=1):
            stored = searcher.doc(address).to_dict()
            point_id = _stored_value(stored, "point_id")
            chunk_id = _stored_value(stored, "chunk_id")
            candidates.append(
                RetrievalCandidate(
                    point_id=point_id,
                    chunk_id=chunk_id,
                    source="sparse",
                    raw_score=float(score),
                    rank=rank,
                    payload={
                        key: value[0]
                        for key, value in stored.items()
                        if value and key not in {"point_id", "chunk_id"}
                    },
                )
            )
        return candidates


def _build_query(query: str, filters: SparseSearchFilter) -> str:
    boosted_parts = _semantic_phrase_boosts(query)
    boosted_parts.append(f"({_escape_user_query(query)})")
    boosted_parts.extend(
        _boosted_terms(
            "document_number_normalized",
            (normalize_legal_identifier(item) for item in filters.document_numbers),
            boost=10,
        )
    )
    if not filters.require_structure:
        boosted_parts.extend(_boosted_terms("article", filters.article_hints, boost=5))
        boosted_parts.extend(_boosted_terms("clause", filters.clause_hints, boost=5))
        boosted_parts.extend(_boosted_terms("point", filters.point_hints, boost=5))
    value = " OR ".join(boosted_parts)

    required = [
        _required_group("doc_id", filters.doc_ids),
        _required_group(
            "document_number_normalized",
            tuple(normalize_legal_identifier(item) for item in filters.document_numbers),
        ),
        _required_group("document_type", filters.document_types),
        _required_group("authority", filters.authorities),
        _required_group("legal_field", filters.legal_fields),
    ]
    if filters.require_structure:
        required.extend(
            (
                _required_group("article", filters.article_hints),
                _required_group("clause", filters.clause_hints),
                _required_group("point", filters.point_hints),
            )
        )
    return " AND ".join([f"({value})", *(item for item in required if item)])


def _semantic_phrase_boosts(query: str) -> list[str]:
    """Boost legal content phrases without sacrificing raw-query recall.

    Vietnamese yes/no questions often wrap the useful terms in
    ``<subject> có được <legal action> không``.  Raw OR parsing makes common
    words dominate long documents.  Keep subject/action phrases together and
    add an AND core while retaining the original query as a fallback.
    """
    normalized = " ".join(query.split()).strip()
    match = _YES_NO_QUESTION_RE.search(normalized)
    phrases: list[str] = []
    if match:
        prefix = normalized[: match.start()].strip(" ,;:?.!")
        body = match.group("body").strip(" ,;:?.!")
        phrases.extend(item for item in (prefix, body) if item)
    else:
        value = normalized.strip(" ,;:?.!")
        tokens = _LEXICAL_TOKEN_RE.findall(value)
        if 2 <= len(tokens) <= 12:
            phrases.append(" ".join(tokens))

    output: list[str] = [
        f'"{_escape_phrase(expansion)}"^{boost}'
        for pattern, expansion, boost in _LEGAL_PHRASE_EXPANSIONS
        if pattern.search(normalized)
    ]
    core_terms: list[str] = []
    for phrase in phrases:
        tokens = _LEXICAL_TOKEN_RE.findall(phrase)
        if len(tokens) < 2:
            continue
        rendered = " ".join(tokens)
        boost = 10 if len(tokens) >= 3 else 4
        output.append(f'"{_escape_phrase(rendered)}"^{boost}')
        core_terms.extend(tokens)
    unique_terms = tuple(dict.fromkeys(item.casefold() for item in core_terms))
    if 2 <= len(unique_terms) <= 10:
        joined = " AND ".join(
            f'"{_escape_phrase(term)}"' for term in unique_terms
        )
        output.append(f"({joined})^2")
    return output


def _boosted_terms(field: str, values, *, boost: int) -> list[str]:
    return [
        f'{field}:"{_escape_phrase(normalize_keyword(value))}"^{boost}'
        for value in values
        if value and value.strip()
    ]


def _required_group(field: str, values: tuple[str, ...]) -> str:
    normalized = tuple(
        normalize_keyword(value)
        for value in values
        if value and value.strip()
    )
    if not normalized:
        return ""
    terms = " OR ".join(
        f'{field}:"{_escape_phrase(value)}"' for value in normalized
    )
    return f"({terms})"


def _escape_user_query(value: str) -> str:
    # Lenient parsing handles legal punctuation. Prevent user input from
    # injecting field-scoped clauses into the generated query.
    return value.replace("\\", " ").replace(":", " ")


def _escape_phrase(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _stored_value(document: dict, field: str) -> str:
    values = document.get(field) or []
    if not values:
        raise SparseIndexUnavailable(f"BM25 result is missing stored field {field}")
    return str(values[0])


def _read_json(path: Path, *, description: str) -> dict:
    if not path.exists():
        raise SparseIndexUnavailable(f"{description} does not exist: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SparseIndexUnavailable(f"cannot read {description}: {path}") from exc
