from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from typing import Any

from qdrant_client import QdrantClient, models

from indexing.embeddings import EmbeddingBackend


@dataclass(frozen=True, slots=True)
class DenseSearchFilter:
    doc_ids: tuple[str, ...] = ()
    document_numbers: tuple[str, ...] = ()
    document_types: tuple[str, ...] = ()
    authorities: tuple[str, ...] = ()
    statuses: tuple[str, ...] = ()
    legal_fields: tuple[str, ...] = ()
    articles: tuple[str, ...] = ()
    clauses: tuple[str, ...] = ()
    points: tuple[str, ...] = ()
    content_as_of: date | None = None
    exclude_document_types: tuple[str, ...] = ("Bản dịch văn bản",)


@dataclass(frozen=True, slots=True)
class DenseSearchResult:
    point_id: str
    score: float
    payload: dict[str, Any]


class DenseSearcher:
    def __init__(
        self,
        client: QdrantClient,
        embedder: EmbeddingBackend,
        *,
        collection: str,
    ) -> None:
        self.client = client
        self.embedder = embedder
        self.collection = collection

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        score_threshold: float | None = None,
        filters: DenseSearchFilter | None = None,
    ) -> list[DenseSearchResult]:
        normalized_query = " ".join(query.split()).strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        if limit <= 0:
            raise ValueError("limit must be > 0")
        vector = self._encode_query(normalized_query)
        response = self.client.query_points(
            collection_name=self.collection,
            query=list(vector),
            query_filter=_build_filter(filters),
            limit=limit,
            with_payload=True,
            with_vectors=False,
            score_threshold=score_threshold,
        )
        return [
            DenseSearchResult(
                point_id=str(point.id),
                score=float(point.score),
                payload=dict(point.payload or {}),
            )
            for point in response.points
        ]

    @lru_cache(maxsize=256)
    def _encode_query(self, normalized_query: str) -> tuple[float, ...]:
        vector = self.embedder.encode_queries([normalized_query])[0]
        return tuple(float(value) for value in vector)


def _build_filter(
    filters: DenseSearchFilter | None,
) -> models.Filter | None:
    if filters is None:
        return None
    must: list = []
    for key, values in (
        ("doc_id", filters.doc_ids),
        ("document_number", filters.document_numbers),
        ("document_type", filters.document_types),
        ("authority", filters.authorities),
        ("status", filters.statuses),
        ("legal_field", filters.legal_fields),
        ("article", filters.articles),
        ("clause", filters.clauses),
        ("point", filters.points),
    ):
        normalized = [value.strip() for value in values if value.strip()]
        if normalized:
            must.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchAny(any=normalized),
                )
            )
    if filters.content_as_of is not None:
        must.append(
            models.FieldCondition(
                key="content_valid_from",
                range=models.DatetimeRange(lte=filters.content_as_of),
            )
        )
        must.append(
            models.Filter(
                should=[
                    models.FieldCondition(
                        key="content_valid_to",
                        range=models.DatetimeRange(gt=filters.content_as_of),
                    ),
                    models.IsEmptyCondition(
                        is_empty=models.PayloadField(key="content_valid_to")
                    ),
                ]
            )
        )
    must_not = [
        models.FieldCondition(
            key="document_type",
            match=models.MatchAny(any=list(filters.exclude_document_types)),
        )
    ] if filters.exclude_document_types else None
    return models.Filter(must=must or None, must_not=must_not) if must or must_not else None
