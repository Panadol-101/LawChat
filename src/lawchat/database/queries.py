from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Sequence

from sqlalchemy import Select, and_, case, func, select
from sqlalchemy.orm import aliased

from .models import (
    Article,
    Chunk,
    ChunkVectorRef,
    Document,
    DocumentVersion,
    EffectiveStatus,
    LegalStatus,
    ProvisionEffectiveStatus,
)


DEFAULT_ACTIVE_STATUSES = (
    LegalStatus.EFFECTIVE.value,
    LegalStatus.PARTIALLY_EFFECTIVE.value,
)
TRANSLATION_DOCUMENT_TYPES = ("Bản dịch văn bản",)


@dataclass(frozen=True, slots=True)
class LegalMetadataFilter:
    """Deterministic pre-filter applied before vector similarity search."""

    as_of: date
    statuses: tuple[str, ...] = DEFAULT_ACTIVE_STATUSES
    document_numbers: tuple[str, ...] = ()
    document_ids: tuple[str, ...] = ()
    document_types: tuple[str, ...] = ()
    authorities: tuple[str, ...] = ()
    legal_fields: tuple[str, ...] = ()
    content_scope: str = "current"
    include_translations: bool = False

    def __post_init__(self) -> None:
        if not self.statuses:
            raise ValueError("statuses must not be empty")
        unknown = set(self.statuses) - {item.value for item in LegalStatus}
        if unknown:
            raise ValueError(f"Unknown legal statuses: {sorted(unknown)}")
        if self.content_scope not in {"current", "historical"}:
            raise ValueError("content_scope must be current or historical")


class MetadataQueries:
    """SQL builders kept separate so API/worker layers share one legal rule."""

    @staticmethod
    def effective_documents(
        filters: LegalMetadataFilter,
    ) -> Select[tuple[Document]]:
        query = (
            select(Document)
            .join(EffectiveStatus, EffectiveStatus.document_id == Document.id)
            .where(
                EffectiveStatus.status.in_(filters.statuses),
                EffectiveStatus.valid_period.op("@>")(filters.as_of),
            )
        )
        return _apply_document_filters(query, filters)

    @staticmethod
    def effective_document_external_ids(
        filters: LegalMetadataFilter,
    ) -> Select[tuple[str]]:
        """Return ``doc_id`` values suitable for a Qdrant payload filter."""
        query = (
            select(Document.external_id)
            .join(EffectiveStatus, EffectiveStatus.document_id == Document.id)
            .where(
                EffectiveStatus.status.in_(filters.statuses),
                EffectiveStatus.valid_period.op("@>")(filters.as_of),
            )
            .order_by(Document.external_id)
        )
        return _apply_document_filters(query, filters)

    @staticmethod
    def qdrant_point_ids(
        filters: LegalMetadataFilter,
        *,
        collection: str | None = None,
    ) -> Select[tuple[str]]:
        """Return allowed Qdrant point IDs for current document versions."""
        vector_ref = aliased(ChunkVectorRef, name="vector_ref")
        point_id = func.coalesce(vector_ref.point_id, Chunk.qdrant_point_id)
        query = (
            select(point_id)
            .select_from(Chunk)
            .join(DocumentVersion, DocumentVersion.id == Chunk.version_id)
            .join(Document, Document.id == Chunk.document_id)
            .join(EffectiveStatus, EffectiveStatus.document_id == Document.id)
            .outerjoin(
                vector_ref,
                and_(
                    vector_ref.chunk_id == Chunk.id,
                    vector_ref.collection == collection,
                ),
            )
            .where(
                _content_version_predicate(DocumentVersion, filters),
                Chunk.is_indexable.is_(True),
                (Chunk.qdrant_point_id.is_not(None))
                | (vector_ref.point_id.is_not(None)),
                EffectiveStatus.status.in_(filters.statuses),
                EffectiveStatus.valid_period.op("@>")(filters.as_of),
            )
            .order_by(point_id)
        )
        if collection is not None:
            query = query.where(
                (vector_ref.collection == collection)
                | (Chunk.qdrant_collection == collection)
            )
        return _apply_document_filters(query, filters)

    @staticmethod
    def hydrate_qdrant_points(
        point_ids: Sequence[str],
        filters: LegalMetadataFilter,
        *,
        collection: str | None = None,
    ) -> Select:
        """Load ranked Qdrant candidates and enforce legal validity in SQL.

        Ranking is restored by the caller because SQL ``IN`` does not preserve
        the vector search order. Parent text is loaded in the same statement to
        avoid an N+1 query when building legal context.
        """
        parent = aliased(Chunk, name="parent_chunk")
        provision_status = aliased(
            ProvisionEffectiveStatus,
            name="provision_status",
        )
        vector_ref = aliased(ChunkVectorRef, name="vector_ref")
        point_id = func.coalesce(vector_ref.point_id, Chunk.qdrant_point_id)
        provision_key = func.concat(
            "article:",
            func.lower(
                func.btrim(
                    func.coalesce(
                        Chunk.metadata_json.op("->>")("article"),
                        Article.article_number,
                        "",
                    )
                )
            ),
            "/clause:",
            func.lower(
                func.btrim(
                    func.coalesce(Chunk.metadata_json.op("->>")("clause"), "")
                )
            ),
            "/point:",
            func.lower(
                func.btrim(
                    func.coalesce(Chunk.metadata_json.op("->>")("point"), "")
                )
            ),
        )
        has_resolved_provision_status = and_(
            provision_status.id.is_not(None),
            provision_status.status != LegalStatus.UNKNOWN.value,
        )
        resolved_status = case(
            (
                has_resolved_provision_status,
                provision_status.status,
            ),
            else_=EffectiveStatus.status,
        )
        query = (
            select(
                point_id.label("point_id"),
                Chunk.external_id.label("chunk_id"),
                Chunk.chunk_type,
                Chunk.text_content.label("text"),
                Chunk.retrieval_text,
                Chunk.metadata_json.label("chunk_metadata"),
                parent.external_id.label("parent_chunk_id"),
                parent.text_content.label("parent_text"),
                Document.external_id.label("document_id"),
                Document.title,
                Document.document_number,
                Document.document_type,
                Document.authority,
                Document.legal_field,
                Document.source_url,
                DocumentVersion.id.label("version_id"),
                DocumentVersion.source_url.label("version_source_url"),
                func.coalesce(
                    DocumentVersion.source_revision,
                    DocumentVersion.metadata_json.op("->>")("dataset_revision"),
                ).label("version_source_revision"),
                DocumentVersion.content_valid_from,
                DocumentVersion.content_valid_to,
                Article.article_number.label("article_number"),
                resolved_status.label("status"),
                case(
                    (has_resolved_provision_status, provision_status.valid_from),
                    else_=EffectiveStatus.valid_from,
                ).label("valid_from"),
                case(
                    (has_resolved_provision_status, provision_status.valid_to),
                    else_=EffectiveStatus.valid_to,
                ).label("valid_to"),
                case(
                    (has_resolved_provision_status, "provision"),
                    else_="document",
                ).label("status_scope"),
            )
            .select_from(Chunk)
            .join(DocumentVersion, DocumentVersion.id == Chunk.version_id)
            .join(Document, Document.id == Chunk.document_id)
            .join(EffectiveStatus, EffectiveStatus.document_id == Document.id)
            .outerjoin(
                vector_ref,
                and_(
                    vector_ref.chunk_id == Chunk.id,
                    vector_ref.collection == collection,
                ),
            )
            .outerjoin(Article, Article.id == Chunk.article_id)
            .outerjoin(
                provision_status,
                and_(
                    provision_status.document_id == Document.id,
                    provision_status.provision_key == provision_key,
                    provision_status.valid_period.op("@>")(filters.as_of),
                ),
            )
            .outerjoin(parent, parent.id == Chunk.parent_chunk_id)
            .where(
                _content_version_predicate(DocumentVersion, filters),
                Chunk.is_indexable.is_(True),
                point_id.in_(tuple(point_ids)),
                resolved_status.in_(filters.statuses),
                EffectiveStatus.valid_period.op("@>")(filters.as_of),
            )
        )
        if collection is not None:
            query = query.where(
                (vector_ref.collection == collection)
                | (Chunk.qdrant_collection == collection)
            )
        return _apply_document_filters(query, filters)


def _apply_document_filters(
    query: Select,
    filters: LegalMetadataFilter,
) -> Select:
    clauses: list = []
    if not filters.include_translations:
        clauses.append(
            (Document.document_type.is_(None))
            | (Document.document_type.not_in(TRANSLATION_DOCUMENT_TYPES))
        )
    _append_in_filter(clauses, Document.external_id, filters.document_ids)
    _append_in_filter(clauses, Document.document_number, filters.document_numbers)
    _append_in_filter(clauses, Document.document_type, filters.document_types)
    _append_in_filter(clauses, Document.authority, filters.authorities)
    _append_in_filter(clauses, Document.legal_field, filters.legal_fields)
    return query.where(and_(*clauses)) if clauses else query


def _append_in_filter(clauses: list, column, values: Iterable[str]) -> None:
    normalized = tuple(value.strip() for value in values if value.strip())
    if normalized:
        clauses.append(column.in_(normalized))


def _content_version_predicate(version, filters: LegalMetadataFilter):
    if filters.content_scope == "historical":
        return version.content_valid_period.op("@>")(filters.as_of)
    return version.is_current.is_(True)
