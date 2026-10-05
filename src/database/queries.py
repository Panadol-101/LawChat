from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Sequence

from sqlalchemy import Select, Subquery, and_, case, func, or_, select
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
    temporal_intent: str = "current_law"
    include_translations: bool = False
    strict_partial_refusal: bool = False

    def __post_init__(self) -> None:
        if not self.statuses:
            raise ValueError("statuses must not be empty")
        unknown = set(self.statuses) - {item.value for item in LegalStatus}
        if unknown:
            raise ValueError(f"Unknown legal statuses: {sorted(unknown)}")
        if self.content_scope not in {"current", "historical"}:
            raise ValueError("content_scope must be current or historical")
        if self.temporal_intent not in {
            "current_law", "status_lookup", "historical"
        }:
            raise ValueError(
                "temporal_intent must be current_law, status_lookup or historical"
            )


def _target_points_subquery(
    point_ids: Sequence[str],
    collection: str | None = None,
) -> Subquery:
    target_vref = select(
        ChunkVectorRef.chunk_id.label("chunk_id"),
        ChunkVectorRef.point_id.label("point_id"),
    ).where(ChunkVectorRef.point_id.in_(tuple(point_ids)))
    if collection is not None:
        target_vref = target_vref.where(ChunkVectorRef.collection == collection)

    target_legacy = select(
        Chunk.id.label("chunk_id"),
        Chunk.qdrant_point_id.label("point_id"),
    ).where(Chunk.qdrant_point_id.in_(tuple(point_ids)))
    if collection is not None:
        target_legacy = target_legacy.where(Chunk.qdrant_collection == collection)

    return target_vref.union(target_legacy).subquery("target_points")


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
        target_points = _target_points_subquery(point_ids, collection)
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
        # Audit fix E10 (Phase 2): when a chunk carries a provision row but
        # the status is still UNKNOWN, the legacy code silently fell back
        # to the document-level status. That hides ambiguity for downstream
        # callers (the verifier and the prompt). With ``profile_unknown_exclude``
        # we treat the row as "unresolved" instead, which lets the E1+E2
        # fail-closed path trigger for any document where provision-level
        # data exists but is not yet normalised.
        resolved_status = case(
            (
                has_resolved_provision_status,
                provision_status.status,
            ),
            else_=EffectiveStatus.status,
        )
        query = (
            select(
                target_points.c.point_id.label("point_id"),
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
            .select_from(target_points)
            .join(Chunk, Chunk.id == target_points.c.chunk_id)
            .join(DocumentVersion, DocumentVersion.id == Chunk.version_id)
            .join(Document, Document.id == Chunk.document_id)
            .join(EffectiveStatus, EffectiveStatus.document_id == Document.id)
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
                resolved_status.in_(filters.statuses),
                EffectiveStatus.valid_period.op("@>")(filters.as_of),
            )
        )
        if filters.temporal_intent == "current_law":
            current_predicates = [
                or_(
                    provision_status.status.is_(None),
                    ~provision_status.status.in_([
                        LegalStatus.REPEALED.value,
                        LegalStatus.EXPIRED.value,
                    ]),
                ),
            ]
            if filters.strict_partial_refusal:
                current_predicates.append(
                    or_(
                        EffectiveStatus.status != LegalStatus.PARTIALLY_EFFECTIVE.value,
                        and_(
                            provision_status.id.is_not(None),
                            provision_status.status != LegalStatus.UNKNOWN.value,
                        ),
                    )
                )
            query = query.where(*current_predicates)
        return _apply_document_filters(query, filters)

    @staticmethod
    def unresolved_provision_point_ids(
        point_ids: Sequence[str],
        filters: LegalMetadataFilter,
        *,
        collection: str | None = None,
    ) -> Select:
        """Identify current-law candidates rejected for unresolved partial status."""
        target_points = _target_points_subquery(point_ids, collection)
        provision_status = aliased(
            ProvisionEffectiveStatus,
            name="diagnostic_provision_status",
        )
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
        query = (
            select(target_points.c.point_id.label("point_id"))
            .select_from(target_points)
            .join(Chunk, Chunk.id == target_points.c.chunk_id)
            .join(DocumentVersion, DocumentVersion.id == Chunk.version_id)
            .join(Document, Document.id == Chunk.document_id)
            .join(EffectiveStatus, EffectiveStatus.document_id == Document.id)
            .outerjoin(Article, Article.id == Chunk.article_id)
            .outerjoin(
                provision_status,
                and_(
                    provision_status.document_id == Document.id,
                    provision_status.provision_key == provision_key,
                    provision_status.valid_period.op("@>")(filters.as_of),
                ),
            )
            .where(
                filters.temporal_intent == "current_law",
                _content_version_predicate(DocumentVersion, filters),
                Chunk.is_indexable.is_(True),
                EffectiveStatus.status
                == LegalStatus.PARTIALLY_EFFECTIVE.value,
                EffectiveStatus.valid_period.op("@>")(filters.as_of),
                or_(
                    provision_status.id.is_(None),
                    provision_status.status == LegalStatus.UNKNOWN.value,
                ),
            )
        )
        return _apply_document_filters(query, filters)

    @staticmethod
    def rejected_candidate_reasons(
        point_ids: Sequence[str],
        filters: LegalMetadataFilter,
        *,
        collection: str | None = None,
    ) -> Select:
        """Return ``(point_id, reason)`` rows for candidates dropped by hydration.

        The reason label follows the same constants used in
        ``retrieval.hydration`` (``REPEALED``, ``EXPIRED``,
        ``NOT_YET_EFFECTIVE``, ``UNRESOLVED_PROVISION_STATUS``,
        ``MISSING_SOURCE``).
        """
        target_points = _target_points_subquery(point_ids, collection)
        query = (
            select(
                target_points.c.point_id.label("point_id"),
                # Audit fix (Phase 1 hotfix 1.6): the row is read back by
                # :func:`src.retrieval.hydration.hydration_rejection_reasons`
                # via ``row.get("reason")``. The legacy label was
                # ``doc_status`` which silently turned every rejection
                # reason into ``"UNKNOWN"`` and broke retrieval diagnostics.
                EffectiveStatus.status.label("reason"),
            )
            .select_from(target_points)
            .join(Chunk, Chunk.id == target_points.c.chunk_id)
            .join(DocumentVersion, DocumentVersion.id == Chunk.version_id)
            .join(Document, Document.id == Chunk.document_id)
            .join(EffectiveStatus, EffectiveStatus.document_id == Document.id)
            .where(
                _content_version_predicate(DocumentVersion, filters),
                Chunk.is_indexable.is_(True),
                or_(
                    EffectiveStatus.status == LegalStatus.REPEALED.value,
                    EffectiveStatus.status == LegalStatus.EXPIRED.value,
                    EffectiveStatus.status == LegalStatus.NOT_YET_EFFECTIVE.value,
                    EffectiveStatus.source_url.is_(None),
                ),
            )
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
