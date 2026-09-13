from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from sqlalchemy.orm import Session, sessionmaker

from lawchat.database import LegalMetadataFilter, MetadataQueries

from .models import HydratedLegalChunk


class LegalContextBuilder:
    def __init__(self, *, max_characters_per_result: int = 12_000) -> None:
        if max_characters_per_result <= 0:
            raise ValueError("max_characters_per_result must be > 0")
        self.max_characters_per_result = max_characters_per_result

    def build(self, chunk: HydratedLegalChunk) -> str:
        header_parts = [chunk.title]
        if chunk.document_number:
            header_parts.append(f"Số hiệu: {chunk.document_number}")
        location = _legal_location(chunk)
        if location:
            header_parts.append(location)
        header_parts.append(
            f"Hiệu lực tại thời điểm truy vấn: {chunk.status} "
            f"(từ {chunk.valid_from.isoformat()}"
            f"{f' đến trước {chunk.valid_to.isoformat()}' if chunk.valid_to else ''})"
        )

        leaf_section = f"Đoạn liên quan:\n{chunk.text.strip()}"
        base = "\n".join(header_parts + ["", leaf_section])
        if len(base) >= self.max_characters_per_result:
            return base[: self.max_characters_per_result].rstrip() + "…"

        parent = (chunk.parent_text or "").strip()
        if not parent or parent == chunk.text.strip():
            return base

        parent_prefix = "Ngữ cảnh cấp trên:\n"
        separator = "\n\n"
        remaining = (
            self.max_characters_per_result
            - len(base)
            - len(parent_prefix)
            - len(separator)
        )
        if remaining <= 0:
            return base
        parent_excerpt = parent[:remaining].rstrip()
        if len(parent_excerpt) < len(parent):
            parent_excerpt += "…"
        return "\n".join(header_parts + ["", parent_prefix + parent_excerpt, "", leaf_section])


def _legal_location(chunk: HydratedLegalChunk) -> str:
    return " ".join(
        value
        for value in (
            f"Điều {chunk.article}" if chunk.article else None,
            f"Khoản {chunk.clause}" if chunk.clause else None,
            f"Điểm {chunk.point}" if chunk.point else None,
        )
        if value
    )


class ChunkHydrator(Protocol):
    def hydrate(
        self,
        point_ids: Sequence[str],
        filters: LegalMetadataFilter,
        *,
        collection: str | None = None,
    ) -> list[HydratedLegalChunk]: ...


class PostgresChunkHydrator:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.session_factory = session_factory

    def hydrate(
        self,
        point_ids: Sequence[str],
        filters: LegalMetadataFilter,
        *,
        collection: str | None = None,
    ) -> list[HydratedLegalChunk]:
        if not point_ids:
            return []
        statement = MetadataQueries.hydrate_qdrant_points(
            point_ids,
            filters,
            collection=collection,
        )
        with self.session_factory() as session:
            rows = session.execute(statement).mappings().all()

        by_point_id = {
            row["point_id"]: _to_hydrated_chunk(row)
            for row in rows
            if row["point_id"] is not None
        }
        return [
            by_point_id[point_id]
            for point_id in point_ids
            if point_id in by_point_id
        ]


def _to_hydrated_chunk(row) -> HydratedLegalChunk:
    metadata = dict(row["chunk_metadata"] or {})
    return HydratedLegalChunk(
        point_id=row["point_id"],
        chunk_id=row["chunk_id"],
        chunk_type=row["chunk_type"],
        text=row["text"],
        retrieval_text=row["retrieval_text"],
        chunk_metadata=metadata,
        parent_chunk_id=row["parent_chunk_id"],
        parent_text=row["parent_text"],
        document_id=row["document_id"],
        title=row["title"],
        document_number=row["document_number"],
        document_type=row["document_type"],
        authority=row["authority"],
        legal_field=row["legal_field"],
        source_url=row["source_url"],
        article=_metadata_value(metadata, "article") or row["article_number"],
        clause=_metadata_value(metadata, "clause"),
        point=_metadata_value(metadata, "point"),
        status=row["status"],
        valid_from=row["valid_from"],
        valid_to=row["valid_to"],
        status_scope=row["status_scope"],
        version_id=str(row["version_id"]) if row["version_id"] else None,
        version_source_url=row["version_source_url"],
        version_source_revision=row["version_source_revision"],
        content_valid_from=row["content_valid_from"],
        content_valid_to=row["content_valid_to"],
    )


def _metadata_value(metadata: dict, key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None
