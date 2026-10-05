from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Iterator, Sequence

import numpy as np
from qdrant_client import QdrantClient, models
from sqlalchemy import Engine, text

from .embeddings import EmbeddingBackend


POINT_NAMESPACE = uuid.UUID("fe51c6ce-c1aa-5d5d-9c18-caa1e617cb91")


@dataclass(frozen=True, slots=True)
class QdrantSettings:
    url: str = "http://localhost:6333"
    api_key: str | None = None
    collection: str = "legal_chunks_bge_m3_1024_v1"
    alias: str = "legal_chunks_current"
    timeout_seconds: int = 60
    prefer_grpc: bool = False

    @classmethod
    def from_env(cls) -> "QdrantSettings":
        return cls(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY") or None,
            collection=os.getenv(
                "QDRANT_COLLECTION", "legal_chunks_bge_m3_1024_v1"
            ),
            alias=os.getenv("QDRANT_ALIAS", "legal_chunks_current"),
            timeout_seconds=int(os.getenv("QDRANT_TIMEOUT", "60")),
            prefer_grpc=os.getenv("QDRANT_PREFER_GRPC", "false").casefold()
            in {"1", "true", "yes", "on"},
        )

    def create_client(self) -> QdrantClient:
        return QdrantClient(
            url=self.url,
            api_key=self.api_key,
            timeout=self.timeout_seconds,
            prefer_grpc=self.prefer_grpc,
        )


@dataclass(frozen=True, slots=True)
class IndexableChunk:
    database_id: uuid.UUID
    version_id: uuid.UUID
    external_id: str
    retrieval_text: str
    payload: dict[str, Any]


@dataclass(slots=True)
class DenseIndexReport:
    collection: str
    model_name: str
    indexed_points: int = 0
    batches: int = 0
    last_chunk_id: str | None = None


def stable_point_id(version_id: uuid.UUID | str, chunk_id: str) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, f"{version_id}:{chunk_id}"))


class PostgresChunkSource:
    """Keyset-paginated source of current or versioned historical chunks."""

    def __init__(
        self,
        engine: Engine,
        *,
        fetch_size: int = 256,
        version_scope: str = "current",
    ) -> None:
        if fetch_size <= 0:
            raise ValueError("fetch_size must be > 0")
        self.engine = engine
        self.fetch_size = fetch_size
        if version_scope not in {"current", "historical"}:
            raise ValueError("version_scope must be current or historical")
        self.version_scope = version_scope

    def iter_batches(
        self,
        *,
        collection: str,
        force: bool = False,
        limit: int | None = None,
    ) -> Iterator[list[IndexableChunk]]:
        emitted = 0
        cursor: uuid.UUID | None = None
        with self.engine.connect() as connection:
            while limit is None or emitted < limit:
                batch_limit = min(
                    self.fetch_size,
                    limit - emitted if limit is not None else self.fetch_size,
                )
                rows = connection.execute(
                    text(CHUNK_PAGE_SQL),
                    {
                        "cursor": cursor,
                        "collection": collection,
                        "force": force,
                        "version_scope": self.version_scope,
                        "limit": batch_limit,
                    },
                ).mappings().all()
                if not rows:
                    return
                batch = [self._to_chunk(row) for row in rows]
                yield batch
                emitted += len(batch)
                cursor = batch[-1].database_id

    def mark_indexed(
        self,
        chunks: Sequence[IndexableChunk],
        point_ids: Sequence[str],
        *,
        collection: str,
    ) -> None:
        if len(chunks) != len(point_ids):
            raise ValueError("chunks and point_ids must have equal length")
        parameters = [
            {
                "id": chunk.database_id,
                "collection": collection,
                "point_id": point_id,
            }
            for chunk, point_id in zip(chunks, point_ids, strict=True)
        ]
        with self.engine.begin() as connection:
            connection.execute(text(MARK_VECTOR_REF_SQL), parameters)
            connection.execute(text(MARK_INDEXED_SQL), parameters)

    def count_expected(self) -> int:
        with self.engine.connect() as connection:
            return int(
                connection.execute(
                    text(EXPECTED_COUNT_SQL),
                    {"version_scope": self.version_scope},
                ).scalar_one()
            )

    def count_marked(self, *, collection: str) -> int:
        with self.engine.connect() as connection:
            return int(
                connection.execute(
                    text(MARKED_COUNT_SQL),
                    {"collection": collection, "version_scope": self.version_scope},
                ).scalar_one()
            )

    @staticmethod
    def _to_chunk(row: Any) -> IndexableChunk:
        payload = {
            "doc_id": row["doc_id"],
            "chunk_id": row["chunk_id"],
            "version_id": str(row["version_id"]),
            "article": row["article"],
            "clause": row["clause"],
            "point": row["point"],
            "document_number": row["document_number"],
            "document_type": row["document_type"],
            "authority": row["authority"],
            "legal_field": row["legal_field"],
            "effective_date": _iso_date(row["effective_date"]),
            "expiry_date": _iso_date(row["expiry_date"]),
            "status": row["status"],
            "chunk_type": row["chunk_type"],
            "content_valid_from": _iso_date(row["content_valid_from"]),
            "content_valid_to": _iso_date(row["content_valid_to"]),
            "version_source_url": row["version_source_url"],
        }
        return IndexableChunk(
            database_id=row["database_id"],
            version_id=row["version_id"],
            external_id=row["chunk_id"],
            retrieval_text=row["retrieval_text"],
            payload={key: value for key, value in payload.items() if value is not None},
        )


class QdrantDenseIndex:
    def __init__(
        self,
        client: QdrantClient,
        embedder: EmbeddingBackend,
        source: PostgresChunkSource,
        *,
        collection: str,
        alias: str | None = None,
        on_disk: bool = True,
        scalar_quantization: bool = True,
    ) -> None:
        self.client = client
        self.embedder = embedder
        self.source = source
        self.collection = collection
        self.alias = alias
        self.on_disk = on_disk
        self.scalar_quantization = scalar_quantization

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            info = self.client.get_collection(self.collection)
            vectors = info.config.params.vectors
            actual_size = getattr(vectors, "size", None)
            if actual_size != self.embedder.dimension:
                raise ValueError(
                    f"Collection {self.collection} has vector size "
                    f"{actual_size}, expected {self.embedder.dimension}"
                )
            actual_distance = getattr(vectors, "distance", None)
            if actual_distance != models.Distance.COSINE:
                raise ValueError(
                    f"Collection {self.collection} uses {actual_distance}, "
                    "expected cosine distance"
                )
            metadata = info.config.metadata or {}
            actual_model = metadata.get("embedding_model")
            if actual_model and actual_model != self.embedder.model_name:
                raise ValueError(
                    f"Collection {self.collection} uses model {actual_model}, "
                    f"expected {self.embedder.model_name}"
                )
            return

        quantization = None
        if self.scalar_quantization:
            quantization = models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=False,
                )
            )
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=self.embedder.dimension,
                distance=models.Distance.COSINE,
                on_disk=self.on_disk,
            ),
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=100,
                on_disk=self.on_disk,
            ),
            quantization_config=quantization,
            on_disk_payload=True,
            metadata={
                "embedding_model": self.embedder.model_name,
                "embedding_dimension": self.embedder.dimension,
            },
        )
        self._create_payload_indexes()

    def build(
        self,
        *,
        limit: int | None = None,
        force: bool = False,
        promote_alias: bool = False,
        progress: Callable[[DenseIndexReport], None] | None = None,
    ) -> DenseIndexReport:
        if promote_alias and limit is not None:
            raise ValueError(
                "Refusing to promote an alias from a partial --limit build"
            )
        self.ensure_collection()
        report = DenseIndexReport(
            collection=self.collection,
            model_name=self.embedder.model_name,
        )
        for chunks in self.source.iter_batches(
            collection=self.collection,
            force=force,
            limit=limit,
        ):
            chunks = [chunk for chunk in chunks if not _is_quarantined(chunk)]
            if not chunks:
                continue
            vectors = self.embedder.encode_documents(
                [chunk.retrieval_text for chunk in chunks]
            )
            self._validate_vectors(vectors, len(chunks))
            point_ids = [
                stable_point_id(chunk.version_id, chunk.external_id)
                for chunk in chunks
            ]
            self.client.upsert(
                collection_name=self.collection,
                wait=True,
                points=models.Batch(
                    ids=point_ids,
                    vectors=vectors.tolist(),
                    payloads=[chunk.payload for chunk in chunks],
                ),
            )
            # Qdrant is the first commit. A crash before this database update
            # is safe because stable point IDs make the next upsert idempotent.
            self.source.mark_indexed(
                chunks, point_ids, collection=self.collection
            )
            report.indexed_points += len(chunks)
            report.batches += 1
            report.last_chunk_id = chunks[-1].external_id
            if progress is not None:
                progress(report)

        if promote_alias and self.alias:
            self.promote_alias()
        return report

    def promote_alias(self) -> None:
        if not self.alias:
            raise ValueError("alias is not configured")
        self._verify_complete()
        aliases = self.client.get_aliases().aliases
        operations: list[Any] = []
        if any(item.alias_name == self.alias for item in aliases):
            operations.append(
                models.DeleteAliasOperation(
                    delete_alias=models.DeleteAlias(alias_name=self.alias)
                )
            )
        operations.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=self.collection,
                    alias_name=self.alias,
                )
            )
        )
        self.client.update_collection_aliases(operations)

    def _verify_complete(self) -> None:
        expected = self.source.count_expected()
        marked = self.source.count_marked(collection=self.collection)
        qdrant_count = self.client.count(
            collection_name=self.collection, exact=True
        ).count
        if marked != expected or qdrant_count != expected:
            raise ValueError(
                "Refusing alias promotion: completeness check failed "
                f"(expected={expected}, postgres_marked={marked}, "
                f"qdrant_points={qdrant_count})"
            )

    def _create_payload_indexes(self) -> None:
        schemas = {
            "doc_id": models.PayloadSchemaType.KEYWORD,
            "chunk_id": models.PayloadSchemaType.KEYWORD,
            "version_id": models.PayloadSchemaType.KEYWORD,
            "document_number": models.PayloadSchemaType.KEYWORD,
            "document_type": models.PayloadSchemaType.KEYWORD,
            "authority": models.PayloadSchemaType.KEYWORD,
            "legal_field": models.PayloadSchemaType.KEYWORD,
            "status": models.PayloadSchemaType.KEYWORD,
            "article": models.PayloadSchemaType.KEYWORD,
            "clause": models.PayloadSchemaType.KEYWORD,
            "effective_date": models.PayloadSchemaType.DATETIME,
            "expiry_date": models.PayloadSchemaType.DATETIME,
            "content_valid_from": models.PayloadSchemaType.DATETIME,
            "content_valid_to": models.PayloadSchemaType.DATETIME,
        }
        for field_name, schema in schemas.items():
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field_name,
                field_schema=schema,
                wait=True,
            )

    def _validate_vectors(self, vectors: np.ndarray, expected_rows: int) -> None:
        if vectors.shape != (expected_rows, self.embedder.dimension):
            raise ValueError(
                f"Embedding shape {vectors.shape} does not match "
                f"({expected_rows}, {self.embedder.dimension})"
            )
        if not np.isfinite(vectors).all():
            raise ValueError("Embedding batch contains NaN or infinity")


def _iso_date(value: date | None) -> str | None:
    return value.isoformat() if value else None


def _is_quarantined(chunk: IndexableChunk) -> bool:
    """Drop chunks from documents whose parse_quality is set to 'quarantine'."""
    payload = chunk.payload or {}
    if (payload.get("parse_quality") or "").strip().casefold() == "quarantine":
        return True
    metadata = payload.get("metadata") or {}
    if isinstance(metadata, dict):
        return (metadata.get("parse_quality") or "").strip().casefold() == "quarantine"
    return False


CHUNK_PAGE_SQL = """
SELECT c.id AS database_id, c.external_id AS chunk_id,
       c.retrieval_text, c.chunk_type, c.metadata->>'article' AS article,
       c.metadata->>'clause' AS clause, c.metadata->>'point' AS point,
       v.id AS version_id, v.content_valid_from, v.content_valid_to,
       v.source_url AS version_source_url, d.external_id AS doc_id,
       d.document_number, d.document_type, d.authority, d.legal_field,
       d.effective_date, d.expiry_date, d.status
FROM chunks c
JOIN document_versions v ON v.id=c.version_id
JOIN documents d ON d.id=c.document_id
LEFT JOIN chunk_vector_refs indexed_ref
  ON indexed_ref.chunk_id=c.id AND indexed_ref.collection=:collection
WHERE ((:version_scope='current' AND v.is_current)
       OR (:version_scope='historical' AND v.content_valid_period IS NOT NULL))
  AND c.is_indexable
  AND c.retrieval_text IS NOT NULL AND c.retrieval_text<>''
  AND (:force OR indexed_ref.chunk_id IS NULL)
  AND (CAST(:cursor AS uuid) IS NULL OR c.id>CAST(:cursor AS uuid))
ORDER BY c.id
LIMIT :limit
"""

MARK_INDEXED_SQL = """
UPDATE chunks
SET qdrant_collection=:collection, qdrant_point_id=:point_id, updated_at=now()
WHERE id=:id
"""

MARK_VECTOR_REF_SQL = """
INSERT INTO chunk_vector_refs (chunk_id, collection, point_id)
VALUES (:id, :collection, :point_id)
ON CONFLICT (chunk_id, collection) DO UPDATE SET
    point_id=EXCLUDED.point_id, updated_at=now()
"""

EXPECTED_COUNT_SQL = """
SELECT count(*)
FROM chunks c JOIN document_versions v ON v.id=c.version_id
WHERE ((:version_scope='current' AND v.is_current)
       OR (:version_scope='historical' AND v.content_valid_period IS NOT NULL))
  AND c.is_indexable
  AND c.retrieval_text IS NOT NULL AND c.retrieval_text<>''
"""

MARKED_COUNT_SQL = """
SELECT count(*)
FROM chunks c JOIN document_versions v ON v.id=c.version_id
JOIN chunk_vector_refs r ON r.chunk_id=c.id AND r.collection=:collection
WHERE ((:version_scope='current' AND v.is_current)
       OR (:version_scope='historical' AND v.content_valid_period IS NOT NULL))
  AND c.is_indexable
  AND c.retrieval_text IS NOT NULL AND c.retrieval_text<>''
  AND r.point_id IS NOT NULL
"""
