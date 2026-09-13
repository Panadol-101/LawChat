from __future__ import annotations

import json
import os
import unicodedata
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

import tantivy
from sqlalchemy import Engine, text


BM25_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class BM25Settings:
    root: Path = Path("data/indexes/bm25")
    index_name: str = "legal_bm25_v1"
    alias: str = "legal_bm25_current"
    source_collection: str = "legal_chunks_bge_m3_1024_v1"
    writer_heap_size: int = 256_000_000
    writer_threads: int = 2

    @classmethod
    def from_env(cls) -> "BM25Settings":
        return cls(
            root=Path(os.getenv("BM25_INDEX_ROOT", "data/indexes/bm25")),
            index_name=os.getenv("BM25_INDEX_NAME", "legal_bm25_v1"),
            alias=os.getenv("BM25_INDEX_ALIAS", "legal_bm25_current"),
            source_collection=os.getenv(
                "QDRANT_COLLECTION",
                "legal_chunks_bge_m3_1024_v1",
            ),
            writer_heap_size=_positive_int("BM25_WRITER_HEAP_SIZE", 256_000_000),
            writer_threads=_positive_int("BM25_WRITER_THREADS", 2),
        )

    @property
    def index_path(self) -> Path:
        return self.root / "indexes" / self.index_name

    @property
    def manifest_path(self) -> Path:
        return self.index_path / "lawchat_manifest.json"

    @property
    def alias_path(self) -> Path:
        return self.root / "aliases" / f"{self.alias}.json"


@dataclass(frozen=True, slots=True)
class LexicalChunk:
    database_id: uuid.UUID
    point_id: str
    chunk_id: str
    doc_id: str
    document_number: str | None
    document_type: str | None
    authority: str | None
    legal_field: str | None
    title: str
    article: str | None
    clause: str | None
    point: str | None
    retrieval_text: str


@dataclass(slots=True)
class BM25BuildReport:
    index_name: str
    source_collection: str
    expected_points: int
    indexed_points: int = 0
    newly_indexed_points: int = 0
    batches: int = 0
    last_database_id: str | None = None
    complete: bool = False


@dataclass(frozen=True, slots=True)
class BM25Manifest:
    schema_version: int
    index_name: str
    source_collection: str
    expected_points: int
    indexed_points: int
    last_database_id: str | None
    complete: bool
    tantivy_version: str


class LexicalChunkSource(Protocol):
    def iter_batches(
        self,
        *,
        collection: str,
        after: uuid.UUID | None = None,
        limit: int | None = None,
    ) -> Iterator[list[LexicalChunk]]: ...

    def count_expected(self, *, collection: str) -> int: ...


class PostgresLexicalSource:
    def __init__(
        self,
        engine: Engine,
        *,
        fetch_size: int = 10_000,
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
        after: uuid.UUID | None = None,
        limit: int | None = None,
    ) -> Iterator[list[LexicalChunk]]:
        emitted = 0
        cursor = after
        while limit is None or emitted < limit:
            batch_limit = min(
                self.fetch_size,
                limit - emitted if limit is not None else self.fetch_size,
            )
            with self.engine.connect() as connection:
                rows = connection.execute(
                    text(BM25_PAGE_SQL),
                    {
                        "collection": collection,
                        "version_scope": self.version_scope,
                        "cursor": cursor,
                        "limit": batch_limit,
                    },
                ).mappings().all()
            if not rows:
                return
            batch = [_to_lexical_chunk(row) for row in rows]
            yield batch
            emitted += len(batch)
            cursor = batch[-1].database_id

    def count_expected(self, *, collection: str) -> int:
        with self.engine.connect() as connection:
            return int(
                connection.execute(
                    text(BM25_EXPECTED_COUNT_SQL),
                    {"collection": collection, "version_scope": self.version_scope},
                ).scalar_one()
            )


class TantivyBM25Index:
    def __init__(self, source: LexicalChunkSource, settings: BM25Settings) -> None:
        self.source = source
        self.settings = settings

    def build(
        self,
        *,
        limit: int | None = None,
        promote_alias: bool = False,
        progress: Callable[[BM25BuildReport], None] | None = None,
    ) -> BM25BuildReport:
        if limit is not None and limit <= 0:
            raise ValueError("limit must be > 0")
        manifest = self._load_manifest()
        index = self._open_or_create_index(manifest)
        expected = self.source.count_expected(
            collection=self.settings.source_collection
        )
        if manifest is not None and manifest.expected_points != expected:
            raise ValueError(
                "PostgreSQL source count changed; build a new BM25 index version "
                f"(manifest={manifest.expected_points}, current={expected})"
            )

        indexed_points = manifest.indexed_points if manifest else 0
        cursor = (
            uuid.UUID(manifest.last_database_id)
            if manifest and manifest.last_database_id
            else None
        )
        report = BM25BuildReport(
            index_name=self.settings.index_name,
            source_collection=self.settings.source_collection,
            expected_points=expected,
            indexed_points=indexed_points,
            last_database_id=str(cursor) if cursor else None,
            complete=bool(manifest and manifest.complete),
        )
        if report.complete:
            index.reload()
            if index.searcher().num_docs != expected:
                raise ValueError("complete BM25 manifest does not match Tantivy count")
            if promote_alias:
                self.promote_alias()
            return report

        writer = index.writer(
            heap_size=self.settings.writer_heap_size,
            num_threads=self.settings.writer_threads,
        )
        reconcile_first_batch = manifest is not None and indexed_points > 0
        try:
            for batch in self.source.iter_batches(
                collection=self.settings.source_collection,
                after=cursor,
                limit=limit,
            ):
                for chunk in batch:
                    if reconcile_first_batch:
                        writer.delete_documents_by_term("point_id", chunk.point_id)
                    writer.add_document(_to_tantivy_document(chunk))
                writer.commit()
                index.reload()
                reconcile_first_batch = False
                indexed_points += len(batch)
                cursor = batch[-1].database_id
                report.indexed_points = indexed_points
                report.newly_indexed_points += len(batch)
                report.batches += 1
                report.last_database_id = str(cursor)
                self._write_manifest(
                    self._manifest(expected, indexed_points, cursor, complete=False)
                )
                if progress is not None:
                    progress(report)
        finally:
            writer.wait_merging_threads()

        index.reload()
        actual_docs = index.searcher().num_docs
        report.complete = limit is None and actual_docs == expected
        if limit is None and not report.complete:
            raise ValueError(
                "BM25 completeness check failed "
                f"(expected={expected}, manifest={indexed_points}, tantivy={actual_docs})"
            )
        self._write_manifest(
            self._manifest(expected, indexed_points, cursor, report.complete)
        )
        if promote_alias:
            if not report.complete:
                raise ValueError("cannot promote an incomplete BM25 index")
            self.promote_alias()
        return report

    def promote_alias(self) -> None:
        manifest = self._load_manifest()
        if manifest is None or not manifest.complete:
            raise ValueError("cannot promote BM25 index without a complete manifest")
        _atomic_write_json(
            self.settings.alias_path,
            {
                "index_name": self.settings.index_name,
                "schema_version": BM25_SCHEMA_VERSION,
                "source_collection": self.settings.source_collection,
            },
        )

    def _open_or_create_index(
        self,
        manifest: BM25Manifest | None,
    ) -> tantivy.Index:
        if manifest is not None:
            if manifest.schema_version != BM25_SCHEMA_VERSION:
                raise ValueError(
                    f"BM25 schema version {manifest.schema_version} is unsupported"
                )
            return tantivy.Index.open(str(self.settings.index_path))
        if self.settings.index_path.exists() and any(
            self.settings.index_path.iterdir()
        ):
            raise ValueError(
                f"BM25 index directory exists without a manifest: "
                f"{self.settings.index_path}"
            )
        self.settings.index_path.mkdir(parents=True, exist_ok=True)
        return tantivy.Index(
            create_bm25_schema(),
            path=str(self.settings.index_path),
        )

    def _load_manifest(self) -> BM25Manifest | None:
        path = self.settings.manifest_path
        if not path.exists():
            return None
        return BM25Manifest(**json.loads(path.read_text(encoding="utf-8")))

    def _write_manifest(self, manifest: BM25Manifest) -> None:
        _atomic_write_json(self.settings.manifest_path, asdict(manifest))

    def _manifest(
        self,
        expected: int,
        indexed: int,
        cursor: uuid.UUID | None,
        complete: bool,
    ) -> BM25Manifest:
        return BM25Manifest(
            schema_version=BM25_SCHEMA_VERSION,
            index_name=self.settings.index_name,
            source_collection=self.settings.source_collection,
            expected_points=expected,
            indexed_points=indexed,
            last_database_id=str(cursor) if cursor else None,
            complete=complete,
            tantivy_version=tantivy.__version__,
        )


def create_bm25_schema() -> tantivy.Schema:
    builder = tantivy.SchemaBuilder()
    for field in ("point_id", "chunk_id"):
        builder.add_text_field(field, stored=True, tokenizer_name="raw")
    for field in (
        "doc_id",
        "document_number_normalized",
        "document_type",
        "authority",
        "legal_field",
        "article",
        "clause",
        "point",
    ):
        builder.add_text_field(field, tokenizer_name="raw")
    builder.add_text_field("document_number")
    builder.add_text_field("title")
    builder.add_text_field("retrieval_text")
    return builder.build()


def normalize_legal_identifier(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value).casefold().strip()
    normalized = normalized.replace("–", "-").replace("—", "-")
    return "".join(normalized.split())


def normalize_keyword(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def _to_tantivy_document(chunk: LexicalChunk) -> tantivy.Document:
    values: dict[str, str] = {
        "point_id": chunk.point_id,
        "chunk_id": chunk.chunk_id,
        "doc_id": normalize_keyword(chunk.doc_id),
        "title": chunk.title,
        "retrieval_text": chunk.retrieval_text,
    }
    optional = {
        "document_number": chunk.document_number,
        "document_number_normalized": (
            normalize_legal_identifier(chunk.document_number)
            if chunk.document_number
            else None
        ),
        "document_type": _normalized_optional(chunk.document_type),
        "authority": _normalized_optional(chunk.authority),
        "legal_field": _normalized_optional(chunk.legal_field),
        "article": _normalized_optional(chunk.article),
        "clause": _normalized_optional(chunk.clause),
        "point": _normalized_optional(chunk.point),
    }
    values.update({key: value for key, value in optional.items() if value})
    return tantivy.Document(**values)


def _normalized_optional(value: str | None) -> str | None:
    return normalize_keyword(value) if value else None


def _to_lexical_chunk(row: Any) -> LexicalChunk:
    return LexicalChunk(
        database_id=row["database_id"],
        point_id=row["point_id"],
        chunk_id=row["chunk_id"],
        doc_id=row["doc_id"],
        document_number=row["document_number"],
        document_type=row["document_type"],
        authority=row["authority"],
        legal_field=row["legal_field"],
        title=row["title"],
        article=row["article"],
        clause=row["clause"],
        point=row["point"],
        retrieval_text=row["retrieval_text"],
    )


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


BM25_PAGE_SQL = """
SELECT c.id AS database_id, r.point_id,
       c.external_id AS chunk_id, d.external_id AS doc_id,
       d.document_number, d.document_type, d.authority, d.legal_field,
       d.title, c.metadata->>'article' AS article,
       c.metadata->>'clause' AS clause, c.metadata->>'point' AS point,
       c.retrieval_text
FROM chunks c
JOIN document_versions v ON v.id=c.version_id
JOIN documents d ON d.id=c.document_id
JOIN chunk_vector_refs r ON r.chunk_id=c.id AND r.collection=:collection
WHERE ((:version_scope='current' AND v.is_current)
       OR (:version_scope='historical' AND v.content_valid_period IS NOT NULL))
  AND c.is_indexable
  AND c.retrieval_text IS NOT NULL AND c.retrieval_text<>''
  AND r.point_id IS NOT NULL
  AND (CAST(:cursor AS uuid) IS NULL OR c.id>CAST(:cursor AS uuid))
ORDER BY c.id
LIMIT :limit
"""


BM25_EXPECTED_COUNT_SQL = """
SELECT count(*)
FROM chunks c
JOIN document_versions v ON v.id=c.version_id
JOIN chunk_vector_refs r ON r.chunk_id=c.id AND r.collection=:collection
WHERE ((:version_scope='current' AND v.is_current)
       OR (:version_scope='historical' AND v.content_valid_period IS NOT NULL))
  AND c.is_indexable
  AND c.retrieval_text IS NOT NULL AND c.retrieval_text<>''
  AND r.point_id IS NOT NULL
"""
