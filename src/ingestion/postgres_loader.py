from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import pyarrow.parquet as pq
from sqlalchemy import Connection, Engine, text

from database.models import LegalStatus, RelationshipType


IMPORTER_VERSION = "huggingface_snapshot_v1"
DEFAULT_DATASET_URL = (
    "https://huggingface.co/datasets/"
    "th1nhng0/vietnamese-legal-documents"
)


@dataclass(frozen=True, slots=True)
class EffectivePeriod:
    status: str
    valid_from: date
    valid_to: date | None
    reason: str


@dataclass(slots=True)
class ImportReport:
    run_id: str | None = None
    dry_run: bool = False
    selected_documents: int | None = None
    counters: Counter[str] = field(default_factory=Counter)
    warnings: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dry_run": self.dry_run,
            "selected_documents": self.selected_documents,
            "counters": dict(sorted(self.counters.items())),
            "warnings": dict(sorted(self.warnings.items())),
        }


def normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = " ".join(str(value).split()).strip()
    return normalized or None


def normalize_date(value: Any) -> date | None:
    normalized = normalize_text(value)
    if not normalized:
        return None
    for pattern in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(normalized, pattern).date()
        except ValueError:
            continue
    return None


STATUS_MAPPING = {
    "còn hiệu lực": LegalStatus.EFFECTIVE.value,
    "hết hiệu lực toàn bộ": LegalStatus.EXPIRED.value,
    "hết hiệu lực một phần": LegalStatus.PARTIALLY_EFFECTIVE.value,
    "chưa xác định": LegalStatus.UNKNOWN.value,
    "ngưng hiệu lực": LegalStatus.SUSPENDED.value,
    "ngưng hiệu lực một phần": LegalStatus.PARTIALLY_EFFECTIVE.value,
    "chưa có hiệu lực": LegalStatus.NOT_YET_EFFECTIVE.value,
    # The source label does not state a formal repeal/expiry event.
    "không còn phù hợp": LegalStatus.UNKNOWN.value,
}


def normalize_legal_status(value: Any) -> str:
    normalized = normalize_text(value)
    if not normalized:
        return LegalStatus.UNKNOWN.value
    return STATUS_MAPPING.get(normalized.casefold(), LegalStatus.UNKNOWN.value)


def derive_effective_periods(
    *,
    source_status: Any,
    issued_date: date | None,
    effective_date: date | None,
    expiry_date: date | None,
) -> list[EffectivePeriod]:
    """Derive conservative valid-time rows from one source snapshot.

    The source provides current status, not full event history. Every derived
    row is therefore tagged as such by the loader; unknown transition dates
    are never invented.
    """
    status = normalize_legal_status(source_status)
    start = effective_date or issued_date
    if start is None:
        return []
    if expiry_date is not None and expiry_date <= start:
        expiry_date = None

    if status == LegalStatus.EXPIRED.value and expiry_date is not None:
        return [
            EffectivePeriod(
                LegalStatus.EFFECTIVE.value,
                start,
                expiry_date,
                "derived_from_effective_and_expiry_dates",
            ),
            EffectivePeriod(
                LegalStatus.EXPIRED.value,
                expiry_date,
                None,
                "derived_from_source_expiry_status",
            ),
        ]

    if status == LegalStatus.EXPIRED.value:
        return [
            EffectivePeriod(
                LegalStatus.UNKNOWN.value,
                start,
                None,
                "expiry_transition_date_missing",
            )
        ]

    if status == LegalStatus.NOT_YET_EFFECTIVE.value and effective_date:
        if issued_date is not None and issued_date < effective_date:
            return [
                EffectivePeriod(
                    LegalStatus.NOT_YET_EFFECTIVE.value,
                    issued_date,
                    effective_date,
                    "derived_from_issued_and_effective_dates",
                ),
                EffectivePeriod(
                    LegalStatus.EFFECTIVE.value,
                    effective_date,
                    expiry_date,
                    "scheduled_effective_date_from_source",
                ),
            ]
        start = effective_date

    if (
        status in {LegalStatus.EFFECTIVE.value, LegalStatus.PARTIALLY_EFFECTIVE.value}
        and expiry_date is not None
    ):
        return [
            EffectivePeriod(
                status,
                start,
                expiry_date,
                "derived_from_source_snapshot",
            ),
            EffectivePeriod(
                LegalStatus.EXPIRED.value,
                expiry_date,
                None,
                "expired_after_scheduled_expiry_date",
            ),
        ]

    return [
        EffectivePeriod(
            status,
            start,
            expiry_date,
            "derived_from_source_snapshot",
        )
    ]


RELATIONSHIP_MAPPING = {
    # Direct: doc_id acts on other_doc_id
    "thay thế": RelationshipType.REPLACES.value,
    "bãi bỏ": RelationshipType.REPEALS.value,
    "văn bản hết hiệu lực": RelationshipType.REPEALS.value,
    "sửa đổi, bổ sung": RelationshipType.AMENDS.value,
    "văn bản bổ sung": RelationshipType.SUPPLEMENTS.value,
    "quy định chi tiết, hướng dẫn thi hành": RelationshipType.GUIDES.value,
    "văn bản hd, qđ chi tiết": RelationshipType.GUIDES.value,
    "hướng dẫn áp dụng": RelationshipType.GUIDES.value,
    "căn cứ": RelationshipType.CITES.value,
    "văn bản căn cứ": RelationshipType.CITES.value,
    "dẫn chiếu": RelationshipType.CITES.value,
    "văn bản dẫn chiếu": RelationshipType.CITES.value,
    # Inverted: other_doc_id acts on doc_id
    "văn bản quy định hết hiệu lực": RelationshipType.REPEALS.value,
    "văn bản sửa đổi": RelationshipType.AMENDS.value,
    "văn bản quy định hết hiệu lực 1 phần": RelationshipType.AMENDS.value,
    "văn bản bị hết hiệu lực 1 phần": RelationshipType.AMENDS.value,
    "văn bản được hd, qđ chi tiết": RelationshipType.GUIDES.value,
    "văn bản được bổ sung": RelationshipType.SUPPLEMENTS.value,
    "văn bản được sửa đổi": RelationshipType.AMENDS.value,
    "đính chính": RelationshipType.AMENDS.value,
    "hợp nhất": RelationshipType.RELATED_TO.value,
    "tạm ngưng hiệu lực": RelationshipType.RELATED_TO.value,
    "đình chỉ thi hành": RelationshipType.RELATED_TO.value,
}

INVERTED_RELATIONSHIP_LABELS = frozenset({
    "văn bản quy định hết hiệu lực",
    "văn bản sửa đổi",
    "văn bản quy định hết hiệu lực 1 phần",
    "văn bản được hd, qđ chi tiết",
    "văn bản được bổ sung",
    "văn bản được sửa đổi",
})


def normalize_relationship_type(value: Any) -> str:
    normalized = normalize_text(value)
    if not normalized:
        return RelationshipType.RELATED_TO.value
    return RELATIONSHIP_MAPPING.get(
        normalized.casefold(), RelationshipType.RELATED_TO.value
    )


class PostgresMetadataLoader:
    """Streaming, idempotent Parquet-to-PostgreSQL loader."""

    def __init__(
        self,
        engine: Engine,
        *,
        batch_size: int = 10_000,
        dataset_url: str = DEFAULT_DATASET_URL,
        dataset_revision: str = "local-snapshot",
        dry_run: bool = False,
        version_mode: str = "current",
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.engine = engine
        self.batch_size = batch_size
        self.dataset_url = dataset_url.rstrip("/")
        self.dataset_revision = dataset_revision
        self.dry_run = dry_run
        if version_mode not in {"current", "historical"}:
            raise ValueError("version_mode must be current or historical")
        self.version_mode = version_mode
        self.report = ImportReport(dry_run=dry_run)
        self._selected_ids: set[str] | None = None

    def load(
        self,
        *,
        metadata_path: Path,
        structured_path: Path | None = None,
        chunks_path: Path | None = None,
        relationships_path: Path | None = None,
        limit_documents: int | None = None,
    ) -> ImportReport:
        self._validate_path(metadata_path, "metadata")
        for label, path in (
            ("structured", structured_path),
            ("chunks", chunks_path),
            ("relationships", relationships_path),
        ):
            if path is not None:
                self._validate_path(path, label)

        if limit_documents is not None:
            if limit_documents <= 0:
                raise ValueError("limit_documents must be > 0")
            self._selected_ids = self._read_selected_ids(
                metadata_path, limit_documents
            )
            self.report.selected_documents = len(self._selected_ids)

        if self.dry_run:
            self._scan_metadata(metadata_path)
            if structured_path:
                self._scan_structured(structured_path)
            if chunks_path:
                self._scan_chunks(chunks_path)
            if relationships_path:
                self._scan_relationships(relationships_path)
            return self.report

        with self.engine.connect() as connection:
            connection.execute(
                text(
                    "SELECT pg_advisory_lock("
                    "hashtext('lawchat_metadata_import'))"
                )
            )
            # The lock is session-scoped and survives this commit. End the
            # SQLAlchemy autobegin transaction before per-stage transactions.
            connection.commit()
            try:
                self._create_staging_tables(connection)
                self.report.run_id = self._start_crawl_run(connection)
                self._load_metadata(connection, metadata_path)
                if structured_path:
                    self._load_versions(connection, structured_path)
                if chunks_path:
                    self._load_chunks(connection, chunks_path)
                    self._resolve_all_chunk_parents(connection)
                if relationships_path:
                    self._load_relationships(connection, relationships_path)
                self._finish_crawl_run(connection, "SUCCEEDED")
            except BaseException as exc:
                self.report.warnings[f"fatal:{type(exc).__name__}"] += 1
                self._finish_crawl_run(connection, "FAILED", str(exc))
                raise
            finally:
                connection.execute(
                    text(
                        "SELECT pg_advisory_unlock("
                        "hashtext('lawchat_metadata_import'))"
                    )
                )
                connection.commit()
        return self.report

    def resume_chunk_cleanup(
        self,
        *,
        run_id: str,
        expected_generation_rows: int,
    ) -> ImportReport:
        """Finish a fully loaded generation after interrupted cleanup."""
        if expected_generation_rows <= 0:
            raise ValueError("expected_generation_rows must be > 0")
        self.report.run_id = run_id
        with self.engine.connect() as connection:
            connection.execute(
                text(
                    "SELECT pg_advisory_lock("
                    "hashtext('lawchat_metadata_import'))"
                )
            )
            connection.commit()
            try:
                self._create_staging_tables(connection)
                with connection.begin():
                    connection.execute(
                        text(CHUNK_SCOPE_FROM_GENERATION_SQL),
                        {"run_id": run_id},
                    )
                    actual = int(
                        connection.execute(
                            text(GENERATION_COUNT_SQL),
                            {"run_id": run_id},
                        ).scalar_one()
                    )
                if actual != expected_generation_rows:
                    raise ValueError(
                        "Generation is incomplete; refusing stale cleanup "
                        f"(expected={expected_generation_rows}, actual={actual})"
                    )
                self.report.counters["generation_chunks_verified"] = actual
                self._delete_stale_chunks(connection)
                self._resolve_all_chunk_parents(connection)
                self._finish_crawl_run(connection, "SUCCEEDED")
            except BaseException as exc:
                self.report.warnings[f"fatal:{type(exc).__name__}"] += 1
                self._finish_crawl_run(connection, "FAILED", str(exc))
                raise
            finally:
                connection.execute(
                    text(
                        "SELECT pg_advisory_unlock("
                        "hashtext('lawchat_metadata_import'))"
                    )
                )
                connection.commit()
        return self.report

    @staticmethod
    def _validate_path(path: Path, label: str) -> None:
        if not path.exists():
            raise FileNotFoundError(f"{label}: {path}")

    def _read_selected_ids(self, path: Path, limit: int) -> set[str]:
        selected: list[str] = []
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            columns=["id"], batch_size=min(self.batch_size, limit)
        ):
            for value in batch.column(0).to_pylist():
                external_id = normalize_text(value)
                if external_id:
                    selected.append(external_id)
                    if len(selected) >= limit:
                        return set(selected)
        return set(selected)

    def _accept_document(self, external_id: str | None) -> bool:
        return bool(
            external_id
            and (
                self._selected_ids is None
                or external_id in self._selected_ids
            )
        )

    def _source_url(self, external_id: str) -> str:
        return f"{self.dataset_url}?document_id={external_id}"

    def _iter_parquet(
        self,
        path: Path,
        columns: Sequence[str],
        optional_columns: Sequence[str] = (),
    ) -> Iterator[list[dict[str, Any]]]:
        parquet = pq.ParquetFile(path)
        available = set(parquet.schema_arrow.names)
        missing = set(columns) - available
        if missing:
            raise ValueError(
                f"{path} missing columns: {', '.join(sorted(missing))}"
            )
        selected = [*columns, *(item for item in optional_columns if item in available)]
        for batch in parquet.iter_batches(
            columns=selected, batch_size=self.batch_size
        ):
            yield batch.to_pylist()

    def _normalize_metadata_row(
        self, row: Mapping[str, Any]
    ) -> tuple[tuple[Any, ...], list[tuple[Any, ...]]] | None:
        external_id = normalize_text(row.get("id"))
        title = normalize_text(row.get("title"))
        if not self._accept_document(external_id):
            return None
        if not title:
            self.report.warnings["metadata_missing_title"] += 1
            title = f"Văn bản {external_id}"

        issued = normalize_date(row.get("ngay_ban_hanh"))
        effective = normalize_date(row.get("ngay_co_hieu_luc"))
        expiry = normalize_date(row.get("ngay_het_hieu_luc"))
        date_errors = [
            field_name
            for field_name, parsed in (
                ("ngay_ban_hanh", issued),
                ("ngay_co_hieu_luc", effective),
                ("ngay_het_hieu_luc", expiry),
            )
            if normalize_text(row.get(field_name)) and parsed is None
        ]
        for field_name in date_errors:
            self.report.warnings[f"invalid_date:{field_name}"] += 1
        if effective and expiry and expiry < effective:
            date_errors.append("expiry_before_effective_date")
            self.report.warnings["invalid_date_order"] += 1
            expiry = None

        source_status = normalize_text(row.get("tinh_trang_hieu_luc"))
        legal_status = normalize_legal_status(source_status)
        if expiry and expiry <= date.today():
            if legal_status in {LegalStatus.EFFECTIVE.value, LegalStatus.PARTIALLY_EFFECTIVE.value}:
                legal_status = LegalStatus.EXPIRED.value
        metadata = {
            "importer": IMPORTER_VERSION,
            "dataset_revision": self.dataset_revision,
            "source_status": source_status,
            "collection_source": normalize_text(row.get("nguon_thu_thap")),
            "industry": normalize_text(row.get("nganh")),
            "signer_title": normalize_text(row.get("chuc_danh")),
            "signer": normalize_text(row.get("nguoi_ky")),
            "scope": normalize_text(row.get("pham_vi")),
            "application_info": normalize_text(row.get("thong_tin_ap_dung")),
            "normalization_errors": date_errors,
            "source_dates": {
                "issued": normalize_text(row.get("ngay_ban_hanh")),
                "effective": normalize_text(row.get("ngay_co_hieu_luc")),
                "expiry": normalize_text(row.get("ngay_het_hieu_luc")),
            },
        }
        document_row = (
            external_id,
            title,
            normalize_text(row.get("so_ky_hieu")),
            normalize_text(row.get("loai_van_ban")),
            normalize_text(row.get("co_quan_ban_hanh")),
            issued,
            effective,
            expiry,
            legal_status,
            normalize_text(row.get("linh_vuc")),
            self._source_url(external_id),
            _json(metadata),
        )

        period_rows = []
        for period in derive_effective_periods(
            source_status=source_status,
            issued_date=issued,
            effective_date=effective,
            expiry_date=expiry,
        ):
            period_rows.append(
                (
                    external_id,
                    period.status,
                    period.valid_from,
                    period.valid_to,
                    period.reason,
                    self._source_url(external_id),
                    _json(
                        {
                            "importer": IMPORTER_VERSION,
                            "dataset_revision": self.dataset_revision,
                            "source_status": source_status,
                            "derivation": period.reason,
                        }
                    ),
                )
            )
        if not period_rows:
            self.report.warnings["status_without_start_date"] += 1
        return document_row, period_rows

    def _scan_metadata(self, path: Path) -> None:
        for rows in self._iter_parquet(path, METADATA_COLUMNS):
            for row in rows:
                normalized = self._normalize_metadata_row(row)
                if normalized:
                    self.report.counters["documents"] += 1
                    self.report.counters["effective_status"] += len(
                        normalized[1]
                    )

    def _scan_structured(self, path: Path) -> None:
        for rows in self._iter_parquet(path, STRUCTURED_COLUMNS, OPTIONAL_VERSION_COLUMNS):
            for row in rows:
                if self._normalize_version_row(row):
                    self.report.counters["document_versions"] += 1

    def _scan_chunks(self, path: Path) -> None:
        for rows in self._iter_parquet(path, CHUNK_COLUMNS, OPTIONAL_CHUNK_COLUMNS):
            for row in rows:
                if self._normalize_chunk_row(row):
                    self.report.counters["chunks"] += 1

    def _scan_relationships(self, path: Path) -> None:
        for rows in self._iter_parquet(path, RELATIONSHIP_COLUMNS):
            for row in rows:
                if self._normalize_relationship_row(row):
                    self.report.counters["relationships"] += 1

    def _create_staging_tables(self, connection: Connection) -> None:
        ddl = """
        CREATE TEMP TABLE IF NOT EXISTS stg_documents (
            external_id text, title text, document_number text,
            document_type text, authority text, issued_date date,
            effective_date date, expiry_date date, status text,
            legal_field text, source_url text, metadata jsonb
        ) ON COMMIT PRESERVE ROWS;
        CREATE TEMP TABLE IF NOT EXISTS stg_effective_status (
            external_id text, status text, valid_from date, valid_to date,
            reason text, source_url text, metadata jsonb
        ) ON COMMIT PRESERVE ROWS;
        CREATE TEMP TABLE IF NOT EXISTS stg_versions (
            external_id text, content_hash text, source_url text,
            source_revision text, content_valid_from date, content_valid_to date,
            is_current boolean, parser_version text, metadata jsonb
        ) ON COMMIT PRESERVE ROWS;
        CREATE TEMP TABLE IF NOT EXISTS stg_chunks (
            document_external_id text, external_id text,
            version_source_revision text,
            parent_external_id text, article_key text, article_number text,
            chunk_type text, strategy text, structure_type text,
            ordinal integer, is_indexable boolean, text_content text,
            retrieval_text text, approx_token_count integer,
            chunker_version text, path jsonb, metadata jsonb
        ) ON COMMIT PRESERVE ROWS;
        CREATE TEMP TABLE IF NOT EXISTS stg_relationships (
            source_external_id text, target_external_id text,
            source_relationship text, relationship_type text, metadata jsonb
        ) ON COMMIT PRESERVE ROWS;
        CREATE TEMP TABLE IF NOT EXISTS stg_chunk_import_scope (
            external_id text PRIMARY KEY
        ) ON COMMIT PRESERVE ROWS;
        CREATE TEMP TABLE IF NOT EXISTS stg_version_import_scope (
            external_id text NOT NULL, source_revision text NOT NULL,
            PRIMARY KEY (external_id, source_revision)
        ) ON COMMIT PRESERVE ROWS;
        """
        with connection.begin():
            connection.exec_driver_sql(ddl)
            connection.exec_driver_sql(
                "TRUNCATE stg_chunk_import_scope, stg_version_import_scope"
            )

    def _start_crawl_run(self, connection: Connection) -> str:
        with connection.begin():
            run_id = connection.execute(
                text(
                    "INSERT INTO crawl_runs "
                    "(source, status, crawler_version, stats) "
                    "VALUES (:source, 'RUNNING', :version, '{}'::jsonb) "
                    "RETURNING id"
                ),
                {
                    "source": "huggingface_snapshot",
                    "version": IMPORTER_VERSION,
                },
            ).scalar_one()
        return str(run_id)

    def _finish_crawl_run(
        self,
        connection: Connection,
        status: str,
        error_message: str | None = None,
    ) -> None:
        if not self.report.run_id:
            return
        if connection.in_transaction():
            connection.rollback()
        with connection.begin():
            connection.execute(
                text(
                    "UPDATE crawl_runs SET status=:status, finished_at=now(), "
                    "stats=CAST(:stats AS jsonb), error_message=:error "
                    "WHERE id=CAST(:run_id AS uuid)"
                ),
                {
                    "status": status,
                    "stats": _json(self.report.to_dict()),
                    "error": error_message,
                    "run_id": self.report.run_id,
                },
            )

    def _load_metadata(self, connection: Connection, path: Path) -> None:
        for rows in self._iter_parquet(path, METADATA_COLUMNS):
            documents: list[tuple[Any, ...]] = []
            statuses: list[tuple[Any, ...]] = []
            for row in rows:
                normalized = self._normalize_metadata_row(row)
                if normalized:
                    documents.append(normalized[0])
                    statuses.extend(normalized[1])
            if not documents:
                continue
            with connection.begin():
                self._truncate(connection, "stg_documents", "stg_effective_status")
                self._copy(
                    connection,
                    "stg_documents",
                    DOCUMENT_STAGE_COLUMNS,
                    documents,
                )
                self._copy(
                    connection,
                    "stg_effective_status",
                    STATUS_STAGE_COLUMNS,
                    statuses,
                )
                connection.exec_driver_sql(DOCUMENT_UPSERT_SQL)
                connection.exec_driver_sql(STATUS_REPLACE_SQL)
            self.report.counters["documents"] += len(documents)
            self.report.counters["effective_status"] += len(statuses)

    def _normalize_version_row(
        self, row: Mapping[str, Any]
    ) -> tuple[Any, ...] | None:
        external_id = normalize_text(row.get("id"))
        if not self._accept_document(external_id):
            return None
        structure_json = row.get("structure_json")
        if not structure_json:
            self.report.warnings["version_without_structure"] += 1
            return None
        if not isinstance(structure_json, str):
            structure_json = _json(structure_json)
        content_hash = hashlib.sha256(structure_json.encode("utf-8")).hexdigest()
        source_revision = normalize_text(row.get("source_revision"))
        content_valid_from = normalize_date(row.get("content_valid_from"))
        content_valid_to = normalize_date(row.get("content_valid_to"))
        explicit_source_url = normalize_text(row.get("source_url"))
        source_url = explicit_source_url or self._source_url(external_id)
        if self.version_mode == "historical":
            if not source_revision:
                raise ValueError(
                    f"historical version {external_id} is missing source_revision"
                )
            if content_valid_from is None:
                raise ValueError(
                    f"historical version {external_id} is missing content_valid_from"
                )
            if not explicit_source_url:
                raise ValueError(
                    f"historical version {external_id} requires a source reference"
                )
            if content_valid_to is not None and content_valid_to <= content_valid_from:
                raise ValueError(
                    f"historical version {external_id} has an invalid content interval"
                )
        metadata = {
            "importer": IMPORTER_VERSION,
            "dataset_revision": self.dataset_revision,
            "hash_basis": "structure_json_sha256",
            "parser_status": normalize_text(row.get("parser_status")),
            "parse_quality": normalize_text(row.get("parse_quality")),
            "quality_flags": _json_value(row.get("quality_flags")),
            "structure_type": normalize_text(row.get("structure_type")),
            "is_trusted_snapshot": self.version_mode == "historical",
            "source_reference": (
                source_url if self.version_mode == "historical" else None
            ),
        }
        return (
            external_id,
            content_hash,
            source_url,
            source_revision,
            content_valid_from,
            content_valid_to,
            self.version_mode == "current",
            normalize_text(row.get("parser_version")),
            _json(metadata),
        )

    def _load_versions(self, connection: Connection, path: Path) -> None:
        for rows in self._iter_parquet(path, STRUCTURED_COLUMNS, OPTIONAL_VERSION_COLUMNS):
            normalized = [
                item
                for row in rows
                if (item := self._normalize_version_row(row)) is not None
            ]
            if not normalized:
                continue
            with connection.begin():
                self._truncate(connection, "stg_versions")
                self._copy(
                    connection,
                    "stg_versions",
                    VERSION_STAGE_COLUMNS,
                    normalized,
                )
                connection.exec_driver_sql(CHUNK_SCOPE_FROM_VERSIONS_SQL)
                connection.exec_driver_sql(VERSION_DEACTIVATE_SQL)
                connection.execute(
                    text(VERSION_INSERT_SQL),
                    {"crawl_run_id": self.report.run_id},
                )
                connection.exec_driver_sql(STATUS_VERSION_LINK_SQL)
            self.report.counters["document_versions"] += len(normalized)

    def _normalize_chunk_row(
        self, row: Mapping[str, Any]
    ) -> tuple[Any, ...] | None:
        document_id = normalize_text(row.get("doc_id"))
        external_id = normalize_text(row.get("chunk_id"))
        if not self._accept_document(document_id) or not external_id:
            return None
        version_source_revision = normalize_text(row.get("version_source_revision"))
        if self.version_mode == "historical" and not version_source_revision:
            raise ValueError(
                f"historical chunk {external_id} is missing version_source_revision"
            )
        article_number = normalize_text(row.get("article"))
        path = {
            key: normalize_text(row.get(key))
            for key in ("part", "chapter", "section", "appendix")
            if normalize_text(row.get(key))
        }
        article_key = _article_key(path, article_number)
        metadata = {
            "importer": IMPORTER_VERSION,
            "import_run_id": self.report.run_id,
            "version_source_revision": version_source_revision,
            "parent_external_id": normalize_text(row.get("parent_chunk_id")),
            "parent_type": normalize_text(row.get("parent_type")),
            "article": article_number,
            "clause": normalize_text(row.get("clause")),
            "point": normalize_text(row.get("point")),
            "overlap_from_chunk_id": normalize_text(
                row.get("overlap_from_chunk_id")
            ),
            "overlap_token_count": int(row.get("overlap_token_count") or 0),
            "parse_quality": normalize_text(row.get("parse_quality")),
            "quality_flags": _json_value(row.get("quality_flags")),
        }
        return (
            document_id,
            external_id,
            version_source_revision,
            normalize_text(row.get("parent_chunk_id")),
            article_key,
            article_number,
            normalize_text(row.get("chunk_type")) or "unknown",
            normalize_text(row.get("strategy")) or "unknown",
            normalize_text(row.get("structure_type")),
            int(row.get("ordinal") or 0),
            bool(row.get("is_indexable")),
            normalize_text(row.get("text")) or "",
            normalize_text(row.get("retrieval_text")),
            int(row.get("approx_token_count") or 0),
            normalize_text(row.get("chunker_version")),
            _json(path),
            _json(metadata),
        )

    def _load_chunks(self, connection: Connection, path: Path) -> None:
        for rows in self._iter_parquet(path, CHUNK_COLUMNS, OPTIONAL_CHUNK_COLUMNS):
            normalized = [
                item
                for row in rows
                if (item := self._normalize_chunk_row(row)) is not None
            ]
            if not normalized:
                continue
            with connection.begin():
                self._truncate(connection, "stg_chunks")
                self._copy(
                    connection,
                    "stg_chunks",
                    CHUNK_STAGE_COLUMNS,
                    normalized,
                )
                connection.exec_driver_sql(CHUNK_SCOPE_FROM_CHUNKS_SQL)
                connection.exec_driver_sql(ARTICLE_UPSERT_SQL)
                connection.exec_driver_sql(CHUNK_UPSERT_SQL)
                connection.exec_driver_sql(CHUNK_PARENT_BATCH_SQL)
                connection.exec_driver_sql(CHUNKER_VERSION_SQL)
            self.report.counters["chunks"] += len(normalized)
        self._delete_stale_chunks(connection)

    def _delete_stale_chunks(self, connection: Connection) -> None:
        with connection.begin():
            stale = connection.execute(
                text(STALE_CHUNK_DELETE_SQL),
                {"run_id": self.report.run_id, "version_mode": self.version_mode},
            )
            stale_articles = connection.execute(
                text(STALE_ARTICLE_DELETE_SQL),
                {"version_mode": self.version_mode},
            )
        self.report.counters["stale_chunks_deleted"] += max(
            stale.rowcount, 0
        )
        self.report.counters["stale_articles_deleted"] += max(
            stale_articles.rowcount, 0
        )

    def _resolve_all_chunk_parents(self, connection: Connection) -> None:
        with connection.begin():
            result = connection.exec_driver_sql(CHUNK_PARENT_RECONCILE_SQL)
            self.report.counters["chunk_parents_resolved"] += max(
                result.rowcount, 0
            )

    def _normalize_relationship_row(
        self, row: Mapping[str, Any]
    ) -> tuple[Any, ...] | None:
        raw_source_id = normalize_text(row.get("doc_id"))
        raw_target_id = normalize_text(row.get("other_doc_id"))
        label = normalize_text(row.get("relationship"))
        if not raw_source_id or not raw_target_id or not label:
            return None
        is_inverted = label.casefold() in INVERTED_RELATIONSHIP_LABELS
        source_id = raw_target_id if is_inverted else raw_source_id
        target_id = raw_source_id if is_inverted else raw_target_id
        if not self._accept_document(raw_source_id):
            return None
        relation_type = normalize_relationship_type(label)
        return (
            source_id,
            target_id,
            label,
            relation_type,
            _json(
                {
                    "importer": IMPORTER_VERSION,
                    "dataset_revision": self.dataset_revision,
                    "source_relationship": label,
                    "is_inverted": is_inverted,
                }
            ),
        )

    def _load_relationships(self, connection: Connection, path: Path) -> None:
        for rows in self._iter_parquet(path, RELATIONSHIP_COLUMNS):
            normalized = [
                item
                for row in rows
                if (item := self._normalize_relationship_row(row)) is not None
            ]
            if not normalized:
                continue
            with connection.begin():
                self._truncate(connection, "stg_relationships")
                self._copy(
                    connection,
                    "stg_relationships",
                    RELATIONSHIP_STAGE_COLUMNS,
                    normalized,
                )
                connection.exec_driver_sql(RELATIONSHIP_UPSERT_SQL)
            self.report.counters["relationships"] += len(normalized)

    @staticmethod
    def _truncate(connection: Connection, *tables: str) -> None:
        connection.exec_driver_sql("TRUNCATE " + ", ".join(tables))

    @staticmethod
    def _copy(
        connection: Connection,
        table: str,
        columns: Sequence[str],
        rows: Iterable[tuple[Any, ...]],
    ) -> None:
        raw_connection = connection.connection.driver_connection
        sql = f"COPY {table} ({', '.join(columns)}) FROM STDIN"
        with raw_connection.cursor().copy(sql) as copy:
            for row in rows:
                copy.write_row(row)


def _article_key(path: Mapping[str, str], article: str | None) -> str | None:
    if not article:
        return None
    parts = [
        f"{key}={path.get(key, '-')}"
        for key in ("part", "chapter", "section", "appendix")
    ]
    parts.append(f"article={article}")
    return "|".join(parts)


def _json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), default=str
    )


def _json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


METADATA_COLUMNS = (
    "id", "title", "so_ky_hieu", "ngay_ban_hanh", "loai_van_ban",
    "ngay_co_hieu_luc", "ngay_het_hieu_luc", "nguon_thu_thap",
    "nganh", "linh_vuc", "co_quan_ban_hanh", "chuc_danh", "nguoi_ky",
    "pham_vi", "thong_tin_ap_dung", "tinh_trang_hieu_luc",
)
STRUCTURED_COLUMNS = (
    "id", "parser_status", "parse_quality", "quality_flags",
    "structure_type", "parser_version", "structure_json",
)
OPTIONAL_VERSION_COLUMNS = (
    "source_revision", "source_url", "content_valid_from", "content_valid_to",
)
CHUNK_COLUMNS = (
    "chunk_id", "doc_id", "parent_chunk_id", "parent_type", "chunk_type",
    "strategy", "is_indexable", "structure_type", "part", "chapter",
    "section", "appendix", "article", "clause", "point", "ordinal",
    "text", "retrieval_text", "approx_token_count", "overlap_from_chunk_id",
    "overlap_token_count", "parse_quality", "quality_flags", "chunker_version",
)
OPTIONAL_CHUNK_COLUMNS = ("version_source_revision",)
RELATIONSHIP_COLUMNS = ("doc_id", "other_doc_id", "relationship")

DOCUMENT_STAGE_COLUMNS = (
    "external_id", "title", "document_number", "document_type", "authority",
    "issued_date", "effective_date", "expiry_date", "status", "legal_field",
    "source_url", "metadata",
)
STATUS_STAGE_COLUMNS = (
    "external_id", "status", "valid_from", "valid_to", "reason",
    "source_url", "metadata",
)
VERSION_STAGE_COLUMNS = (
    "external_id", "content_hash", "source_url", "source_revision",
    "content_valid_from", "content_valid_to", "is_current", "parser_version",
    "metadata",
)
CHUNK_STAGE_COLUMNS = (
    "document_external_id", "external_id", "version_source_revision",
    "parent_external_id", "article_key",
    "article_number", "chunk_type", "strategy", "structure_type", "ordinal",
    "is_indexable", "text_content", "retrieval_text", "approx_token_count",
    "chunker_version", "path", "metadata",
)
RELATIONSHIP_STAGE_COLUMNS = (
    "source_external_id", "target_external_id", "source_relationship",
    "relationship_type", "metadata",
)

DOCUMENT_UPSERT_SQL = """
INSERT INTO documents (
    external_id, title, document_number, document_type, authority, issued_date,
    effective_date, expiry_date, status, legal_field, source_url, metadata
)
SELECT external_id, title, document_number, document_type, authority, issued_date,
       effective_date, expiry_date, status, legal_field, source_url, metadata
FROM stg_documents
ON CONFLICT (external_id) DO UPDATE SET
    title=CASE WHEN documents.metadata->>'official_verified'='true' OR documents.metadata->>'trusted_snapshot'='true'
               THEN documents.title ELSE EXCLUDED.title END,
    document_number=CASE WHEN documents.metadata->>'official_verified'='true' OR documents.metadata->>'trusted_snapshot'='true'
               THEN documents.document_number ELSE EXCLUDED.document_number END,
    document_type=CASE WHEN documents.metadata->>'official_verified'='true' OR documents.metadata->>'trusted_snapshot'='true'
               THEN documents.document_type ELSE EXCLUDED.document_type END,
    authority=CASE WHEN documents.metadata->>'official_verified'='true' OR documents.metadata->>'trusted_snapshot'='true'
               THEN documents.authority ELSE EXCLUDED.authority END,
    issued_date=CASE WHEN documents.metadata->>'official_verified'='true' OR documents.metadata->>'trusted_snapshot'='true'
               THEN documents.issued_date ELSE EXCLUDED.issued_date END,
    effective_date=CASE WHEN documents.metadata->>'official_verified'='true' OR documents.metadata->>'trusted_snapshot'='true'
               THEN documents.effective_date ELSE EXCLUDED.effective_date END,
    expiry_date=CASE WHEN documents.metadata->>'official_verified'='true' OR documents.metadata->>'trusted_snapshot'='true'
               THEN documents.expiry_date ELSE EXCLUDED.expiry_date END,
    status=CASE WHEN documents.metadata->>'official_verified'='true' OR documents.metadata->>'trusted_snapshot'='true'
               THEN documents.status ELSE EXCLUDED.status END,
    legal_field=CASE WHEN documents.metadata->>'official_verified'='true' OR documents.metadata->>'trusted_snapshot'='true'
               THEN documents.legal_field ELSE EXCLUDED.legal_field END,
    source_url=CASE WHEN documents.metadata->>'official_verified'='true' OR documents.metadata->>'trusted_snapshot'='true'
               THEN documents.source_url ELSE EXCLUDED.source_url END,
    metadata=documents.metadata || EXCLUDED.metadata, updated_at=now()
"""

STATUS_REPLACE_SQL = f"""
DELETE FROM effective_status es
USING documents d, stg_documents s
WHERE es.document_id=d.id AND d.external_id=s.external_id
  AND es.metadata->>'importer'='{IMPORTER_VERSION}';
INSERT INTO effective_status (
    document_id, status, valid_from, valid_to, reason, source_url, metadata
)
SELECT d.id, s.status, s.valid_from, s.valid_to, s.reason, s.source_url, s.metadata
FROM stg_effective_status s
JOIN documents d ON d.external_id=s.external_id
WHERE NOT EXISTS (
    SELECT 1 FROM effective_status official
    WHERE official.document_id=d.id
      AND COALESCE((official.metadata->>'is_official')::boolean, false)=true
      AND official.valid_period
          && daterange(s.valid_from, s.valid_to, '[)')
)
"""

VERSION_DEACTIVATE_SQL = """
UPDATE document_versions v SET is_current=false, updated_at=now()
FROM documents d JOIN stg_versions s ON s.external_id=d.external_id
WHERE v.document_id=d.id AND v.is_current AND s.is_current
  AND v.content_hash<>s.content_hash
"""

CHUNK_SCOPE_FROM_VERSIONS_SQL = """
INSERT INTO stg_chunk_import_scope (external_id)
SELECT DISTINCT external_id FROM stg_versions
ON CONFLICT (external_id) DO NOTHING
;
INSERT INTO stg_version_import_scope (external_id, source_revision)
SELECT DISTINCT external_id, COALESCE(source_revision, '__CURRENT__')
FROM stg_versions
ON CONFLICT (external_id, source_revision) DO NOTHING
"""

VERSION_INSERT_SQL = """
INSERT INTO document_versions (
    document_id, crawl_run_id, version_number, content_hash, source_url,
    source_revision, content_valid_from, content_valid_to,
    is_current, parser_version, metadata
)
SELECT d.id, CAST(:crawl_run_id AS uuid),
       COALESCE(existing.max_version, 0)
       + row_number() OVER (
           PARTITION BY d.id
           ORDER BY s.content_valid_from NULLS LAST, s.content_hash
         ),
       s.content_hash, s.source_url, s.source_revision,
       s.content_valid_from, s.content_valid_to,
       s.is_current, s.parser_version, s.metadata
FROM stg_versions s JOIN documents d ON d.external_id=s.external_id
LEFT JOIN LATERAL (
    SELECT max(v.version_number) AS max_version
    FROM document_versions v WHERE v.document_id=d.id
) existing ON true
ON CONFLICT (document_id, content_hash) DO UPDATE SET
    crawl_run_id=EXCLUDED.crawl_run_id, source_url=EXCLUDED.source_url,
    source_revision=COALESCE(EXCLUDED.source_revision, document_versions.source_revision),
    content_valid_from=COALESCE(EXCLUDED.content_valid_from, document_versions.content_valid_from),
    content_valid_to=COALESCE(EXCLUDED.content_valid_to, document_versions.content_valid_to),
    is_current=(document_versions.is_current OR EXCLUDED.is_current),
    parser_version=EXCLUDED.parser_version,
    metadata=document_versions.metadata || EXCLUDED.metadata, updated_at=now()
"""

STATUS_VERSION_LINK_SQL = f"""
UPDATE effective_status es SET source_version_id=v.id, updated_at=now()
FROM documents d JOIN stg_versions s ON s.external_id=d.external_id
JOIN document_versions v ON v.document_id=d.id AND v.is_current
WHERE es.document_id=d.id
  AND s.is_current
  AND es.metadata->>'importer'='{IMPORTER_VERSION}'
"""

ARTICLE_UPSERT_SQL = """
INSERT INTO articles (
    version_id, document_id, article_key, article_number, ordinal,
    path, text_content, metadata
)
SELECT DISTINCT ON (v.id, s.article_key)
       v.id, d.id, s.article_key, s.article_number, s.ordinal,
       s.path,
       CASE WHEN s.chunk_type IN ('article', 'article_parent')
            THEN s.text_content ELSE NULL END,
       jsonb_build_object('importer', 'huggingface_snapshot_v1')
FROM stg_chunks s
JOIN documents d ON d.external_id=s.document_external_id
JOIN document_versions v ON v.document_id=d.id
 AND ((s.version_source_revision IS NOT NULL AND v.source_revision=s.version_source_revision)
      OR (s.version_source_revision IS NULL AND v.is_current))
WHERE s.article_key IS NOT NULL
ORDER BY v.id, s.article_key,
         (s.chunk_type IN ('article', 'article_parent')) DESC, s.ordinal
ON CONFLICT (version_id, article_key) DO UPDATE SET
    article_number=EXCLUDED.article_number, path=EXCLUDED.path,
    text_content=COALESCE(EXCLUDED.text_content, articles.text_content),
    metadata=articles.metadata || EXCLUDED.metadata, updated_at=now()
"""

CHUNK_UPSERT_SQL = """
INSERT INTO chunks (
    version_id, document_id, article_id, external_id, chunk_type, strategy,
    structure_type, ordinal, is_indexable, text_content, retrieval_text,
    approx_token_count, metadata
)
SELECT v.id, d.id, a.id, s.external_id, s.chunk_type, s.strategy,
       s.structure_type, s.ordinal, s.is_indexable, s.text_content,
       s.retrieval_text, s.approx_token_count, s.metadata
FROM stg_chunks s
JOIN documents d ON d.external_id=s.document_external_id
JOIN document_versions v ON v.document_id=d.id
 AND ((s.version_source_revision IS NOT NULL AND v.source_revision=s.version_source_revision)
      OR (s.version_source_revision IS NULL AND v.is_current))
LEFT JOIN articles a ON a.version_id=v.id AND a.article_key=s.article_key
ON CONFLICT (version_id, external_id) DO UPDATE SET
    article_id=EXCLUDED.article_id, chunk_type=EXCLUDED.chunk_type,
    strategy=EXCLUDED.strategy, structure_type=EXCLUDED.structure_type,
    ordinal=EXCLUDED.ordinal, is_indexable=EXCLUDED.is_indexable,
    text_content=EXCLUDED.text_content, retrieval_text=EXCLUDED.retrieval_text,
    approx_token_count=EXCLUDED.approx_token_count,
    metadata=chunks.metadata || EXCLUDED.metadata, updated_at=now()
"""

CHUNK_SCOPE_FROM_CHUNKS_SQL = """
INSERT INTO stg_chunk_import_scope (external_id)
SELECT DISTINCT document_external_id FROM stg_chunks
ON CONFLICT (external_id) DO NOTHING
;
INSERT INTO stg_version_import_scope (external_id, source_revision)
SELECT DISTINCT document_external_id,
       COALESCE(version_source_revision, '__CURRENT__')
FROM stg_chunks
ON CONFLICT (external_id, source_revision) DO NOTHING
"""

STALE_CHUNK_DELETE_SQL = f"""
DELETE FROM chunks c
USING document_versions v, documents d, stg_version_import_scope scope
WHERE c.version_id=v.id
  AND c.document_id=d.id AND d.external_id=scope.external_id
  AND ((scope.source_revision='__CURRENT__' AND v.is_current)
       OR v.source_revision=scope.source_revision)
  AND c.metadata->>'importer'='{IMPORTER_VERSION}'
  AND c.metadata->>'import_run_id' IS DISTINCT FROM :run_id
"""

STALE_ARTICLE_DELETE_SQL = f"""
DELETE FROM articles a
USING document_versions v, documents d, stg_version_import_scope scope
WHERE a.version_id=v.id
  AND a.document_id=d.id AND d.external_id=scope.external_id
  AND ((scope.source_revision='__CURRENT__' AND v.is_current)
       OR v.source_revision=scope.source_revision)
  AND a.metadata->>'importer'='{IMPORTER_VERSION}'
  AND NOT EXISTS (SELECT 1 FROM chunks c WHERE c.article_id=a.id)
"""

CHUNK_SCOPE_FROM_GENERATION_SQL = """
INSERT INTO stg_chunk_import_scope (external_id)
SELECT DISTINCT d.external_id
FROM chunks c JOIN documents d ON d.id=c.document_id
WHERE c.metadata->>'import_run_id'=:run_id
ON CONFLICT (external_id) DO NOTHING
;
INSERT INTO stg_version_import_scope (external_id, source_revision)
SELECT DISTINCT d.external_id,
       COALESCE(c.metadata->>'version_source_revision', '__CURRENT__')
FROM chunks c JOIN documents d ON d.id=c.document_id
WHERE c.metadata->>'import_run_id'=:run_id
ON CONFLICT (external_id, source_revision) DO NOTHING
"""

GENERATION_COUNT_SQL = """
SELECT count(*) FROM chunks
WHERE metadata->>'import_run_id'=:run_id
"""

CHUNK_PARENT_BATCH_SQL = """
UPDATE chunks child SET parent_chunk_id=parent.id, updated_at=now()
FROM stg_chunks s
JOIN documents d ON d.external_id=s.document_external_id
JOIN document_versions v ON v.document_id=d.id
 AND ((s.version_source_revision IS NOT NULL AND v.source_revision=s.version_source_revision)
      OR (s.version_source_revision IS NULL AND v.is_current))
JOIN chunks parent ON parent.version_id=v.id
                  AND parent.external_id=s.parent_external_id
WHERE child.version_id=v.id AND child.external_id=s.external_id
  AND s.parent_external_id IS NOT NULL
  AND child.parent_chunk_id IS DISTINCT FROM parent.id
"""

CHUNKER_VERSION_SQL = """
UPDATE document_versions v SET chunker_version=x.chunker_version, updated_at=now()
FROM (
    SELECT d.id AS document_id, s.version_source_revision,
           max(s.chunker_version) AS chunker_version
    FROM stg_chunks s JOIN documents d ON d.external_id=s.document_external_id
    WHERE s.chunker_version IS NOT NULL GROUP BY d.id, s.version_source_revision
) x
WHERE v.document_id=x.document_id
  AND ((x.version_source_revision IS NOT NULL AND v.source_revision=x.version_source_revision)
       OR (x.version_source_revision IS NULL AND v.is_current))
  AND v.chunker_version IS DISTINCT FROM x.chunker_version
"""

CHUNK_PARENT_RECONCILE_SQL = """
UPDATE chunks child SET parent_chunk_id=parent.id, updated_at=now()
FROM chunks parent, document_versions v, documents d,
     stg_chunk_import_scope scope
WHERE child.version_id=v.id
  AND child.document_id=d.id AND d.external_id=scope.external_id
  AND parent.version_id=child.version_id
  AND child.metadata->>'parent_external_id'=parent.external_id
  AND child.parent_chunk_id IS DISTINCT FROM parent.id
"""

RELATIONSHIP_UPSERT_SQL = """
INSERT INTO document_relationships (
    source_document_id, target_document_id, relationship_type,
    source_relationship, target_external_id, metadata
)
SELECT DISTINCT ON (source.id, s.target_external_id, s.source_relationship)
       source.id, target.id, s.relationship_type, s.source_relationship,
       s.target_external_id, s.metadata
FROM stg_relationships s
JOIN documents source ON source.external_id=s.source_external_id
LEFT JOIN documents target ON target.external_id=s.target_external_id
ORDER BY source.id, s.target_external_id, s.source_relationship
ON CONFLICT ON CONSTRAINT uq_document_relationships_source_edge DO UPDATE SET
    target_document_id=EXCLUDED.target_document_id,
    relationship_type=EXCLUDED.relationship_type,
    metadata=document_relationships.metadata || EXCLUDED.metadata,
    updated_at=now()
"""
