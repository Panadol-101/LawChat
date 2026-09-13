from datetime import date

import pytest
from sqlalchemy.dialects import postgresql

from lawchat.database import LegalMetadataFilter, MetadataQueries


def _sql(statement) -> str:
    return str(
        statement.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )


def test_effective_document_query_uses_valid_period_and_metadata_filters():
    filters = LegalMetadataFilter(
        as_of=date(2026, 8, 25),
        document_numbers=("123/2026/NĐ-CP",),
        authorities=("Chính phủ",),
    )

    sql = _sql(MetadataQueries.effective_documents(filters))

    assert "effective_status.valid_period @> '2026-08-25'" in sql
    assert "effective_status.status IN ('EFFECTIVE', 'PARTIALLY_EFFECTIVE')" in sql
    assert "documents.document_number IN ('123/2026/NĐ-CP')" in sql
    assert "documents.authority IN ('Chính phủ')" in sql
    assert "Bản dịch văn bản" in sql
    assert "documents.document_type NOT IN" in sql


def test_metadata_filter_can_target_graph_document_external_ids():
    sql = _sql(
        MetadataQueries.hydrate_qdrant_points(
            ["point-1"],
            LegalMetadataFilter(
                as_of=date(2026, 8, 25),
                document_ids=("target-doc",),
            ),
        )
    )

    assert "documents.external_id IN ('target-doc')" in sql


def test_qdrant_query_only_returns_current_indexable_chunks():
    statement = MetadataQueries.qdrant_point_ids(
        LegalMetadataFilter(as_of=date(2026, 8, 25)),
        collection="legal-v1",
    )

    sql = _sql(statement)

    assert "document_versions.is_current IS true" in sql
    assert "chunks.is_indexable IS true" in sql
    assert "chunks.qdrant_point_id IS NOT NULL" in sql
    assert "chunks.qdrant_collection = 'legal-v1'" in sql


def test_document_ids_are_ready_for_qdrant_doc_id_payload_filter():
    sql = _sql(
        MetadataQueries.effective_document_external_ids(
            LegalMetadataFilter(
                as_of=date(2026, 8, 25),
                legal_fields=("Doanh nghiệp",),
            )
        )
    )

    assert sql.startswith("SELECT documents.external_id")
    assert "documents.legal_field IN ('Doanh nghiệp')" in sql
    assert "ORDER BY documents.external_id" in sql


def test_filter_rejects_empty_or_unknown_statuses():
    with pytest.raises(ValueError, match="must not be empty"):
        LegalMetadataFilter(as_of=date.today(), statuses=())

    with pytest.raises(ValueError, match="Unknown legal statuses"):
        LegalMetadataFilter(as_of=date.today(), statuses=("ACTIVE",))


def test_hydration_query_loads_parent_and_enforces_temporal_validity():
    statement = MetadataQueries.hydrate_qdrant_points(
        ["point-1", "point-2"],
        LegalMetadataFilter(as_of=date(2026, 8, 25)),
        collection="legal-v2",
    )

    sql = _sql(statement)

    assert "LEFT OUTER JOIN chunks AS parent_chunk" in sql
    assert "document_versions.is_current IS true" in sql
    assert "effective_status.valid_period @> '2026-08-25'" in sql
    assert "chunks.qdrant_collection = 'legal-v2'" in sql
