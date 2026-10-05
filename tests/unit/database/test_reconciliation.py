from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, call

import pytest
from qdrant_client.http import models as qmodels

from database.reconciliation import (
    ReconciliationReport,
    StatusReconciler,
)


def test_reconciliation_report_to_dict():
    report = ReconciliationReport(
        as_of="2026-09-26",
        dry_run=True,
        total_reconciled=5,
        effective_to_expired=3,
        partially_effective_to_expired=2,
        sample_documents=[{"external_id": "doc-1", "reason": "expired"}],
    )
    data = report.to_dict()
    assert data["as_of"] == "2026-09-26"
    assert data["dry_run"] is True
    assert data["total_reconciled"] == 5
    assert data["effective_to_expired"] == 3
    assert data["partially_effective_to_expired"] == 2
    assert len(data["sample_documents"]) == 1


def test_status_reconciler_dry_run_does_not_mutate():
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn

    # Mock audit targets query return
    mock_conn.execute.return_value.mappings.return_value.all.side_effect = [
        [{"old_status": "EFFECTIVE", "count": 10}, {"old_status": "PARTIALLY_EFFECTIVE", "count": 5}],
        [{"external_id": "doc-1", "document_number": "123", "old_status": "EFFECTIVE", "repeal_date": "2024-01-01", "reason_summary": "test"}],
    ]
    mock_conn.execute.return_value.fetchall.return_value = [("doc-1",)]

    reconciler = StatusReconciler(mock_engine)
    report = reconciler.reconcile(as_of=date(2026, 1, 1), dry_run=True, sync_qdrant=False)

    assert report.total_reconciled == 15
    assert report.effective_to_expired == 10
    assert report.partially_effective_to_expired == 5
    assert report.dry_run is True
    assert report.effective_status_deleted == 0
    assert report.effective_status_inserted == 0
    assert report.qdrant_points_updated == 0


def test_status_reconciler_syncs_qdrant_payload():
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn

    # 3 docs to update
    mock_conn.execute.return_value.mappings.return_value.all.side_effect = [
        [{"old_status": "EFFECTIVE", "count": 3}],
        [],
    ]
    mock_conn.execute.return_value.fetchall.return_value = [("doc-1",), ("doc-2",), ("doc-3",)]

    # Mock Qdrant client
    mock_qdrant = MagicMock()
    mock_qdrant.collection_exists.return_value = True

    reconciler = StatusReconciler(
        mock_engine,
        qdrant_client=mock_qdrant,
        qdrant_collection="legal_collection_test",
    )

    report = reconciler.reconcile(
        as_of=date(2026, 1, 1),
        dry_run=False,
        sync_qdrant=True,
        qdrant_batch_size=2,
    )

    assert report.total_reconciled == 3
    assert report.qdrant_points_updated == 3

    # Check set_payload was called twice (batch 1: 2 items, batch 2: 1 item)
    assert mock_qdrant.set_payload.call_count == 2
    args_batch1, kwargs_batch1 = mock_qdrant.set_payload.call_args_list[0]
    assert kwargs_batch1["collection_name"] == "legal_collection_test"
    assert kwargs_batch1["payload"] == {"status": "EXPIRED"}
    assert kwargs_batch1["wait"] is False
    filter_obj = kwargs_batch1["points"]
    assert isinstance(filter_obj, qmodels.Filter)
    assert filter_obj.must[0].key == "doc_id"
    assert filter_obj.must[0].match.any == ["doc-1", "doc-2"]


def test_status_reconciler_handles_qdrant_missing_collection():
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_conn.execute.return_value.mappings.return_value.all.side_effect = [
        [{"old_status": "EFFECTIVE", "count": 1}],
        [],
    ]
    mock_conn.execute.return_value.fetchall.return_value = [("doc-1",)]

    mock_qdrant = MagicMock()
    mock_qdrant.collection_exists.return_value = False

    reconciler = StatusReconciler(mock_engine, qdrant_client=mock_qdrant)
    report = reconciler.reconcile(dry_run=False, sync_qdrant=True)

    assert report.qdrant_points_updated == 0
    mock_qdrant.set_payload.assert_not_called()


def test_sync_all_reconciled_to_qdrant():
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_conn.execute.return_value.fetchall.return_value = [("doc-100",), ("doc-101",)]

    mock_qdrant = MagicMock()
    mock_qdrant.collection_exists.return_value = True

    reconciler = StatusReconciler(mock_engine, qdrant_client=mock_qdrant)
    updated = reconciler.sync_all_reconciled_to_qdrant(batch_size=50)

    assert updated == 2
    mock_qdrant.set_payload.assert_called_once()
    _, kwargs = mock_qdrant.set_payload.call_args
    assert kwargs["payload"] == {"status": "EXPIRED"}
    assert kwargs["wait"] is False
    assert kwargs["points"].must[0].match.any == ["doc-100", "doc-101"]
