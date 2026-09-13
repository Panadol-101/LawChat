from datetime import date

from lawchat.database import LegalStatus, RelationshipType
from lawchat.ingestion.postgres_loader import (
    derive_effective_periods,
    normalize_date,
    normalize_legal_status,
    normalize_relationship_type,
)


def test_normalize_date_supports_source_and_iso_formats():
    assert normalize_date("03/02/1997") == date(1997, 2, 3)
    assert normalize_date("2026-01-31") == date(2026, 1, 31)
    assert normalize_date("31-01-2026") == date(2026, 1, 31)
    assert normalize_date("31/02/2026") is None
    assert normalize_date(None) is None


def test_source_status_mapping_is_conservative():
    assert normalize_legal_status("Còn hiệu lực") == LegalStatus.EFFECTIVE.value
    assert (
        normalize_legal_status("Hết hiệu lực toàn bộ")
        == LegalStatus.EXPIRED.value
    )
    assert (
        normalize_legal_status("Không còn phù hợp")
        == LegalStatus.UNKNOWN.value
    )
    assert normalize_legal_status("Nhãn mới") == LegalStatus.UNKNOWN.value


def test_expired_document_gets_non_overlapping_effective_history():
    periods = derive_effective_periods(
        source_status="Hết hiệu lực toàn bộ",
        issued_date=date(2014, 12, 5),
        effective_date=date(2014, 12, 15),
        expiry_date=date(2017, 1, 1),
    )

    assert [(item.status, item.valid_from, item.valid_to) for item in periods] == [
        (LegalStatus.EFFECTIVE.value, date(2014, 12, 15), date(2017, 1, 1)),
        (LegalStatus.EXPIRED.value, date(2017, 1, 1), None),
    ]


def test_status_without_any_start_date_is_not_invented():
    assert derive_effective_periods(
        source_status="Còn hiệu lực",
        issued_date=None,
        effective_date=None,
        expiry_date=None,
    ) == []


def test_expired_status_without_transition_date_is_conservative():
    periods = derive_effective_periods(
        source_status="Hết hiệu lực toàn bộ",
        issued_date=date(2014, 1, 1),
        effective_date=date(2014, 2, 1),
        expiry_date=None,
    )

    assert len(periods) == 1
    assert periods[0].status == LegalStatus.UNKNOWN.value
    assert periods[0].reason == "expiry_transition_date_missing"


def test_relationship_mapping_preserves_only_confident_semantics():
    assert normalize_relationship_type("Thay thế") == RelationshipType.REPLACES.value
    assert normalize_relationship_type("Căn cứ") == RelationshipType.CITES.value
    assert (
        normalize_relationship_type("Văn bản liên quan khác")
        == RelationshipType.RELATED_TO.value
    )
