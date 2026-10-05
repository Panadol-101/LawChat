from scripts.legal_metadata import PROVISION_AUDIT_SQL


def test_audit_counts_only_exact_reviewed_resolved_provisions():
    assert "p.status <> 'UNKNOWN'" in PROVISION_AUDIT_SQL
    assert "p.provision_key = concat(" in PROVISION_AUDIT_SQL
    assert "p.valid_from IS NOT NULL" in PROVISION_AUDIT_SQL
    assert "p.valid_period IS NOT NULL" in PROVISION_AUDIT_SQL
    assert "p.metadata->>'review_status'" in PROVISION_AUDIT_SQL
    assert "= 'VERIFIED'" in PROVISION_AUDIT_SQL
