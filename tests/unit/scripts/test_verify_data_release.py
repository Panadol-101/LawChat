from scripts.verify_deployment import ReleaseAudit


def _audit(**overrides) -> ReleaseAudit:
    values = {
        "collection": "legal-v2",
        "alias": "legal-current",
        "alias_target": "legal-v2",
        "expected_postgres_points": 100,
        "marked_postgres_points": 100,
        "qdrant_points": 100,
        "expected_vector_size": 1024,
        "vector_size": 1024,
        "expected_model": "multilingual-model",
        "actual_model": "multilingual-model",
        "distance": "Cosine",
        "sampled_points": 10,
        "missing_sampled_points": (),
    }
    values.update(overrides)
    return ReleaseAudit(**values)


def test_release_audit_requires_matching_counts_alias_and_samples():
    assert _audit().valid
    assert not _audit(qdrant_points=99).valid
    assert not _audit(alias_target="old-v1").valid
    assert not _audit(missing_sampled_points=("point-1",)).valid
    assert not _audit(distance="Dot").valid
