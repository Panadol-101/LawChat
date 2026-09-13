import pytest

from lawchat.retrieval import (
    RRFSettings,
    RetrievalCandidate,
    reciprocal_rank_fusion,
)


def _candidate(point_id, source, rank, score=1.0):
    return RetrievalCandidate(
        point_id=point_id,
        chunk_id=f"chunk-{point_id}",
        source=source,
        raw_score=score,
        rank=rank,
    )


def test_rrf_rewards_candidates_returned_by_both_sources():
    fused = reciprocal_rank_fusion(
        {
            "dense": [
                _candidate("a", "dense", 1, 0.9),
                _candidate("b", "dense", 2, 0.8),
            ],
            "sparse": [
                _candidate("b", "sparse", 1, 12.0),
                _candidate("c", "sparse", 2, 10.0),
            ],
        },
        settings=RRFSettings(k=60),
    )

    assert [item.point_id for item in fused] == ["b", "a", "c"]
    assert fused[0].source_ranks == {"dense": 2, "sparse": 1}
    assert fused[0].source_scores == {"dense": 0.8, "sparse": 12.0}


def test_rrf_validates_settings_and_point_identity():
    with pytest.raises(ValueError, match="at least one"):
        RRFSettings(dense_weight=0, sparse_weight=0)
    with pytest.raises(ValueError, match="multiple chunk IDs"):
        reciprocal_rank_fusion(
            {
                "dense": [_candidate("same", "dense", 1)],
                "sparse": [
                    RetrievalCandidate(
                        point_id="same",
                        chunk_id="different",
                        source="sparse",
                        raw_score=1,
                        rank=1,
                    )
                ],
            }
        )
