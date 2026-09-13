from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
from sqlalchemy import text

from lawchat.database import DatabaseSettings, create_db_engine
from lawchat.indexing import QdrantSettings, SentenceTransformerEmbedder
from scripts import dispatch


@dataclass(frozen=True, slots=True)
class ReleaseAudit:
    collection: str
    alias: str
    alias_target: str | None
    expected_postgres_points: int
    marked_postgres_points: int
    qdrant_points: int
    expected_vector_size: int
    vector_size: int | None
    expected_model: str
    actual_model: str | None
    distance: str | None
    sampled_points: int
    missing_sampled_points: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return (
            self.alias_target == self.collection
            and self.expected_postgres_points > 0
            and self.expected_postgres_points == self.marked_postgres_points
            and self.expected_postgres_points == self.qdrant_points
            and self.vector_size == self.expected_vector_size
            and self.distance is not None
            and "COSINE" in self.distance.upper()
            and self.actual_model == self.expected_model
            and self.sampled_points > 0
            and not self.missing_sampled_points
        )


def release_main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that PostgreSQL and Qdrant form one complete release."
    )
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--expected-vector-size", type=int, default=1024)
    args = parser.parse_args()
    if args.sample_size <= 0:
        parser.error("--sample-size must be > 0")
    if args.expected_vector_size <= 0:
        parser.error("--expected-vector-size must be > 0")

    db_engine = create_db_engine(DatabaseSettings.from_env())
    settings = QdrantSettings.from_env()
    client = settings.create_client()
    try:
        with db_engine.connect() as connection:
            counts = connection.execute(
                text(
                    """
                    SELECT
                      count(*) FILTER (
                        WHERE v.is_current AND c.is_indexable
                          AND c.retrieval_text IS NOT NULL
                          AND c.retrieval_text <> ''
                      ) AS expected,
                      count(*) FILTER (
                        WHERE v.is_current AND c.is_indexable
                          AND c.retrieval_text IS NOT NULL
                          AND c.retrieval_text <> ''
                          AND c.qdrant_collection = :collection
                          AND c.qdrant_point_id IS NOT NULL
                      ) AS marked
                    FROM chunks c
                    JOIN document_versions v ON v.id = c.version_id
                    """
                ),
                {"collection": settings.collection},
            ).mappings().one()
            sample_ids = tuple(
                connection.execute(
                    text(
                        """
                        SELECT c.qdrant_point_id
                        FROM chunks c
                        JOIN document_versions v ON v.id = c.version_id
                        WHERE v.is_current AND c.is_indexable
                          AND c.retrieval_text IS NOT NULL
                          AND c.retrieval_text <> ''
                          AND c.qdrant_collection = :collection
                          AND c.qdrant_point_id IS NOT NULL
                        ORDER BY c.id
                        LIMIT :sample_size
                        """
                    ),
                    {
                        "collection": settings.collection,
                        "sample_size": args.sample_size,
                    },
                ).scalars()
            )

        alias_target = next(
            (
                item.collection_name
                for item in client.get_aliases().aliases
                if item.alias_name == settings.alias
            ),
            None,
        )
        info = client.get_collection(settings.collection)
        vectors = info.config.params.vectors
        metadata = info.config.metadata or {}
        found = client.retrieve(
            collection_name=settings.collection,
            ids=list(sample_ids),
            with_payload=False,
            with_vectors=False,
        ) if sample_ids else []
        found_ids = {str(point.id) for point in found}
        audit = ReleaseAudit(
            collection=settings.collection,
            alias=settings.alias,
            alias_target=alias_target,
            expected_postgres_points=int(counts["expected"]),
            marked_postgres_points=int(counts["marked"]),
            qdrant_points=int(
                client.count(settings.collection, exact=True).count
            ),
            expected_vector_size=args.expected_vector_size,
            vector_size=getattr(vectors, "size", None),
            expected_model=os.getenv(
                "EMBEDDING_MODEL",
                "BAAI/bge-m3",
            ),
            actual_model=metadata.get("embedding_model"),
            distance=str(getattr(vectors, "distance", None)),
            sampled_points=len(sample_ids),
            missing_sampled_points=tuple(
                point_id for point_id in sample_ids if point_id not in found_ids
            ),
        )
        print(json.dumps({**asdict(audit), "valid": audit.valid}, indent=2))
        if not audit.valid:
            raise SystemExit(1)
    finally:
        client.close()
        db_engine.dispose()


def embedding_main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the SentenceTransformers/PyTorch embedding runtime."
    )
    parser.add_argument("--device", default=os.getenv("EMBEDDING_DEVICE", "auto"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("HF_HUB_OFFLINE", "false").casefold()
        in {"1", "true", "yes", "on"},
    )
    args = parser.parse_args()

    embedder = SentenceTransformerEmbedder.from_env(
        device=args.device,
        batch_size=args.batch_size,
        local_files_only=args.local_files_only,
    )
    documents = embedder.encode_documents(
        [
            "Điều 1. Phạm vi điều chỉnh của văn bản.",
            "Người lao động có quyền đơn phương chấm dứt hợp đồng.",
        ]
    )
    queries = embedder.encode_queries(
        ["Khi nào người lao động được nghỉ việc?"]
    )
    report = {
        "model": embedder.model_name,
        "revision": embedder.revision,
        "device": embedder.resolved_device,
        "dimension": embedder.dimension,
        "max_length": embedder.max_length,
        "document_shape": list(documents.shape),
        "query_shape": list(queries.shape),
        "document_norms": np.linalg.norm(documents, axis=1).tolist(),
        "query_norms": np.linalg.norm(queries, axis=1).tolist(),
        "finite": bool(np.isfinite(documents).all() and np.isfinite(queries).all()),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if (
        documents.shape != (2, embedder.dimension)
        or queries.shape != (1, embedder.dimension)
        or not report["finite"]
        or not np.allclose(report["document_norms"], 1.0, atol=1e-5)
        or not np.allclose(report["query_norms"], 1.0, atol=1e-5)
    ):
        raise SystemExit(1)


def main() -> None:
    dispatch(
        "Verify a LawChat deployment.",
        {"release": release_main, "embedding": embedding_main},
    )


if __name__ == "__main__":
    main()
