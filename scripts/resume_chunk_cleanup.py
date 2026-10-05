from __future__ import annotations

import argparse
import json

from database import DatabaseSettings, create_db_engine
from ingestion import PostgresMetadataLoader


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume verified stale-chunk cleanup for an interrupted run."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-generation-rows", required=True, type=int)
    args = parser.parse_args()

    engine = create_db_engine(DatabaseSettings.from_env())
    try:
        report = PostgresMetadataLoader(engine).resume_chunk_cleanup(
            run_id=args.run_id,
            expected_generation_rows=args.expected_generation_rows,
        )
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
