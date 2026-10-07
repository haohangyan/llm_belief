"""Add statement-level correctness scores to a curation database."""

import argparse
import sqlite3
import time
from pathlib import Path

from llm_belief.build_curation_database import (
    STATEMENT_TABLE,
    TABLE,
    create_statement_correctness,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    args = parser.parse_args()

    if not args.database.is_file():
        raise FileNotFoundError(args.database)

    started = time.perf_counter()
    with sqlite3.connect(args.database) as connection:
        evidence_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (TABLE,),
        ).fetchone()
        if not evidence_table:
            raise ValueError(f"{args.database} does not contain {TABLE}")

        print("[score] calculating statement correctness", flush=True)
        connection.execute(f"DROP TABLE IF EXISTS {STATEMENT_TABLE}")
        create_statement_correctness(connection)
        statements = connection.execute(
            f"SELECT COUNT(*) FROM {STATEMENT_TABLE}"
        ).fetchone()[0]

    print(f"database={args.database}")
    print(f"statements={statements:,}")
    print(f"elapsed={time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
