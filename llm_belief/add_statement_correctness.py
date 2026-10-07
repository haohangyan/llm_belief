"""Add statement correctness scores from JSONL results to a database."""

import argparse
import json
import sqlite3
import time
from pathlib import Path

from tqdm import tqdm

from llm_belief.build_curation_database import STATEMENT_TABLE, TABLE


COUNT_SHIFT = 32
COUNT_MASK = (1 << COUNT_SHIFT) - 1


def load_counts(result_files):
    counts = {}
    evidence_count = 0
    total_bytes = sum(path.stat().st_size for path in result_files)

    with tqdm(
        total=total_bytes,
        desc="Reading results",
        unit="B",
        unit_scale=True,
    ) as progress:
        for path in result_files:
            with path.open("rb") as file:
                for line_number, line in enumerate(file, 1):
                    progress.update(len(line))
                    try:
                        result = json.loads(line)
                        statement_hash = int(result["statement_hash"])
                        judgment = result["judgment"]
                    except (
                        json.JSONDecodeError,
                        KeyError,
                        TypeError,
                        ValueError,
                    ) as error:
                        raise ValueError(
                            f"{path}:{line_number}: {error}"
                        ) from error
                    if judgment not in {"correct", "incorrect", "uncertain"}:
                        raise ValueError(
                            f"{path}:{line_number}: "
                            f"invalid judgment {judgment!r}"
                        )

                    counts[statement_hash] = counts.get(statement_hash, 0) + (
                        (1 << COUNT_SHIFT) + (judgment == "correct")
                    )
                    evidence_count += 1
                    if evidence_count % 100_000 == 0:
                        progress.set_postfix(
                            evidence=f"{evidence_count:,}",
                            statements=f"{len(counts):,}",
                        )

    return counts, evidence_count


def write_scores(database, counts, batch_size=10_000):
    insert_sql = f"INSERT INTO {STATEMENT_TABLE} VALUES (?, ?, ?, ?)"
    with sqlite3.connect(database) as connection:
        evidence_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (TABLE,),
        ).fetchone()
        if not evidence_table:
            raise ValueError(f"{database} does not contain {TABLE}")

        connection.execute(f"DROP TABLE IF EXISTS {STATEMENT_TABLE}")
        connection.execute(f"""
            CREATE TABLE {STATEMENT_TABLE} (
                statement_hash INTEGER PRIMARY KEY,
                correct_count INTEGER NOT NULL,
                total_count INTEGER NOT NULL,
                correctness_percent REAL NOT NULL
            ) WITHOUT ROWID
        """)

        batch = []
        hashes = sorted(counts)
        for statement_hash in tqdm(hashes, desc="Writing scores", unit="stmt"):
            packed = counts[statement_hash]
            correct_count = packed & COUNT_MASK
            total_count = packed >> COUNT_SHIFT
            batch.append(
                (
                    statement_hash,
                    correct_count,
                    total_count,
                    round(100.0 * correct_count / total_count, 2),
                )
            )
            if len(batch) >= batch_size:
                connection.executemany(insert_sql, batch)
                batch.clear()
        if batch:
            connection.executemany(insert_sql, batch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_directory", type=Path)
    parser.add_argument("database", type=Path)
    args = parser.parse_args()

    if not args.results_directory.is_dir():
        raise FileNotFoundError(args.results_directory)
    if not args.database.is_file():
        raise FileNotFoundError(args.database)
    result_files = sorted(args.results_directory.glob("results_*.jsonl"))
    if not result_files:
        raise FileNotFoundError(
            f"No results_*.jsonl files found in {args.results_directory}"
        )

    started = time.perf_counter()
    counts, evidence_count = load_counts(result_files)
    write_scores(args.database, counts)

    print(f"database={args.database}")
    print(f"evidences={evidence_count:,}")
    print(f"statements={len(counts):,}")
    print(f"elapsed={time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
