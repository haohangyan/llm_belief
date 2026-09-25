"""Build a SQLite database from gene curation JSONL results."""

import argparse
import json
import sqlite3
from pathlib import Path


TABLE = "llm_evidence_curation"
INSERT_SQL = f"""
INSERT OR REPLACE INTO {TABLE} (
    statement_hash,
    source_hash,
    statement,
    evidence_text,
    judgment,
    error_category,
    explanation
) VALUES (?, ?, ?, ?, ?, ?, ?)
"""


def create_table(connection):
    connection.execute(f"""
        CREATE TABLE {TABLE} (
            statement_hash INTEGER NOT NULL,
            source_hash INTEGER NOT NULL,
            statement TEXT NOT NULL,
            evidence_text TEXT NOT NULL,
            judgment TEXT NOT NULL CHECK (
                judgment IN ('correct', 'incorrect', 'uncertain')
            ),
            error_category TEXT,
            explanation TEXT NOT NULL,
            PRIMARY KEY (statement_hash, source_hash)
        ) WITHOUT ROWID
    """)


def database_row(result, path, line_number):
    required = (
        "statement_hash",
        "source_hash",
        "statement",
        "evidence_text",
        "judgment",
    )
    missing = [key for key in required if key not in result]
    if missing:
        raise ValueError(
            f"{path}:{line_number}: missing fields: {', '.join(missing)}"
        )

    judgment = result["judgment"]
    if judgment not in {"correct", "incorrect", "uncertain"}:
        raise ValueError(
            f"{path}:{line_number}: invalid judgment: {judgment!r}"
        )

    return (
        int(result["statement_hash"]),
        int(result["source_hash"]),
        result["statement"],
        result["evidence_text"],
        judgment,
        result.get("error_category"),
        result.get("explanation", ""),
    )


def insert_file(connection, path, batch_size=10_000):
    count = 0
    batch = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            batch.append(database_row(result, path, line_number))
            if len(batch) >= batch_size:
                connection.executemany(INSERT_SQL, batch)
                count += len(batch)
                batch.clear()

    if batch:
        connection.executemany(INSERT_SQL, batch)
        count += len(batch)
    connection.commit()
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Build a SQLite database from curation result JSONL files."
    )
    parser.add_argument("results_directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    results_directory = args.results_directory
    if not results_directory.is_dir():
        raise FileNotFoundError(results_directory)

    result_files = sorted(results_directory.glob("*.jsonl"))
    if not result_files:
        raise FileNotFoundError(f"No JSONL files found in {results_directory}")

    output = args.output or results_directory.with_suffix(".sqlite")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{output} already exists; use --overwrite to replace it"
            )
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)

    loaded = 0
    with sqlite3.connect(output) as connection:
        create_table(connection)
        for index, path in enumerate(result_files, 1):
            count = insert_file(connection, path)
            loaded += count
            print(
                f"[{index}/{len(result_files)}] {path.name}: "
                f"{count:,} rows ({loaded:,} total)",
                flush=True,
            )

        connection.execute(
            f"CREATE INDEX ix_{TABLE}_source_hash ON {TABLE} (source_hash)"
        )
        connection.commit()
        stored = connection.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]

    print(f"database={output}")
    print(f"loaded={loaded:,} stored={stored:,} duplicates={loaded - stored:,}")


if __name__ == "__main__":
    main()
