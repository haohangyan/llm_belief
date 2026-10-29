"""Build partial-ingestion files for LLM evidence verification results."""

import argparse
import csv
import gzip
import json
import os
import sqlite3
from collections import Counter
from pathlib import Path

from tqdm import tqdm


EVIDENCE_TABLE = "llm_evidence_curation"
STATEMENT_TABLE = "llm_statement_correctness"
PYSTOW_HOME = Path(os.environ.get("PYSTOW_HOME", Path.home() / ".data"))
DEFAULT_EVIDENCE_NODES = (
    PYSTOW_HOME
    / "indra"
    / "cogex"
    / "indra_db_evidence"
    / "nodes_Evidence.tsv.gz"
)
DEFAULT_STATEMENT_EDGES = (
    PYSTOW_HOME / "indra" / "cogex" / "database" / "edges.tsv.gz"
)


def _require_columns(header, required, path):
    missing = set(required) - set(header)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")


def _check_tables(connection, database, required_tables):
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    missing = set(required_tables) - tables
    if missing:
        raise ValueError(
            f"{database} is missing tables: {', '.join(sorted(missing))}"
        )


def _write_evidence_batch(connection, batch, writer, counts):
    connection.execute("DELETE FROM wanted_evidence")
    connection.executemany(
        "INSERT INTO wanted_evidence VALUES (?, ?, ?)", batch
    )
    judgments = dict(connection.execute(f"""
        SELECT wanted.evidence_id, curation.judgment
        FROM wanted_evidence AS wanted
        JOIN {EVIDENCE_TABLE} AS curation
          ON curation.statement_hash = wanted.statement_hash
         AND curation.source_hash = wanted.source_hash
    """))
    for evidence_id, _, _ in batch:
        judgment = judgments.get(evidence_id)
        counts[judgment or "unmatched"] += 1
        if judgment in {"correct", "incorrect"}:
            writer.writerow((evidence_id, "Evidence", judgment))


def write_evidence_updates(connection, source, output, batch_size):
    connection.execute("""
        CREATE TEMP TABLE wanted_evidence (
            evidence_id TEXT PRIMARY KEY,
            statement_hash INTEGER NOT NULL,
            source_hash INTEGER NOT NULL
        ) WITHOUT ROWID
    """)
    counts = Counter()
    missing_source_hash = 0
    batch = []
    with gzip.open(source, "rt") as source_file, gzip.open(output, "wt") as out_file:
        reader = csv.reader(source_file, delimiter="\t")
        header = next(reader)
        _require_columns(
            header, ("id:ID", "stmt_hash:int", "evidence:string"), source
        )
        id_index = header.index("id:ID")
        statement_index = header.index("stmt_hash:int")
        evidence_index = header.index("evidence:string")
        writer = csv.writer(out_file, delimiter="\t")
        writer.writerow(("id:ID", ":LABEL", "llm_verification:string"))

        for row in tqdm(reader, desc="Matching Evidence nodes", unit="node"):
            evidence = json.loads(row[evidence_index])
            source_hash = evidence.get("source_hash")
            if source_hash is None:
                missing_source_hash += 1
                continue
            batch.append(
                (
                    row[id_index],
                    int(row[statement_index]),
                    int(source_hash),
                )
            )
            if len(batch) >= batch_size:
                _write_evidence_batch(connection, batch, writer, counts)
                batch.clear()
        if batch:
            _write_evidence_batch(connection, batch, writer, counts)

    matched = counts["correct"] + counts["incorrect"]
    if not matched:
        raise RuntimeError(
            "No Evidence nodes matched the LLM database; make sure the Evidence "
            "file came from the same CoGEx build."
        )
    return counts, missing_source_hash


def _write_statement_batch(connection, batch, writer, counts):
    connection.execute("DELETE FROM wanted_statement")
    connection.executemany(
        "INSERT OR IGNORE INTO wanted_statement VALUES (?)",
        ((row[3],) for row in batch),
    )
    correct_counts = dict(connection.execute(f"""
        SELECT wanted.statement_hash, correctness.correct_count
        FROM wanted_statement AS wanted
        JOIN {STATEMENT_TABLE} AS correctness
          ON correctness.statement_hash = wanted.statement_hash
    """))
    for start_id, end_id, relation_type, statement_hash in batch:
        correct_count = correct_counts.get(statement_hash)
        if correct_count is None:
            counts["unmatched"] += 1
            continue
        writer.writerow(
            (
                start_id,
                end_id,
                relation_type,
                statement_hash,
                correct_count,
            )
        )
        counts["matched"] += 1


def write_statement_updates(connection, source, output, batch_size):
    connection.execute("""
        CREATE TEMP TABLE wanted_statement (
            statement_hash INTEGER PRIMARY KEY
        ) WITHOUT ROWID
    """)
    counts = Counter()
    batch = []
    with gzip.open(source, "rt") as source_file, gzip.open(output, "wt") as out_file:
        reader = csv.reader(source_file, delimiter="\t")
        header = next(reader)
        _require_columns(
            header, (":START_ID", ":END_ID", ":TYPE", "stmt_hash:int"), source
        )
        start_index = header.index(":START_ID")
        end_index = header.index(":END_ID")
        type_index = header.index(":TYPE")
        statement_index = header.index("stmt_hash:int")
        writer = csv.writer(out_file, delimiter="\t")
        writer.writerow(
            (
                ":START_ID",
                ":END_ID",
                ":TYPE",
                "stmt_hash:int",
                "llm_correct_evidence_count:int",
            )
        )

        for row in tqdm(reader, desc="Matching statement edges", unit="edge"):
            batch.append(
                (
                    row[start_index],
                    row[end_index],
                    row[type_index],
                    int(row[statement_index]),
                )
            )
            if len(batch) >= batch_size:
                _write_statement_batch(connection, batch, writer, counts)
                batch.clear()
        if batch:
            _write_statement_batch(connection, batch, writer, counts)

    if not counts["matched"]:
        raise RuntimeError(
            "No statement edges matched the LLM database; make sure the edge "
            "file came from the same CoGEx build."
        )
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Build CoGEx partial-ingestion files from LLM verification data."
    )
    parser.add_argument("database", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--evidence-nodes", type=Path, default=DEFAULT_EVIDENCE_NODES
    )
    parser.add_argument(
        "--statement-edges", type=Path, default=DEFAULT_STATEMENT_EDGES
    )
    parser.add_argument(
        "--statements-only",
        action="store_true",
        help="Only build the statement edge update file.",
    )
    parser.add_argument("--batch-size", type=int, default=10_000)
    args = parser.parse_args()

    input_paths = [args.database, args.statement_edges]
    if not args.statements_only:
        input_paths.append(args.evidence_nodes)
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    args.output_directory.mkdir(parents=True, exist_ok=True)
    statement_output = (
        args.output_directory / "edges_indra_rel_llm_verification.tsv.gz"
    )

    uri = f"file:{args.database.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        required_tables = {STATEMENT_TABLE}
        if not args.statements_only:
            required_tables.add(EVIDENCE_TABLE)
        _check_tables(connection, args.database, required_tables)
        if not args.statements_only:
            evidence_output = (
                args.output_directory / "nodes_Evidence_llm_verification.tsv.gz"
            )
            evidence_counts, missing_source_hash = write_evidence_updates(
                connection,
                args.evidence_nodes,
                evidence_output,
                args.batch_size,
            )
        statement_counts = write_statement_updates(
            connection,
            args.statement_edges,
            statement_output,
            args.batch_size,
        )

    if not args.statements_only:
        print(f"evidence_output={evidence_output}")
        print(
            f"evidence_correct={evidence_counts['correct']:,} "
            f"evidence_incorrect={evidence_counts['incorrect']:,} "
            f"evidence_uncertain={evidence_counts['uncertain']:,} "
            f"evidence_unmatched={evidence_counts['unmatched']:,} "
            f"evidence_missing_source_hash={missing_source_hash:,}"
        )
    print(f"statement_output={statement_output}")
    print(
        f"statement_edges_matched={statement_counts['matched']:,} "
        f"statement_edges_unmatched={statement_counts['unmatched']:,}"
    )


if __name__ == "__main__":
    main()
