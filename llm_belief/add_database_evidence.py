"""Add textless database evidence to a gene curation result directory."""

import argparse
import csv
import gzip
import json
import pickle
import sys
from pathlib import Path

from indra.sources import SOURCE_INFO
from indra.statements import stmt_from_json
from indra_db.readonly_dumping.util import clean_json_loads
from tqdm import tqdm


DATA_DIRECTORY = Path("/scratch/h.yan/data")
GENE_HASHES_PATH = DATA_DIRECTORY / "gene_stmt_hashes.pkl"
PROCESSED_STATEMENTS_PATH = DATA_DIRECTORY / "processed_statements.tsv.gz"
OUTPUT_NAME = "results_database_evidence.jsonl"
HASH_MASK = (1 << 64) - 1


def evidence_key(statement_hash, source_hash):
    return (
        (int(statement_hash) & HASH_MASK) << 64
        | (int(source_hash) & HASH_MASK)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_directory", type=Path)
    parser.add_argument(
        "--processed-statements",
        type=Path,
        default=PROCESSED_STATEMENTS_PATH,
    )
    parser.add_argument(
        "--gene-hashes",
        type=Path,
        default=GENE_HASHES_PATH,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.results_directory.is_dir():
        raise FileNotFoundError(args.results_directory)
    if not args.processed_statements.is_file():
        raise FileNotFoundError(args.processed_statements)
    if not args.gene_hashes.is_file():
        raise FileNotFoundError(args.gene_hashes)

    output = args.results_directory / OUTPUT_NAME
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output} already exists; use --overwrite to replace it"
        )

    with args.gene_hashes.open("rb") as file:
        gene_hashes = {int(value) for value in pickle.load(file)}
    print(f"gene_hashes={len(gene_hashes):,}", flush=True)

    csv.field_size_limit(sys.maxsize)
    written = 0
    duplicate = 0
    seen = set()
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    with (
        gzip.open(
            args.processed_statements,
            "rt",
            encoding="utf-8",
            newline="",
        ) as input_file,
        temporary_output.open(
            "w", encoding="utf-8", buffering=1024 * 1024
        ) as output_file,
    ):
        rows = tqdm(
            csv.reader(input_file, delimiter="\t"),
            desc="Adding database evidence",
            unit="stmt",
            unit_scale=True,
        )
        for row_number, row in enumerate(rows, 1):
            if len(row) != 2:
                raise ValueError(
                    f"{args.processed_statements}:{row_number}: "
                    "expected two TSV columns"
                )
            statement_hash = int(row[0])
            if statement_hash not in gene_hashes:
                continue

            statement = stmt_from_json(clean_json_loads(row[1]))
            statement_text = None
            for evidence in statement.evidence:
                if evidence.text or SOURCE_INFO.get(
                    evidence.source_api, {}
                ).get("type") != "database":
                    continue

                source_hash = evidence.get_source_hash()
                key = evidence_key(statement_hash, source_hash)
                if key in seen:
                    duplicate += 1
                    continue
                seen.add(key)
                if statement_text is None:
                    statement_text = str(statement)

                result = {
                    "statement_hash": statement_hash,
                    "source_hash": source_hash,
                    "statement": statement_text,
                    "evidence_text": "",
                    "judgment": "correct",
                    "error_category": None,
                    "explanation": "",
                }
                output_file.write(
                    json.dumps(result, ensure_ascii=False) + "\n"
                )
                written += 1
                if written % 100_000 == 0:
                    rows.set_postfix(
                        written=f"{written:,}", duplicate=f"{duplicate:,}"
                    )

    temporary_output.replace(output)
    print(f"output={output}")
    print(f"written={written:,} duplicates={duplicate:,}")


if __name__ == "__main__":
    main()
