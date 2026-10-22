"""Build hashes for statements newly admitted when FamPlex is allowed."""

import argparse
import csv
import gzip
import pickle
import sys
from pathlib import Path

from indra.statements import stmt_from_json
from indra.tools import assemble_corpus as ac
from indra_db.readonly_dumping.util import clean_json_loads
from tqdm import tqdm


DATA_DIRECTORY = Path("/scratch/h.yan/data")
PROCESSED_STATEMENTS_PATH = DATA_DIRECTORY / "processed_statements.tsv.gz"
GENE_HASHES_PATH = DATA_DIRECTORY / "gene_stmt_hashes.pkl"
FAMPLEX_HASHES_PATH = DATA_DIRECTORY / "famplex_stmt_hashes.pkl"


def build_famplex_hashes(
    processed_statements=PROCESSED_STATEMENTS_PATH,
    gene_hashes_path=GENE_HASHES_PATH,
    output_path=FAMPLEX_HASHES_PATH,
    batch_size=100_000,
):
    """Save statements added by allowing families to a hash-set pickle."""
    with gene_hashes_path.open("rb") as file:
        gene_hashes = {int(value) for value in pickle.load(file)}

    famplex_hashes = set()
    batch_hashes = []
    batch_statements = []

    def filter_batch():
        filtered = ac.filter_genes_only(
            batch_statements,
            specific_only=False,
        )
        filtered_ids = {id(statement) for statement in filtered}
        famplex_hashes.update(
            statement_hash
            for statement_hash, statement in zip(
                batch_hashes, batch_statements
            )
            if statement_hash not in gene_hashes
            and id(statement) in filtered_ids
        )
        batch_hashes.clear()
        batch_statements.clear()

    csv.field_size_limit(sys.maxsize)
    with gzip.open(
        processed_statements, "rt", encoding="utf-8", newline=""
    ) as file:
        for statement_hash, statement_json in tqdm(
            csv.reader(file, delimiter="\t"),
            desc="Finding FamPlex statements",
            unit="stmt",
            unit_scale=True,
        ):
            batch_hashes.append(int(statement_hash))
            batch_statements.append(
                stmt_from_json(clean_json_loads(statement_json))
            )
            if len(batch_statements) >= batch_size:
                filter_batch()

    if batch_statements:
        filter_batch()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(famplex_hashes, file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"gene_hashes={len(gene_hashes):,}")
    print(f"famplex_hashes={len(famplex_hashes):,}")
    print(f"output={output_path}")
    return famplex_hashes


def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--output",
        type=Path,
        default=FAMPLEX_HASHES_PATH,
    )
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args()
    build_famplex_hashes(
        args.processed_statements,
        args.gene_hashes,
        args.output,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
