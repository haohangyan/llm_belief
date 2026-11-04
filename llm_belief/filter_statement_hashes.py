"""Build statement hash sets from the processed INDRA statement dump."""

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
FILTER_FILENAMES = {
    "gene": "gene_stmt_hashes.pkl",
    "famplex": "famplex_stmt_hashes.pkl",
    "chemical-gene": "chemical_gene_stmt_hashes.pkl",
}


def statement_has_chemical_gene(statement):
    """Return whether direct agents include a chemical and gene/family."""
    namespaces = {
        agent.get_grounding()[0]
        for agent in statement.agent_list()
        if agent is not None
    }
    return "CHEBI" in namespaces and bool({"HGNC", "FPLX"} & namespaces)


def build_statement_hashes(
    processed_statements=PROCESSED_STATEMENTS_PATH,
    filters=tuple(FILTER_FILENAMES),
    output_directory=None,
    batch_size=100_000,
):
    """Scan the processed statements once and write selected hash sets."""
    processed_statements = Path(processed_statements)
    output_directory = Path(output_directory or processed_statements.parent)
    filters = tuple(filters)
    unknown = set(filters) - set(FILTER_FILENAMES)
    if unknown:
        raise ValueError(f"Unknown filters: {', '.join(sorted(unknown))}")

    hash_sets = {name: set() for name in filters}
    batch_hashes = []
    batch_statements = []

    def filter_batch():
        gene_ids = set()
        if "gene" in hash_sets or "famplex" in hash_sets:
            gene_ids = {
                id(statement)
                for statement in ac.filter_genes_only(
                    batch_statements, specific_only=True
                )
            }
        if "gene" in hash_sets:
            hash_sets["gene"].update(
                statement_hash
                for statement_hash, statement in zip(
                    batch_hashes, batch_statements
                )
                if id(statement) in gene_ids
            )
        if "famplex" in hash_sets:
            gene_or_family_ids = {
                id(statement)
                for statement in ac.filter_genes_only(
                    batch_statements, specific_only=False
                )
            }
            hash_sets["famplex"].update(
                statement_hash
                for statement_hash, statement in zip(
                    batch_hashes, batch_statements
                )
                if id(statement) in gene_or_family_ids
                and id(statement) not in gene_ids
            )
        if "chemical-gene" in hash_sets:
            hash_sets["chemical-gene"].update(
                statement_hash
                for statement_hash, statement in zip(
                    batch_hashes, batch_statements
                )
                if statement_has_chemical_gene(statement)
            )
        batch_hashes.clear()
        batch_statements.clear()

    csv.field_size_limit(sys.maxsize)
    with gzip.open(
        processed_statements, "rt", encoding="utf-8", newline=""
    ) as file:
        rows = tqdm(
            csv.reader(file, delimiter="\t"),
            desc="Filtering statements",
            unit="stmt",
            unit_scale=True,
        )
        for statement_hash, statement_json in rows:
            batch_hashes.append(int(statement_hash))
            batch_statements.append(
                stmt_from_json(clean_json_loads(statement_json))
            )
            if len(batch_statements) >= batch_size:
                filter_batch()
                rows.set_postfix(
                    **{
                        name: f"{len(values):,}"
                        for name, values in hash_sets.items()
                    }
                )

    if batch_statements:
        filter_batch()

    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, hashes in hash_sets.items():
        output = output_directory / FILTER_FILENAMES[name]
        temporary_output = output.with_suffix(output.suffix + ".tmp")
        with temporary_output.open("wb") as file:
            pickle.dump(hashes, file, protocol=pickle.HIGHEST_PROTOCOL)
        temporary_output.replace(output)
        outputs[name] = output
        print(f"{name}_hashes={len(hashes):,}")
        print(f"{name}_output={output}")
    return hash_sets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-statements",
        type=Path,
        default=PROCESSED_STATEMENTS_PATH,
    )
    parser.add_argument(
        "--filters",
        nargs="+",
        choices=tuple(FILTER_FILENAMES),
        default=tuple(FILTER_FILENAMES),
    )
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args()
    build_statement_hashes(
        args.processed_statements,
        args.filters,
        args.output_directory,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
