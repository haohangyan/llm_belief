"""Build statement hash sets from the processed INDRA statement dump."""

import argparse
import csv
import gzip
import pickle
import sys
from pathlib import Path

from indra.statements import stmt_from_json
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


def statement_is_gene_only(statement, specific_only):
    """Match INDRA's gene-only filter without retaining a statement batch."""
    namespaces = ("HGNC", "UP") if specific_only else ("HGNC", "UP", "FPLX")
    for agent in statement.agent_list():
        if agent is None:
            continue
        if not any(agent.db_refs.get(namespace) for namespace in namespaces):
            return False
        if any(
            not any(
                condition.agent.db_refs.get(namespace)
                for namespace in namespaces
            )
            for condition in agent.bound_conditions
        ):
            return False
    return True


def build_statement_hashes(
    processed_statements=PROCESSED_STATEMENTS_PATH,
    filters=tuple(FILTER_FILENAMES),
    output_directory=None,
):
    """Scan the processed statements once and write selected hash sets."""
    processed_statements = Path(processed_statements)
    output_directory = Path(output_directory or processed_statements.parent)
    filters = tuple(filters)
    unknown = set(filters) - set(FILTER_FILENAMES)
    if unknown:
        raise ValueError(f"Unknown filters: {', '.join(sorted(unknown))}")

    hash_sets = {name: set() for name in filters}

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
        for row_number, (statement_hash, statement_json) in enumerate(rows, 1):
            statement_hash = int(statement_hash)
            statement = stmt_from_json(clean_json_loads(statement_json))
            gene_only = (
                statement_is_gene_only(statement, specific_only=True)
                if "gene" in hash_sets or "famplex" in hash_sets
                else False
            )
            if "gene" in hash_sets and gene_only:
                hash_sets["gene"].add(statement_hash)
            if (
                "famplex" in hash_sets
                and not gene_only
                and statement_is_gene_only(statement, specific_only=False)
            ):
                hash_sets["famplex"].add(statement_hash)
            if (
                "chemical-gene" in hash_sets
                and statement_has_chemical_gene(statement)
            ):
                hash_sets["chemical-gene"].add(statement_hash)
            if row_number % 100_000 == 0:
                rows.set_postfix(
                    **{
                        name: f"{len(values):,}"
                        for name, values in hash_sets.items()
                    }
                )

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
    args = parser.parse_args()
    build_statement_hashes(
        args.processed_statements,
        args.filters,
        args.output_directory,
    )


if __name__ == "__main__":
    main()
