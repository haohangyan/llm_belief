"""Reprocess evidence affected by site and processed-product context."""

import argparse
import csv
import gzip
import pickle
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

from indra.statements import Phosphorylation, stmt_from_json
from indra_db.readonly_dumping.util import clean_json_loads
from tqdm import tqdm

from llm_belief.build_curation_database import TABLE
from llm_belief.context import (
    _candidate_genes,
    load_processed_product_genes,
)
from llm_belief.curation import get_site_context
from llm_belief.locations import PMCID_TO_PMID_PATH
from llm_belief.process_gene_statements import (
    PROCESSED_STATEMENTS_PATH,
    evidence_pmid,
    run,
)


PRODUCT_ERROR_CATEGORIES = {"entity_boundaries", "grounding", "no_relation"}
OUTPUT_PREFIX = "results_zz_context_updates"


def load_candidate_hashes(database):
    """Load statement hashes whose old errors could change."""
    print("[1/3] scanning old SQLite judgments", flush=True)
    uri = f"file:{Path(database)}?mode=ro"
    site_hashes = set()
    product_hashes = set()
    categories = PRODUCT_ERROR_CATEGORIES | {"mod_site"}
    placeholders = ",".join("?" for _ in categories)
    with (
        sqlite3.connect(uri, uri=True) as connection,
        tqdm(desc="Scanning SQLite", unit="step") as progress,
    ):
        connection.set_progress_handler(lambda: progress.update() or 0, 1_000_000)
        try:
            rows = connection.execute(
                f"""
                SELECT DISTINCT statement_hash, error_category
                FROM {TABLE}
                WHERE judgment = 'incorrect'
                  AND error_category IN ({placeholders})
                """,
                tuple(sorted(categories)),
            )
            for statement_hash, error_category in rows:
                if error_category == "mod_site":
                    site_hashes.add(statement_hash)
                else:
                    product_hashes.add(statement_hash)
        finally:
            connection.set_progress_handler(None, 0)
    print(f"[select] site_candidate_statements={len(site_hashes):,}")
    print(f"[select] product_candidate_statements={len(product_hashes):,}")
    return site_hashes, product_hashes


def load_entries(database, processed_statements, context_batch_size=1_000):
    """Find only old incorrect evidence affected by the two context changes."""
    site_hashes, product_hashes = load_candidate_hashes(database)
    candidate_hashes = site_hashes | product_hashes
    print("[2/3] scanning processed statements and UniProt context", flush=True)

    if not PMCID_TO_PMID_PATH.is_file():
        raise FileNotFoundError(
            f"{PMCID_TO_PMID_PATH}; run python -m llm_belief.pmcid_map first"
        )
    with PMCID_TO_PMID_PATH.open("rb") as file:
        pmcid_to_pmid = pickle.load(file)

    entries_by_pmid = defaultdict(list)
    selected_keys = set()
    candidate_genes = set()
    processed_product_genes = set()
    site_evidence_count = 0
    product_evidence_count = 0
    matched_statements = 0
    batch = []

    uri = f"file:{Path(database)}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)

    def process_batch():
        nonlocal site_evidence_count, product_evidence_count
        product_statements = [
            statement
            for statement_hash, statement in batch
            if statement_hash in product_hashes
        ]
        for statement in product_statements:
            candidate_genes.update(_candidate_genes(statement))
        products_by_hash = load_processed_product_genes(product_statements)
        processed_product_genes.update(
            gene
            for products in products_by_hash.values()
            for gene in products
        )

        for statement_hash, statement in batch:
            old_errors = dict(connection.execute(
                f"""
                SELECT source_hash, error_category
                FROM {TABLE}
                WHERE statement_hash = ? AND judgment = 'incorrect'
                """,
                (statement_hash,),
            ))
            has_products = bool(products_by_hash.get(statement_hash))
            for evidence in statement.evidence:
                if not evidence.text:
                    continue
                source_hash = evidence.get_source_hash()
                error_category = old_errors.get(source_hash)
                site_context = get_site_context(statement, evidence)
                selected_for_site = (
                    statement_hash in site_hashes
                    and error_category == "mod_site"
                    and isinstance(statement, Phosphorylation)
                    and site_context is not None
                )
                selected_for_product = (
                    has_products
                    and error_category in PRODUCT_ERROR_CATEGORIES
                )
                if not selected_for_site and not selected_for_product:
                    continue

                key = (statement_hash, source_hash)
                if key in selected_keys:
                    continue
                selected_keys.add(key)
                site_evidence_count += int(selected_for_site)
                product_evidence_count += int(selected_for_product)
                pmid = evidence_pmid(evidence, pmcid_to_pmid)
                entries_by_pmid[pmid].append({
                    "stmt_hash": statement_hash,
                    "source_hash": source_hash,
                    "statement": statement,
                    "evidence_text": evidence.text,
                    "site_context": site_context,
                    "pmid": pmid,
                })
        batch.clear()

    csv.field_size_limit(sys.maxsize)
    with gzip.open(
        processed_statements, "rt", encoding="utf-8", newline=""
    ) as file:
        rows = tqdm(
            csv.reader(file, delimiter="\t"),
            desc="Finding context updates",
            unit="stmt",
            unit_scale=True,
        )
        for row_number, row in enumerate(rows, 1):
            if len(row) != 2:
                raise ValueError(
                    f"{processed_statements}:{row_number}: expected two TSV columns"
                )
            statement_hash = int(row[0])
            if statement_hash not in candidate_hashes:
                continue
            statement = stmt_from_json(clean_json_loads(row[1]))
            batch.append((statement_hash, statement))
            matched_statements += 1
            if len(batch) >= context_batch_size:
                process_batch()
    if batch:
        process_batch()
    connection.close()

    product_percent = (
        100 * len(processed_product_genes) / len(candidate_genes)
        if candidate_genes else 0
    )
    print(f"[select] matched_statements={matched_statements:,}")
    print(f"[select] candidate_genes={len(candidate_genes):,}")
    print(
        f"[select] processed_product_genes={len(processed_product_genes):,} "
        f"({product_percent:.2f}%)"
    )
    print(f"[select] site_evidence={site_evidence_count:,}")
    print(f"[select] processed_product_evidence={product_evidence_count:,}")
    print(f"[select] unique_evidence={len(selected_keys):,}")
    return entries_by_pmid, len(selected_keys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("results_directory", type=Path)
    parser.add_argument(
        "--processed-statements",
        type=Path,
        default=PROCESSED_STATEMENTS_PATH,
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=5_000)
    parser.add_argument("--context-batch-size", type=int, default=1_000)
    parser.add_argument("--server", action="append")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.database.is_file():
        raise FileNotFoundError(args.database)
    if not args.processed_statements.is_file():
        raise FileNotFoundError(args.processed_statements)
    entries_by_pmid, evidence_count = load_entries(
        args.database,
        args.processed_statements,
        args.context_batch_size,
    )
    if args.dry_run:
        print("[3/3] dry run complete; no model calls or files written")
        return
    if not evidence_count:
        print("[done] no affected evidence found")
        return

    args.results_directory.mkdir(parents=True, exist_ok=True)
    existing_outputs = sorted(
        args.results_directory.glob(f"{OUTPUT_PREFIX}_*.jsonl")
    )
    if existing_outputs and not args.overwrite:
        raise FileExistsError(
            f"{existing_outputs[0]} already exists; use --overwrite to replace it"
        )
    for output in existing_outputs:
        output.unlink()

    print("[3/3] sending selected evidence to the model", flush=True)
    run(
        entries_by_pmid,
        evidence_count,
        args.workers,
        args.chunk_size,
        args.server or ["http://127.0.0.1:8000/v1"],
        include_famplex_context=False,
        run_directory=args.results_directory,
        output_prefix=OUTPUT_PREFIX,
    )


if __name__ == "__main__":
    main()
