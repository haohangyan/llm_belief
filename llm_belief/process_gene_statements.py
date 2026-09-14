"""Curate every evidence of gene-only processed INDRA statements."""

import argparse
import csv
import gzip
import json
import pickle
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from indra.statements import stmt_from_json
from indra_db.readonly_dumping.util import clean_json_loads
from tqdm import tqdm

from llm_belief.abstract_cache import get_abstracts
from llm_belief.context import load_mesh_terms, load_uniprot_contexts
from llm_belief.curation import curate
from llm_belief.llm import LLMClient


DATA_DIRECTORY = Path("/scratch/h.yan/data")
GENE_HASHES_PATH = DATA_DIRECTORY / "gene_stmt_hashes.pkl"
PROCESSED_STATEMENTS_PATH = DATA_DIRECTORY / "processed_statements.tsv.gz"
RESULTS_PATH = DATA_DIRECTORY / "gene_curation_results.jsonl"

MODEL = "openai/gpt-oss-120b"
REASONING_EFFORT = "low"
MAX_TOKENS = 512


def evidence_pmid(evidence):
    pmid = evidence.pmid or evidence.text_refs.get("PMID")
    try:
        return int(pmid) if pmid else None
    except ValueError:
        return None


def load_gene_entries():
    with GENE_HASHES_PATH.open("rb") as file:
        gene_hashes = {int(value) for value in pickle.load(file)}

    if not PROCESSED_STATEMENTS_PATH.exists():
        raise FileNotFoundError(PROCESSED_STATEMENTS_PATH)

    csv.field_size_limit(sys.maxsize)
    entries_by_pmid = defaultdict(list)
    statements_loaded = 0
    evidence_count = 0
    with gzip.open(
        PROCESSED_STATEMENTS_PATH, "rt", encoding="utf-8", newline=""
    ) as file:
        rows = tqdm(
            csv.reader(file, delimiter="\t"),
            desc=f"Loading {PROCESSED_STATEMENTS_PATH.name}",
            unit="stmt",
            unit_scale=True,
        )
        for row_number, row in enumerate(rows, 1):
            if len(row) != 2:
                raise ValueError(
                    f"{PROCESSED_STATEMENTS_PATH}:{row_number}: "
                    "expected two TSV columns"
                )
            stmt_hash, stmt_json = int(row[0]), row[1]
            if stmt_hash not in gene_hashes:
                continue

            statement = stmt_from_json(clean_json_loads(stmt_json))
            statements_loaded += 1
            for evidence in statement.evidence:
                if evidence.text:
                    pmid = evidence_pmid(evidence)
                    entries_by_pmid[pmid].append(
                        {
                            "stmt_hash": stmt_hash,
                            "statement": statement,
                            "evidence_text": evidence.text,
                            "pmid": pmid,
                        }
                    )
                    evidence_count += 1

    print(
        f"[load] gene_statements={statements_loaded:,} "
        f"evidences={evidence_count:,}",
        flush=True,
    )

    return entries_by_pmid, evidence_count


def iter_chunks(entries_by_pmid, chunk_size):
    chunk = []
    for entries in entries_by_pmid.values():
        if chunk and len(chunk) + len(entries) > chunk_size:
            yield chunk
            chunk = []
        chunk.extend(entries)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def process_one(client, entry, abstracts, mesh_terms, uniprot_contexts):
    pmid = entry["pmid"]
    result = curate(
        client,
        entry["statement"],
        entry["evidence_text"],
        abstract=abstracts.get(pmid) if pmid else None,
        mesh_terms=mesh_terms.get(pmid, []) if pmid else [],
        uniprot_context=uniprot_contexts.get(entry["stmt_hash"]),
    )
    judgment = {
        "accepted": "correct",
        "rejected": "incorrect",
        "uncertain": "uncertain",
    }[result["decision"]]
    return {
        "statement_hash": entry["stmt_hash"],
        "statement": str(entry["statement"]),
        "evidence_text": entry["evidence_text"],
        "judgment": judgment,
        "explanation": result.get("reasoning", ""),
    }


def run(entries_by_pmid, evidence_count, workers, chunk_size):
    client = LLMClient(MODEL, MAX_TOKENS, REASONING_EFFORT)
    finished = 0
    failed = 0
    started = time.perf_counter()

    with (
        RESULTS_PATH.open(
            "w", encoding="utf-8", buffering=1024 * 1024
        ) as output_file,
        ThreadPoolExecutor(max_workers=workers) as pool,
        tqdm(total=evidence_count, desc="Curating", unit="evidence") as progress,
    ):
        for chunk in iter_chunks(entries_by_pmid, chunk_size):
            pmids = {entry["pmid"] for entry in chunk if entry["pmid"]}
            abstracts, _ = get_abstracts(pmids)
            mesh_terms = load_mesh_terms(pmids)

            statements = {
                entry["stmt_hash"]: entry["statement"] for entry in chunk
            }
            uniprot_contexts = load_uniprot_contexts(statements.values())

            futures = {
                pool.submit(
                    process_one,
                    client,
                    entry,
                    abstracts,
                    mesh_terms,
                    uniprot_contexts,
                ): entry
                for entry in chunk
            }
            for future in as_completed(futures):
                entry = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    failed += 1
                    print(
                        f"error {entry['stmt_hash']}: {error}",
                        flush=True,
                    )
                    continue
                output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                finished += 1

            output_file.flush()
            elapsed = time.perf_counter() - started
            progress.update(len(chunk))
            progress.set_postfix(
                finished=f"{finished:,}",
                failed=f"{failed:,}",
                rate=f"{finished / elapsed:.2f}/s",
            )

    print(f"[done] output={RESULTS_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=5_000)
    args = parser.parse_args()

    entries_by_pmid, evidence_count = load_gene_entries()
    print(
        f"[load] PMIDs={len(entries_by_pmid):,} "
        f"total evidences={evidence_count:,}",
        flush=True,
    )
    run(entries_by_pmid, evidence_count, args.workers, args.chunk_size)


if __name__ == "__main__":
    main()
