"""Benchmark the Gemma curation workflow against human curations."""

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

from llm_belief.abstract_cache import get_abstracts
from llm_belief.curation import curate
from llm_belief.context import (
    get_uniprot_context,
    load_mesh_terms,
)
from llm_belief.data import load_pickle_statements
from llm_belief.llm import LLMClient, OpenAILLMClient
from llm_belief.locations import CORPUS_PICKLE_PATH, CURATIONS_PATH


def load_gold(limit, seed):
    with CURATIONS_PATH.open() as file:
        curations = json.load(file)

    grouped = defaultdict(list)
    for row in curations:
        grouped[(int(row["pa_hash"]), int(row["source_hash"]))].append(row)

    pairs = sorted(grouped)
    if limit and limit < len(pairs):
        pairs = random.Random(seed).sample(pairs, limit)

    return {
        pair: "correct"
        if all(row["tag"] == "correct" for row in grouped[pair])
        else "incorrect"
        for pair in pairs
    }


def load_entries(gold):
    targets_by_statement = defaultdict(set)
    for matches_hash, source_hash in gold:
        targets_by_statement[matches_hash].add(source_hash)

    found = {}
    for stmt in load_pickle_statements(CORPUS_PICKLE_PATH):
        matches_hash = stmt.get_hash()
        target_sources = targets_by_statement.get(matches_hash)
        if not target_sources:
            continue

        for evidence in stmt.evidence:
            source_hash = evidence.get_source_hash()
            pair = (matches_hash, source_hash)
            if pair not in gold or pair in found:
                continue
            found[pair] = {
                "matches_hash": matches_hash,
                "source_hash": source_hash,
                "statement": stmt,
                "evidence_text": evidence.text,
                "source_api": evidence.source_api,
                "pmid": evidence.pmid or evidence.text_refs.get("PMID"),
            }

        if len(found) == len(gold):
            break

    missing = set(gold) - set(found)
    if missing:
        raise RuntimeError(f"Could not find {len(missing)} curated pairs in the corpus")
    return [found[pair] for pair in sorted(found)]


def read_completed(path, model):
    if not path.exists():
        return {}
    completed = {}
    with path.open() as file:
        for line in file:
            row = json.loads(line)
            if row.get("model") == model and row.get("prediction"):
                pair = (row["matches_hash"], row["source_hash"])
                completed[pair] = row
    return completed


def score(rows, gold):
    predicted = [row for row in rows if row.get("prediction")]
    decided = [row for row in predicted if row["prediction"] != "uncertain"]
    gold_for = lambda row: gold[(row["matches_hash"], row["source_hash"])]
    exact = sum(row["prediction"] == gold_for(row) for row in predicted)
    return {
        "samples": len(predicted),
        "gold": dict(Counter(gold_for(row) for row in predicted)),
        "predictions": dict(Counter(row["prediction"] for row in predicted)),
        "accuracy": exact / len(predicted) if predicted else None,
        "coverage": len(decided) / len(predicted) if predicted else None,
    }


def run_one(
    client,
    entry,
    abstract_by_pmid,
    mesh_by_pmid,
    contexts,
):
    if not entry["evidence_text"]:
        raise ValueError(
            f"Missing evidence text for "
            f"{entry['matches_hash']}:{entry['source_hash']}"
        )
    context = {}
    if "uniprot" in contexts:
        context["uniprot_context"] = get_uniprot_context(entry["statement"])
    if "abstract" in contexts:
        context["abstract"] = (
            abstract_by_pmid.get(int(entry["pmid"])) if entry["pmid"] else None
        )
    if "mesh" in contexts:
        context["mesh_terms"] = (
            mesh_by_pmid.get(int(entry["pmid"]), []) if entry["pmid"] else []
        )
    result = curate(client, entry["statement"], entry["evidence_text"], **context)
    decision = result["decision"]
    prediction = {
        "accepted": "correct",
        "rejected": "incorrect",
        "uncertain": "uncertain",
    }[decision]
    return {
        "matches_hash": entry["matches_hash"],
        "source_hash": entry["source_hash"],
        "evidence_text": entry["evidence_text"],
        "source_api": entry["source_api"],
        "pmid": entry["pmid"],
        "prediction": prediction,
        **result,
    }


def main():
    started_at = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--provider", choices=["local", "openai"], default="local")
    parser.add_argument(
        "--context",
        nargs="+",
        choices=["none", "uniprot", "abstract", "mesh", "full"],
        default=["none"],
    )
    parser.add_argument("--limit", type=int, default=0, help="0 means all unique curated pairs")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    model = args.model or (
        "gpt-5.6-luna"
        if args.provider == "openai"
        else "google/gemma-4-26B-A4B-it"
    )

    contexts = set(args.context) - {"none"}
    if "full" in contexts:
        contexts = {"uniprot", "abstract", "mesh"}
        context_name = "full"
    else:
        context_name = "_".join(
            name for name in ("uniprot", "abstract", "mesh") if name in contexts
        ) or "none"

    safe_model = model.replace("/", "_")
    output = args.output or Path("outputs") / (
        f"{args.provider}_{context_name}_{safe_model}.jsonl"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    gold = load_gold(args.limit or None, args.seed)
    entries = load_entries(gold)
    abstract_by_pmid, abstract_cache_stats = (
        get_abstracts(entry["pmid"] for entry in entries)
        if "abstract" in contexts
        else ({}, None)
    )
    abstract_stats = None
    if "abstract" in contexts:
        available = sum(
            bool(abstract_by_pmid.get(int(entry["pmid"])))
            for entry in entries
            if entry["pmid"]
        )
        abstract_stats = {
            "available": available,
            "missing": len(entries) - available,
            "cache": abstract_cache_stats,
        }
        print(
            f"abstract available={abstract_stats['available']} "
            f"missing={abstract_stats['missing']}"
        )
    mesh_by_pmid = (
        load_mesh_terms(entry["pmid"] for entry in entries)
        if "mesh" in contexts
        else {}
    )
    completed = read_completed(output, model)
    pending = [
        entry
        for entry in entries
        if (entry["matches_hash"], entry["source_hash"]) not in completed
    ]
    print(f"benchmark={len(entries)} completed={len(completed)} pending={len(pending)}")

    client = (
        OpenAILLMClient(model)
        if args.provider == "openai"
        else LLMClient(model)
    )
    successful_this_run = 0
    with output.open("a") as file, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_one,
                client,
                entry,
                abstract_by_pmid,
                mesh_by_pmid,
                contexts,
            ): entry
            for entry in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            entry = futures[future]
            try:
                row = future.result()
            except Exception as error:
                print(
                    f"error {entry['matches_hash']}:{entry['source_hash']}: {error}"
                )
                continue
            completed[(row["matches_hash"], row["source_hash"])] = row
            successful_this_run += 1
            file.write(json.dumps(row) + "\n")
            file.flush()
            if index % 10 == 0 or index == len(pending):
                print(f"finished {index}/{len(pending)}")

    rows = [
        completed[(entry["matches_hash"], entry["source_hash"])]
        for entry in entries
        if (entry["matches_hash"], entry["source_hash"]) in completed
    ]
    summary = score(rows, gold)
    if abstract_stats:
        summary["abstracts"] = abstract_stats
    elapsed_seconds = time.perf_counter() - started_at
    summary["processed_this_run"] = successful_this_run
    summary["elapsed_seconds"] = round(elapsed_seconds, 2)
    summary["elapsed"] = str(timedelta(seconds=round(elapsed_seconds)))
    print(json.dumps(summary, indent=2))
    print(f"results={output}")


if __name__ == "__main__":
    main()
