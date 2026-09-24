"""Benchmark the Gemma curation workflow against human curations."""

import argparse
import csv
import json
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

from llm_belief.abstract_cache import get_abstracts
from llm_belief.curation import ERROR_CATEGORIES, curate
from llm_belief.context import (
    load_mesh_terms,
    load_uniprot_contexts,
)
from llm_belief.data import load_pickle_statements
from llm_belief.llm import LLMClient, OpenAILLMClient
from llm_belief.locations import CORPUS_PICKLE_PATH, CURATIONS_PATH


def load_gold(limit, seed, include_tags=False):
    with CURATIONS_PATH.open() as file:
        curations = json.load(file)

    grouped = defaultdict(list)
    for row in curations:
        grouped[(int(row["pa_hash"]), int(row["source_hash"]))].append(row)

    pairs = sorted(grouped)
    if limit and limit < len(pairs):
        pairs = random.Random(seed).sample(pairs, limit)

    gold = {
        pair: "correct"
        if all(row["tag"] == "correct" for row in grouped[pair])
        else "incorrect"
        for pair in pairs
    }
    if not include_tags:
        return gold
    gold_tags = {
        pair: sorted({row["tag"] for row in grouped[pair]}) for pair in pairs
    }
    return gold, gold_tags


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


def score_by_gold_tag(rows, gold_tags):
    rows_by_tag = defaultdict(list)
    for row in rows:
        pair = (row["matches_hash"], row["source_hash"])
        for tag in gold_tags[pair]:
            rows_by_tag[tag].append(row)

    summary = {}
    for tag, tag_rows in sorted(rows_by_tag.items()):
        expected_prediction = "correct" if tag == "correct" else "incorrect"
        label_matches = sum(
            row["prediction"] == expected_prediction for row in tag_rows
        )
        tag_summary = {
            "samples": len(tag_rows),
            "predictions": dict(Counter(row["prediction"] for row in tag_rows)),
            "error_categories": dict(
                Counter(row.get("error_category") or "null" for row in tag_rows)
            ),
            "label_matches": label_matches,
            "label_accuracy": label_matches / len(tag_rows),
        }
        if tag != "correct":
            category_matches = sum(
                row["prediction"] == "incorrect"
                and row.get("error_category") == tag
                for row in tag_rows
            )
            tag_summary["error_category_matches"] = category_matches
            tag_summary["error_category_accuracy"] = (
                category_matches / len(tag_rows)
            )
        summary[tag] = tag_summary
    return summary


def score(rows, gold, gold_tags=None):
    predicted = [row for row in rows if row.get("prediction")]
    decided = [row for row in predicted if row["prediction"] != "uncertain"]
    gold_for = lambda row: gold[(row["matches_hash"], row["source_hash"])]
    exact = sum(row["prediction"] == gold_for(row) for row in predicted)
    total = len(gold)
    tp = sum(
        row["prediction"] == "correct" and gold_for(row) == "correct"
        for row in decided
    )
    fp = sum(
        row["prediction"] == "correct" and gold_for(row) == "incorrect"
        for row in decided
    )
    tn = sum(
        row["prediction"] == "incorrect" and gold_for(row) == "incorrect"
        for row in decided
    )
    fn = sum(
        row["prediction"] == "incorrect" and gold_for(row) == "correct"
        for row in decided
    )
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    summary = {
        "total": total,
        "samples": len(predicted),
        "failed": total - len(predicted),
        "gold": dict(Counter(gold.values())),
        "predictions": dict(Counter(row["prediction"] for row in predicted)),
        "accuracy": exact / len(predicted) if predicted else None,
        "overall_accuracy": exact / total if total else None,
        "coverage": len(decided) / len(predicted) if predicted else None,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall)
            if precision is not None
            and recall is not None
            and precision + recall
            else None
        ),
    }
    if gold_tags is not None:
        summary["by_gold_tag"] = score_by_gold_tag(predicted, gold_tags)
    return summary


def write_category_tsv(summary, output):
    category_output = output.with_name(f"{output.stem}_categories.tsv")
    categories = ["correct", *ERROR_CATEGORIES]
    with category_output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["gold_category", "n", *categories])
        for gold_tag, tag_summary in summary["by_gold_tag"].items():
            sample_count = tag_summary["samples"]
            counts = tag_summary["error_categories"]
            percentages = []
            for category in categories:
                key = "null" if category == "correct" else category
                percentages.append(
                    f"{100 * counts.get(key, 0) / sample_count:.1f}%"
                )
            writer.writerow([gold_tag, sample_count, *percentages])
    return category_output


def write_disagreements(rows, entries, gold, gold_tags, output):
    entries_by_pair = {
        (entry["matches_hash"], entry["source_hash"]): entry
        for entry in entries
    }
    disagreements = []
    for row in rows:
        pair = (row["matches_hash"], row["source_hash"])
        label = gold[pair]
        if row["prediction"] == label:
            continue
        entry = entries_by_pair[pair]
        disagreements.append(
            {
                "statement_hash": row["matches_hash"],
                "source_hash": row["source_hash"],
                "statement": str(entry["statement"]),
                "evidence_text": entry["evidence_text"],
                "label": label,
                "gold_tags": gold_tags[pair],
                "llm_judgment": row["prediction"],
                "error_category": row.get("error_category"),
                "explanation": row.get("reasoning", ""),
            }
        )

    review_output = output.with_name(f"{output.stem}_disagreements.jsonl")
    with review_output.open("w", encoding="utf-8") as file:
        for row in disagreements:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return review_output, len(disagreements)


def run_one(
    client,
    entry,
    abstract_by_pmid,
    mesh_by_pmid,
    uniprot_by_statement,
    contexts,
):
    if not entry["evidence_text"]:
        raise ValueError(
            f"Missing evidence text for "
            f"{entry['matches_hash']}:{entry['source_hash']}"
        )
    context = {}
    if "uniprot" in contexts:
        context["uniprot_context"] = uniprot_by_statement.get(
            entry["matches_hash"]
        )
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
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--reasoning-effort", choices=["low", "medium", "high"]
    )
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
    reasoning_tag = (
        f"_{args.reasoning_effort}_{args.max_tokens}"
        if args.reasoning_effort
        else ""
    )
    output = args.output or Path("outputs") / (
        f"{args.provider}_{context_name}_{safe_model}{reasoning_tag}.jsonl"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[setup] loading gold curations from {CURATIONS_PATH}", flush=True)
    phase_started = time.perf_counter()
    gold, gold_tags = load_gold(
        args.limit or None, args.seed, include_tags=True
    )
    print(
        f"[setup] loaded {len(gold)} gold pairs "
        f"in {time.perf_counter() - phase_started:.1f}s",
        flush=True,
    )

    print(f"[setup] loading and matching corpus from {CORPUS_PICKLE_PATH}", flush=True)
    phase_started = time.perf_counter()
    entries = load_entries(gold)
    print(
        f"[setup] matched {len(entries)} corpus entries "
        f"in {time.perf_counter() - phase_started:.1f}s",
        flush=True,
    )

    if "abstract" in contexts:
        print("[setup] loading abstracts", flush=True)
        phase_started = time.perf_counter()
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
            f"missing={abstract_stats['missing']} "
            f"elapsed={time.perf_counter() - phase_started:.1f}s",
            flush=True,
        )

    if "mesh" in contexts:
        print("[setup] loading MeSH terms", flush=True)
        phase_started = time.perf_counter()
    mesh_by_pmid = (
        load_mesh_terms(entry["pmid"] for entry in entries)
        if "mesh" in contexts
        else {}
    )
    if "mesh" in contexts:
        print(
            f"[setup] loaded MeSH terms in "
            f"{time.perf_counter() - phase_started:.1f}s",
            flush=True,
        )

    if "uniprot" in contexts:
        print("[setup] preparing UniProt context", flush=True)
        phase_started = time.perf_counter()
    uniprot_by_statement = (
        load_uniprot_contexts(entry["statement"] for entry in entries)
        if "uniprot" in contexts
        else {}
    )
    if "uniprot" in contexts:
        print(
            f"[setup] prepared UniProt context in "
            f"{time.perf_counter() - phase_started:.1f}s",
            flush=True,
        )

    print(f"[setup] reading completed results from {output}", flush=True)
    completed = read_completed(output, model)
    pending = [
        entry
        for entry in entries
        if (entry["matches_hash"], entry["source_hash"]) not in completed
    ]
    pending.sort(key=lambda entry: (int(entry["pmid"] or 0), entry["matches_hash"]))
    print(
        f"benchmark={len(entries)} completed={len(completed)} pending={len(pending)}",
        flush=True,
    )

    client = (
        OpenAILLMClient(model)
        if args.provider == "openai"
        else LLMClient(model, args.max_tokens, args.reasoning_effort)
    )
    print(
        f"[run] starting model requests with {args.workers} workers",
        flush=True,
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
                uniprot_by_statement,
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
    summary = score(rows, gold, gold_tags)
    review_output, disagreement_count = write_disagreements(
        rows,
        entries,
        gold,
        gold_tags,
        output,
    )
    summary["disagreements"] = disagreement_count
    if abstract_stats:
        summary["abstracts"] = abstract_stats
    elapsed_seconds = time.perf_counter() - started_at
    summary["processed_this_run"] = successful_this_run
    summary["elapsed_seconds"] = round(elapsed_seconds, 2)
    summary["elapsed"] = str(timedelta(seconds=round(elapsed_seconds)))
    summary_output = output.with_name(f"{output.stem}_summary.json")
    summary_output.write_text(json.dumps(summary, indent=2) + "\n")
    category_output = write_category_tsv(summary, output)
    print(
        f"accuracy={summary['accuracy']:.4f} "
        f"f1={summary['f1']:.4f} "
        f"disagreements={summary['disagreements']} "
        f"elapsed={summary['elapsed']}"
    )
    print(f"results={output}")
    print(f"summary={summary_output}")
    print(f"categories={category_output}")
    print(f"disagreements={review_output}")


if __name__ == "__main__":
    main()
