"""Run AI curation against the INDRA human-curation benchmark."""

import argparse
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from llm_belief.ai_curation import curate
from llm_belief.data import load_pickle_statements
from llm_belief.llm import OpenAILLMClient
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
        pair: {
            "gold": "correct" if all(row["tag"] == "correct" for row in grouped[pair]) else "incorrect",
            "human_tags": sorted({row["tag"] for row in grouped[pair]}),
        }
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
                "record_id": f"{matches_hash}:{source_hash}",
                "matches_hash": matches_hash,
                "source_hash": source_hash,
                "statement": stmt,
                "evidence_text": evidence.text,
                "source_api": evidence.source_api,
                "pmid": evidence.pmid or evidence.text_refs.get("PMID"),
                **gold[pair],
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
                completed[row["record_id"]] = row
    return completed


def score(rows):
    predicted = [row for row in rows if row.get("prediction")]
    decided = [row for row in predicted if row["prediction"] != "uncertain"]
    exact = sum(row["prediction"] == row["gold"] for row in predicted)
    decided_exact = sum(row["prediction"] == row["gold"] for row in decided)
    return {
        "samples": len(predicted),
        "gold": dict(Counter(row["gold"] for row in predicted)),
        "predictions": dict(Counter(row["prediction"] for row in predicted)),
        "accuracy": exact / len(predicted) if predicted else None,
        "decided_accuracy": decided_exact / len(decided) if decided else None,
        "coverage": len(decided) / len(predicted) if predicted else None,
        "input_tokens": sum(row.get("input_tokens") or 0 for row in predicted),
        "output_tokens": sum(row.get("output_tokens") or 0 for row in predicted),
    }


def run_one(client, entry):
    if not entry["evidence_text"]:
        raise ValueError(f"Missing evidence text for {entry['record_id']}")
    result = curate(client, entry["statement"], entry["evidence_text"])
    decision = result["decision"]
    prediction = {
        "accepted": "correct",
        "rejected": "incorrect",
        "uncertain": "uncertain",
    }[decision]
    return {
        **entry,
        "statement": None,
        "prediction": prediction,
        **result,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--limit", type=int, default=0, help="0 means all unique curated pairs")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    safe_model = args.model.replace("/", "_")
    output = args.output or Path("outputs") / f"openai_{safe_model}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)

    gold = load_gold(args.limit or None, args.seed)
    entries = load_entries(gold)
    completed = read_completed(output, args.model)
    pending = [entry for entry in entries if entry["record_id"] not in completed]
    print(f"benchmark={len(entries)} completed={len(completed)} pending={len(pending)}")

    client = OpenAILLMClient(args.model)
    with output.open("a") as file, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, client, entry): entry for entry in pending}
        for index, future in enumerate(as_completed(futures), start=1):
            entry = futures[future]
            try:
                row = future.result()
            except Exception as error:
                print(f"error {entry['record_id']}: {error}")
                continue
            completed[row["record_id"]] = row
            file.write(json.dumps(row) + "\n")
            file.flush()
            if index % 10 == 0 or index == len(pending):
                print(f"finished {index}/{len(pending)}")

    rows = [completed[entry["record_id"]] for entry in entries if entry["record_id"] in completed]
    print(json.dumps(score(rows), indent=2))
    print(f"results={output}")


if __name__ == "__main__":
    main()
