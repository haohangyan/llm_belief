"""Compare GPT-OSS reasoning effort and output-token limits."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from llm_belief.benchmark import load_gold, read_completed, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument(
        "--context",
        nargs="+",
        choices=["none", "uniprot", "abstract", "mesh", "full"],
        default=["full"],
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    safe_model = args.model.replace("/", "_")
    context_name = "_".join(args.context)
    gold = load_gold(args.limit or None, args.seed)
    summaries = []

    for max_tokens in (512, 1024):
        for effort in ("low", "medium", "high"):
            output = Path("outputs") / (
                f"reasoning_{context_name}_{safe_model}_{effort}_{max_tokens}.jsonl"
            )
            print(
                f"\n[reasoning benchmark] effort={effort} "
                f"max_tokens={max_tokens}",
                flush=True,
            )
            command = [
                sys.executable,
                "-m",
                "llm_belief.benchmark",
                "--provider",
                "local",
                "--model",
                args.model,
                "--context",
                *args.context,
                "--workers",
                str(args.workers),
                "--max-tokens",
                str(max_tokens),
                "--reasoning-effort",
                effort,
                "--output",
                str(output),
                "--seed",
                str(args.seed),
            ]
            if args.limit:
                command.extend(["--limit", str(args.limit)])
            subprocess.run(command, check=True)

            completed = read_completed(output, args.model)
            rows = [row for pair, row in completed.items() if pair in gold]
            summary = score(rows, gold)
            summary = {
                "reasoning_effort": effort,
                "max_tokens": max_tokens,
                **summary,
            }
            summaries.append(summary)

    summary_path = Path("outputs") / (
        f"reasoning_{context_name}_{safe_model}_summary.json"
    )
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
