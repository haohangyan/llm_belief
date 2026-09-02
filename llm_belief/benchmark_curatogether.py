"""Benchmark Curatogether's workflow with a local Gemma model."""

from llm_belief.benchmark_openai import main
from llm_belief.curatogether import curate_curatogether


if __name__ == "__main__":
    main(
        curate_function=curate_curatogether,
        default_context=("full",),
        default_provider="local",
        default_model="google/gemma-4-26B-A4B-it",
        output_tag="curatogether",
    )
