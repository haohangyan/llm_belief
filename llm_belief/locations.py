"""Locations of the external INDRA data files."""

import pystow


# Defaults to ~/.data/llm_belief_data. Set PYSTOW_HOME to move the data root.
DATA_MODULE = pystow.module("llm_belief_data")
DATA_DIRECTORY = DATA_MODULE.base

CURATIONS_PATH = DATA_DIRECTORY / "indra_assembly_curations.json"
CORPUS_JSON_PATH = DATA_DIRECTORY / "indra_benchmark_corpus.json.gz"
CORPUS_PICKLE_PATH = DATA_DIRECTORY / "indra_benchmark_corpus.pkl"
