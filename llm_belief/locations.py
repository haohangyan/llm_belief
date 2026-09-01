"""Locations of the external INDRA data files."""

from pathlib import Path


DATA_DIRECTORY = Path("/scratch/h.yan/data/llm_belief_data")

CURATIONS_PATH = DATA_DIRECTORY / "indra_assembly_curations.json"
CORPUS_JSON_PATH = DATA_DIRECTORY / "indra_benchmark_corpus.json.gz"
CORPUS_PICKLE_PATH = DATA_DIRECTORY / "indra_benchmark_corpus.pkl"
INDRA_DB_LITE_PATH = DATA_DIRECTORY / "indra_lite.db"
UNIPROT_CACHE_PATH = DATA_DIRECTORY / "uniprot_cache.db"
