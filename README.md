# LLM Belief

This package uses an LLM to verify whether INDRA evidence supports its
statement. It contains tools for benchmarking the verifier, processing large
statement sets, building a SQLite database, and preparing CoGEx ingestion
files.

## Installation

Install the package from the repository root:

```bash
python -m pip install -e .
```

Every command supports `--help`:

```bash
python -m llm_belief.benchmark --help
```

Default data paths are defined in `llm_belief/locations.py`. Override the
corresponding command-line arguments when the inputs are stored elsewhere.

## 1. Run a benchmark

The benchmark compares LLM predictions with the human-curated INDRA assembly
curations. `--context full` adds UniProt, FamPlex, abstract, and MeSH context.

```bash
python -m llm_belief.benchmark \
  --provider local \
  --model openai/gpt-oss-120b \
  --context full \
  --workers 64 \
  --max-tokens 512 \
  --reasoning-effort low \
  --output outputs/gpt-oss-120b_full.jsonl
```

For only curated statements containing a FamPlex agent, add
`--famplex-only`.

The command writes:

- the prediction JSONL given by `--output`;
- `<name>_summary.json` with overall and per-category statistics;
- `<name>_categories.tsv` with the category matrix;
- `<name>_disagreements.jsonl` for manual review.

An existing output JSONL is resumed rather than restarted. Use
`--provider openai` with `OPENAI_API_KEY` to call the OpenAI API instead of an
OpenAI-compatible local server.

## 2. Process statements at scale

### Prepare shared data

Create the PMCID-to-PMID map required by the large-scale processor:

```bash
python -m llm_belief.pmcid_map --batch-size 100000
```

The abstract cache is optional but avoids repeated abstract lookups:

```bash
python -m llm_belief.abstract_cache --batch-size 1000
```

Both commands require access to the INDRA database.

### Process gene statements

Start the local vLLM servers first, then run:

```bash
python -m llm_belief.process_gene_statements \
  --workers 192 \
  --chunk-size 5000 \
  --server http://127.0.0.1:8000/v1 \
  --server http://127.0.0.1:8001/v1 \
  --server http://127.0.0.1:8002/v1
```

Results are written to a timestamped result directory as
`results_NNNN.jsonl` files. The command prints the output directory. Repeated
`--server` arguments distribute requests across the listed servers.

To process only evidence missing from an earlier run:

```bash
python -m llm_belief.process_gene_statements \
  --retry-from /path/to/previous_results \
  --workers 192 \
  --chunk-size 5000 \
  --server http://127.0.0.1:8000/v1 \
  --server http://127.0.0.1:8001/v1 \
  --server http://127.0.0.1:8002/v1
```

`--retry-from` must point to an earlier result directory containing
`results_*.jsonl`. It can be repeated when completed results are split across
multiple directories.

### Process FamPlex statements

First build the FamPlex statement hash set:

```bash
python -m llm_belief.build_famplex_hashes
```

Then pass that set to the large-scale processor:

```bash
python -m llm_belief.process_gene_statements \
  --statement-hashes /path/to/famplex_stmt_hashes.pkl \
  --famplex-context \
  --workers 192 \
  --chunk-size 5000 \
  --server http://127.0.0.1:8000/v1 \
  --server http://127.0.0.1:8001/v1
```

### Add textless database evidence

Database evidence without evidence text is treated as correct. Add it to an
existing result directory before building the SQLite database:

```bash
python -m llm_belief.add_database_evidence \
  /path/to/results_directory
```

This writes `results_database_evidence.jsonl` in that directory. Use
`--statement-hashes` when processing a set other than gene statements.

## 3. Build or update the SQLite database

Build a database from one or more result directories:

```bash
python -m llm_belief.build_curation_database \
  /path/to/results_directory_1 \
  /path/to/results_directory_2 \
  --output /path/to/curations.sqlite \
  --overwrite
```

Each positional input is a directory containing result JSONL files from
`process_gene_statements` or a retry run. One input directory is sufficient;
pass more than one to combine runs. `--output` is the SQLite file to create.

Use `--append` instead of `--overwrite` to add results to an existing
database. Appending does not recalculate statement scores, so rebuild only the
score table afterward:

```bash
python -m llm_belief.add_statement_correctness \
  /path/to/results_directory_1 \
  /path/to/results_directory_2 \
  /path/to/curations.sqlite
```

Here the result directories are the source used to recalculate the counts,
and the final positional argument is the existing SQLite database to update.

The SQLite database contains:

- `llm_evidence_curation`: one judgment per `(statement_hash, source_hash)`,
  including the explanation and error category;
- `llm_statement_correctness`: correct count, total count, and correctness
  percentage for each statement.

## 4. Build CoGEx ingestion files

The input node and edge files must come from the same CoGEx build as the
statements in the SQLite database.

Build the complete statement edge file with the LLM property:

```bash
python -m llm_belief.cogex_ingestion \
  /path/to/curations.sqlite \
  /path/to/output_directory \
  --statements-only \
  --statement-edges /path/to/cogex/edges.tsv.gz
```

Build the Evidence node update file:

```bash
python -m llm_belief.cogex_ingestion \
  /path/to/curations.sqlite \
  /path/to/output_directory \
  --evidence-only \
  --evidence-nodes /path/to/cogex/nodes_Evidence.tsv.gz
```

The first positional argument is the SQLite verification database. The second
is the directory in which the generated ingestion files will be written.
`--statement-edges` and `--evidence-nodes` are the original files produced by
the CoGEx INDRA DB processors.

Omit `--statements-only` and `--evidence-only` to create both outputs in one
run. Evidence judgments are loaded from SQLite once and matched in memory.

The outputs and added CoGEx properties are:

| Output | Property | Values |
| --- | --- | --- |
| `nodes_Evidence_llm_verification.tsv.gz` | `Evidence.llm_verification` | `correct`, `incorrect` |
| `edges_indra_rel_llm_verification.tsv.gz` | `indra_rel.llm_correctness` | `weak`, `medium`, `strong` |

Statement correctness levels are calculated as follows:

- `weak`: less than 33% of evidence is correct;
- `medium`: 33% through 66%, or over 66% with fewer than three pieces of
  evidence;
- `strong`: over 66% with at least three pieces of evidence.

The generated files are inputs to CoGEx partial ingestion; this package does
not modify Neo4j directly.

## Script reference

| Module | Purpose |
| --- | --- |
| `benchmark` | Evaluate the verifier against curated evidence. |
| `reasoning_benchmark` | Run a grid of reasoning efforts and token limits. |
| `pmcid_map` | Export the PMCID-to-PMID lookup used by large runs. |
| `abstract_cache` | Build or update the PubMed abstract cache. |
| `build_famplex_hashes` | Create the FamPlex statement hash set. |
| `process_gene_statements` | Verify every evidence item for a statement hash set. |
| `add_database_evidence` | Add textless database evidence as correct results. |
| `build_curation_database` | Build or append to the SQLite result database. |
| `add_statement_correctness` | Rebuild only the statement score table. |
| `cogex_ingestion` | Add LLM properties to CoGEx processor files. |

`context.py`, `curation.py`, `data.py`, and `llm.py` provide shared internal
logic and are not command-line entry points.
