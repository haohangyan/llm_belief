"""Optional MeSH and UniProt context for curation."""

import json
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlencode
from urllib.request import urlopen

from tqdm import tqdm

from llm_belief.locations import (
    INDRA_DB_LITE_PATH,
    UNIPROT_CACHE_PATH,
)


def _mesh_id(mesh_num, is_concept):
    prefix = "C" if is_concept else "D"
    # mesh_num < 66332 and < 588418 use the old six-digit format.
    width = 6 if mesh_num < (588418 if is_concept else 66332) else 9
    return f"{prefix}{mesh_num:0{width}d}"


def load_mesh_terms(pmids):
    if not INDRA_DB_LITE_PATH.exists():
        return {}

    pmids = {int(pmid) for pmid in pmids if pmid}
    if not pmids:
        return {}

    uri = f"file:{INDRA_DB_LITE_PATH}?mode=ro"
    with sqlite3.connect(uri, uri=True) as source:
        source.execute("CREATE TEMP TABLE wanted_pmids (pmid INTEGER PRIMARY KEY)")
        source.executemany(
            "INSERT INTO wanted_pmids VALUES (?)", ((pmid,) for pmid in pmids)
        )
        rows = source.execute("""
            SELECT pmid_num, mesh_num, is_concept
            FROM mesh_pmids
            JOIN wanted_pmids ON pmid_num = wanted_pmids.pmid
        """)

        from indra.databases.mesh_client import mesh_id_to_name

        terms_by_pmid = {pmid: [] for pmid in pmids}
        for pmid, mesh_num, is_concept in rows:
            mesh_id = _mesh_id(mesh_num, is_concept)
            terms_by_pmid[pmid].append(mesh_id_to_name.get(mesh_id, mesh_id))

    return dict(
        (pmid, list(dict.fromkeys(terms)))
        for pmid, terms in terms_by_pmid.items()
    )


def _candidate_genes(statement):
    names = []
    for agent in statement.agent_list():
        if agent and agent.name and re.fullmatch(r"[A-Za-z0-9-]{2,20}", agent.name):
            names.append(agent.name.upper())
    return list(dict.fromkeys(names))


def _fetch_uniprot(gene):
    params = urlencode({
        "query": f"gene:{gene}",
        "format": "json",
        "limit": 1,
        "fields": "gene_names,protein_name,cc_function",
    })
    with urlopen(
        f"https://rest.uniprot.org/uniprotkb/search?{params}", timeout=3.5
    ) as response:
        results = json.load(response).get("results", [])

    if not results:
        return {"error": f"No UniProt entry found for gene: {gene}"}

    result = results[0]
    gene_names = []
    for item in result.get("genes", []):
        for key in ("geneName", "synonyms", "orderedLocusNames", "orfNames"):
            values = [item.get(key)] if key == "geneName" else item.get(key, [])
            gene_names.extend(
                value["value"] for value in values if value and value.get("value")
            )

    description = result.get("proteinDescription", {})
    recommended = description.get("recommendedName", {})
    protein_names = [recommended.get("fullName", {}).get("value")]
    protein_names.extend(name.get("value") for name in recommended.get("shortNames", []))
    protein_names.extend(
        name.get("fullName", {}).get("value")
        for name in description.get("alternativeNames", [])
    )
    function = next(
        (
            comment.get("texts", [{}])[0].get("value")
            for comment in result.get("comments", [])
            if comment.get("commentType") == "FUNCTION" and comment.get("texts")
        ),
        None,
    )
    return {
        "gene_name": gene_names[0] if gene_names else gene,
        "gene_synonyms": list(dict.fromkeys(gene_names)),
        "protein_names": list(dict.fromkeys(name for name in protein_names if name)),
        "function": function,
    }


_UNIPROT_ENTRIES = None


def _load_uniprot_entries():
    global _UNIPROT_ENTRIES
    if _UNIPROT_ENTRIES is not None:
        return _UNIPROT_ENTRIES

    started = time.perf_counter()
    tqdm.write(f"[uniprot] loading cache from {UNIPROT_CACHE_PATH}")
    UNIPROT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(UNIPROT_CACHE_PATH) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS entries (gene TEXT PRIMARY KEY, data TEXT)"
        )
        _UNIPROT_ENTRIES = {
            gene: json.loads(data)
            for gene, data in connection.execute("SELECT gene, data FROM entries")
        }
    tqdm.write(
        f"[uniprot] loaded={len(_UNIPROT_ENTRIES):,} "
        f"elapsed={time.perf_counter() - started:.1f}s"
    )
    return _UNIPROT_ENTRIES


def _fetch_safely(gene):
    try:
        return gene, _fetch_uniprot(gene)
    except Exception as error:
        return gene, {"error": str(error)}


def _add_missing_uniprot(genes, entries):
    missing = genes - entries.keys()
    tqdm.write(
        f"[uniprot] genes={len(genes):,} "
        f"cached={len(genes) - len(missing):,} missing={len(missing):,}"
    )
    if not missing:
        return

    started = time.perf_counter()
    saved = 0
    failed = 0
    pending = []
    with (
        ThreadPoolExecutor(max_workers=min(16, len(missing))) as pool,
        sqlite3.connect(UNIPROT_CACHE_PATH) as connection,
    ):
        futures = [pool.submit(_fetch_safely, gene) for gene in missing]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Fetching UniProt",
            unit="gene",
        ):
            gene, data = future.result()
            entries[gene] = data
            if data.get("error"):
                failed += 1
                continue
            pending.append((gene, json.dumps(data)))
            saved += 1
            if len(pending) >= 100:
                connection.executemany(
                    "INSERT OR REPLACE INTO entries VALUES (?, ?)", pending
                )
                connection.commit()
                pending.clear()
        if pending:
            connection.executemany(
                "INSERT OR REPLACE INTO entries VALUES (?, ?)", pending
            )
            connection.commit()
    tqdm.write(
        f"[uniprot] saved={saved:,} failed={failed:,} "
        f"elapsed={time.perf_counter() - started:.1f}s"
    )


def _strip_pubmed_citations(text):
    text = re.sub(r"\s*\((?:PubMed:[0-9]+(?:,\s*PubMed:[0-9]+)*)\)\s*", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _format_uniprot_context(genes, entries):
    lines = []
    for gene in genes:
        data = entries.get(gene, {})
        if data.get("error"):
            continue
        symbol = data.get("gene_name") or "Unknown"
        gene_synonyms = ", ".join(data.get("gene_synonyms", [])[:12]) or symbol
        protein_names = ", ".join(data.get("protein_names", [])[:8]) or "N/A"
        function = data.get("function")
        function = _strip_pubmed_citations(function) if function else "N/A"
        lines.append(
            f"- {symbol}\n"
            "  Match note: First UniProt search hit for the queried token "
            "(limit=1); may be a non-exact match.\n"
            f"  Gene names/synonyms: {gene_synonyms}\n"
            f"  Protein names/synonyms: {protein_names}\n"
            f"  Function: {function}"
        )

    return f"""ENTITY NORMALIZATION CONTEXT (UniProt; optional)
Queried gene tokens from statement: {', '.join(genes)}
For each token, we attached ONLY the first UniProt search result (query: gene:<token>, limit=1).
This may or may not refer to the exact mentioned entity in the evidence.
Use this ONLY as tentative grounding/disambiguation context, and do NOT infer relations not present in the evidence.

{chr(10).join(lines)}"""


def load_uniprot_contexts(statements):
    """Build CuraTogether-style UniProt context, keyed by statement hash."""
    genes_by_hash = {
        statement.get_hash(): _candidate_genes(statement) for statement in statements
    }
    entries = _load_uniprot_entries()
    genes = {gene for names in genes_by_hash.values() for gene in names}
    _add_missing_uniprot(genes, entries)
    return {
        statement_hash: _format_uniprot_context(genes, entries) if genes else None
        for statement_hash, genes in genes_by_hash.items()
    }


def get_uniprot_context(statement):
    """Return UniProt context for one statement."""
    return load_uniprot_contexts([statement]).get(statement.get_hash())
