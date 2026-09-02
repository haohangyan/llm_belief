"""Optional abstract and UniProt context for curation."""

import json
import re
import sqlite3
from urllib.parse import urlencode
from urllib.request import urlopen

from llm_belief.locations import (
    INDRA_DB_LITE_PATH,
    UNIPROT_CACHE_PATH,
)


def get_abstract(pmid):
    if not pmid or not INDRA_DB_LITE_PATH.exists():
        return None

    query = """
        SELECT best_content.content
        FROM pmid_text_refs
        JOIN best_content USING (text_ref_id)
        WHERE pmid_text_refs.pmid = ? AND best_content.text_type = 'abstract'
        LIMIT 1
    """
    uri = f"file:{INDRA_DB_LITE_PATH}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        row = connection.execute(query, (int(pmid),)).fetchone()

    if not row:
        return None
    return "\n".join(json.loads(row[0]))


def load_abstracts(pmids):
    if not INDRA_DB_LITE_PATH.exists():
        return {}

    pmids = {int(pmid) for pmid in pmids if pmid}
    if not pmids:
        return {}

    uri = f"file:{INDRA_DB_LITE_PATH}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("CREATE TEMP TABLE wanted_pmids (pmid INTEGER PRIMARY KEY)")
        connection.executemany(
            "INSERT INTO wanted_pmids VALUES (?)", ((pmid,) for pmid in pmids)
        )
        rows = connection.execute("""
            SELECT pmid_text_refs.pmid, best_content.content
            FROM pmid_text_refs
            JOIN wanted_pmids ON pmid_text_refs.pmid = wanted_pmids.pmid
            JOIN best_content USING (text_ref_id)
            WHERE best_content.text_type = 'abstract'
        """)
        abstracts = {}
        for pmid, content in rows:
            abstracts.setdefault(pmid, "\n".join(json.loads(content)))
    return abstracts


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


def _read_cache(gene):
    if not UNIPROT_CACHE_PATH.exists():
        return None
    with sqlite3.connect(UNIPROT_CACHE_PATH) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS entries (gene TEXT PRIMARY KEY, data TEXT)"
        )
        row = connection.execute(
            "SELECT data FROM entries WHERE gene = ?", (gene,)
        ).fetchone()
    return json.loads(row[0]) if row else None


def _write_cache(gene, data):
    with sqlite3.connect(UNIPROT_CACHE_PATH) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS entries (gene TEXT PRIMARY KEY, data TEXT)"
        )
        connection.execute(
            "INSERT OR REPLACE INTO entries VALUES (?, ?)",
            (gene, json.dumps(data)),
        )


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
        return {"error": f"No UniProt entry found for {gene}"}

    result = results[0]
    genes = result.get("genes", [])
    gene_names = []
    for item in genes:
        gene_names.extend(
            value.get("value")
            for key in ("geneName", "synonyms", "orderedLocusNames", "orfNames")
            for value in ([item.get(key)] if key == "geneName" else item.get(key, []))
            if value and value.get("value")
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


def get_uniprot_context(statement):
    lines = []
    for gene in _candidate_genes(statement):
        data = _read_cache(gene)
        if data is None:
            try:
                data = _fetch_uniprot(gene)
            except Exception:
                continue
            _write_cache(gene, data)

        if data.get("error"):
            continue
        lines.append(
            f"- {data['gene_name']}\n"
            f"  Gene names: {', '.join(data['gene_synonyms'][:12])}\n"
            f"  Protein names: {', '.join(data['protein_names'][:8])}\n"
            f"  Function: {data.get('function') or 'N/A'}"
        )
    return "\n".join(lines) or None
