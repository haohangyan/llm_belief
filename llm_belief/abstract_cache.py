"""Persistent abstract cache populated from INDRA's principal database."""
# python -m llm_belief.abstract_cache --batch-size 100000
import argparse
import json
import sqlite3
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from llm_belief.locations import ABSTRACT_CACHE_PATH


def _connect():
    connection = sqlite3.connect(ABSTRACT_CACHE_PATH)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS abstracts (
            pmid INTEGER PRIMARY KEY,
            abstract TEXT NOT NULL,
            source TEXT NOT NULL,
            source_priority INTEGER NOT NULL,
            text_content_id INTEGER
        )
    """)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    return connection


def load_abstracts(pmids):
    pmids = {int(pmid) for pmid in pmids if pmid}
    if not pmids or not ABSTRACT_CACHE_PATH.exists():
        return {}

    with _connect() as connection:
        connection.execute("CREATE TEMP TABLE wanted_pmids (pmid INTEGER PRIMARY KEY)")
        connection.executemany(
            "INSERT INTO wanted_pmids VALUES (?)", ((pmid,) for pmid in pmids)
        )
        rows = connection.execute("""
            SELECT abstracts.pmid, abstracts.abstract
            FROM abstracts JOIN wanted_pmids USING (pmid)
        """)
        return dict(rows)


def _fetch_pubmed(pmids):
    pmids = sorted(pmids)
    abstracts = {}
    queried = set()
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    total_batches = (len(pmids) + 199) // 200

    for start in range(0, len(pmids), 200):
        batch = pmids[start : start + 200]
        batch_number = start // 200 + 1
        print(
            f"[abstract] fetching PubMed batch {batch_number}/{total_batches} "
            f"({len(batch)} PMIDs)",
            flush=True,
        )
        data = urlencode(
            {
                "db": "pubmed",
                "id": ",".join(map(str, batch)),
                "retmode": "xml",
                "rettype": "abstract",
                "tool": "llm_belief",
            }
        ).encode()
        request = Request(url, data=data, headers={"User-Agent": "llm-belief/0.1"})
        try:
            with urlopen(request, timeout=30) as response:
                root = ElementTree.parse(response).getroot()
        except Exception as error:
            print(f"PubMed abstract fetch failed: {error}")
            continue
        queried.update(batch)

        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//MedlineCitation/PMID")
            sections = []
            for node in article.findall(".//Article/Abstract/AbstractText"):
                text = "".join(node.itertext()).strip()
                label = node.get("Label")
                if text:
                    sections.append(f"{label}: {text}" if label else text)
            if pmid and sections:
                abstracts[int(pmid)] = "\n".join(sections)

        if start + 200 < len(pmids):
            time.sleep(0.34)

    return abstracts, queried


def get_abstracts(pmids):
    """Read requested abstracts and download only cache misses from PubMed."""
    pmids = {int(pmid) for pmid in pmids if pmid}
    print(
        f"[abstract] checking {len(pmids)} PMIDs in {ABSTRACT_CACHE_PATH}",
        flush=True,
    )
    cached = load_abstracts(pmids)
    missing = pmids - set(cached)
    print(
        f"[abstract] cached={len(cached)} missing={len(missing)}",
        flush=True,
    )
    downloaded, queried = _fetch_pubmed(missing)

    if queried:
        with _connect() as connection:
            connection.executemany(
                """
                INSERT INTO abstracts
                    (pmid, abstract, source, source_priority, text_content_id)
                VALUES (?, ?, 'pubmed_eutils', 0, NULL)
                ON CONFLICT(pmid) DO UPDATE SET
                    abstract = excluded.abstract,
                    source = excluded.source,
                    source_priority = excluded.source_priority
                """,
                ((pmid, downloaded.get(pmid, "")) for pmid in queried),
            )

    abstracts = dict(cached)
    abstracts.update((pmid, downloaded.get(pmid, "")) for pmid in queried)
    return abstracts, {
        "requested_pmids": len(pmids),
        "already_cached": len(cached),
        "downloaded": len(downloaded),
        "missing": len(pmids) - sum(bool(abstracts.get(pmid)) for pmid in pmids),
        "request_failed": len(missing - queried),
    }


def _source_priority(source):
    if source == "pubmed":
        return 0
    if source == "cord19_abstract":
        return 1
    return 2


def export_principal_abstracts(batch_size=1000):
    """Stream all abstracts from INDRA's principal DB into the local cache."""
    from indra_db import get_db
    from indra_db.util import unpack

    db = get_db("primary")
    if db is None:
        raise RuntimeError("Could not connect to the INDRA principal database")

    with _connect() as cache:
        row = cache.execute(
            "SELECT value FROM metadata WHERE key = 'last_text_content_id'"
        ).fetchone()
        last_id = int(row[0]) if row else 0

    query = (
        db.session.query(
            db.TextContent.id,
            db.TextRef.pmid,
            db.TextContent.source,
            db.TextContent.content,
        )
        .join(db.TextRef, db.TextContent.text_ref_id == db.TextRef.id)
        .filter(db.TextContent.text_type == "abstract")
        .filter(db.TextContent.id > last_id)
        .filter(db.TextRef.pmid.isnot(None))
        .order_by(db.TextContent.id)
        .execution_options(stream_results=True)
        .yield_per(batch_size)
    )

    processed = 0
    batch = []
    try:
        for text_content_id, pmid, source, content in query:
            try:
                abstract = unpack(content).strip()
            except Exception:
                continue
            batch.append(
                (
                    int(pmid),
                    abstract,
                    source,
                    _source_priority(source),
                    text_content_id,
                )
            )
            if len(batch) >= batch_size:
                _write_principal_batch(batch)
                processed += len(batch)
                last_id = batch[-1][-1]
                print(f"cached={processed} last_text_content_id={last_id}")
                batch = []

        if batch:
            _write_principal_batch(batch)
            processed += len(batch)
            last_id = batch[-1][-1]
    finally:
        db.session.rollback()
        db.session.close()

    with _connect() as cache:
        total = cache.execute(
            "SELECT COUNT(*) FROM abstracts WHERE abstract != ''"
        ).fetchone()[0]
    print(
        json.dumps(
            {
                "processed": processed,
                "cached_abstracts": total,
                "last_text_content_id": last_id,
            },
            indent=2,
        )
    )


def _write_principal_batch(rows):
    with _connect() as connection:
        connection.executemany(
            """
            INSERT INTO abstracts
                (pmid, abstract, source, source_priority, text_content_id)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(pmid) DO UPDATE SET
                abstract = excluded.abstract,
                source = excluded.source,
                source_priority = excluded.source_priority,
                text_content_id = excluded.text_content_id
            WHERE excluded.source_priority < abstracts.source_priority
               OR abstracts.abstract = ''
               OR (excluded.source_priority = abstracts.source_priority
                   AND excluded.text_content_id > abstracts.text_content_id)
            """,
            rows,
        )
        connection.execute(
            """
            INSERT INTO metadata (key, value)
            VALUES ('last_text_content_id', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (str(rows[-1][-1]),),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()
    export_principal_abstracts(args.batch_size)


if __name__ == "__main__":
    main()
