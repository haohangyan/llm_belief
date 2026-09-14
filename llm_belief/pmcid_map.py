"""Dump the PMCID-to-PMID map from the INDRA principal database."""

import argparse
import pickle

from tqdm import tqdm

from llm_belief.locations import PMCID_TO_PMID_PATH


def normalize_pmcid(pmcid):
    pmcid = str(pmcid).upper()
    return pmcid if pmcid.startswith("PMC") else f"PMC{pmcid}"


def dump_pmcid_map(batch_size):
    from indra_db import get_db

    db = get_db("primary")
    if db is None:
        raise RuntimeError("Could not connect to the INDRA principal database")

    query = (
        db.session.query(db.TextRef.pmcid, db.TextRef.pmid)
        .filter(db.TextRef.pmcid.isnot(None))
        .filter(db.TextRef.pmid.isnot(None))
        .yield_per(batch_size)
    )
    mapping = {}
    try:
        for pmcid, pmid in tqdm(query, desc="Dumping PMCID to PMID", unit="ref"):
            mapping[normalize_pmcid(pmcid)] = int(pmid)
    finally:
        db.session.rollback()
        db.session.close()

    PMCID_TO_PMID_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PMCID_TO_PMID_PATH.open("wb") as file:
        pickle.dump(mapping, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[done] mappings={len(mapping):,} output={PMCID_TO_PMID_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args()
    dump_pmcid_map(args.batch_size)


if __name__ == "__main__":
    main()
