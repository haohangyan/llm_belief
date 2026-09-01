"""Load INDRA statements from the project data files."""

import csv
import gzip
import pickle
import sys

from indra.statements import stmt_from_json
from indra_db.readonly_dumping.util import clean_json_loads


def load_pickle_statements(path):
    with path.open("rb") as file:
        return pickle.load(file)


def iter_tsv_statements(path):
    csv.field_size_limit(sys.maxsize)
    with gzip.open(path, "rt", encoding="utf-8") as file:
        for stmt_hash, stmt_json in csv.reader(file, delimiter="\t"):
            stmt = stmt_from_json(clean_json_loads(stmt_json))
            yield int(stmt_hash), stmt
