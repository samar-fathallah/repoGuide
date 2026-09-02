import sqlite3

import pytest

from repoguide.indexing.structural_index import (
    CallSite,
    Definition,
    ImportRecord,
    StructuralData,
    create_schema,
    store_structural_data,
)
from repoguide.tools.find_definition import find_definition
from repoguide.tools.find_usages import find_usages

REPO_ID = "sample-repo"

SAMPLE_DATA = StructuralData(
    definitions=[
        Definition("service.py", "Service", "class", None, 1, 10),
        Definition("service.py", "Service.run", "method", "Service", 2, 3),
        Definition("service.py", "Service.prepare", "method", "Service", 5, 6),
        # A second, unrelated class with a same-named method.
        Definition("other.py", "Other.prepare", "method", "Other", 1, 2),
        # A decoy whose name would falsely suffix-match "get_all" if the
        # "_" in a naive LIKE pattern were left as a wildcard.
        Definition("weird.py", "Weird.getXall", "function", "Weird", 1, 2),
    ],
    imports=[
        ImportRecord("service.py", 1, "os", None, None),
        ImportRecord("service.py", 2, "typing", "Optional", None),
    ],
    calls=[
        # Stored qualified, e.g. from `self.prepare()` inside Service.run.
        CallSite("service.py", 3, "Service.run", "self.prepare"),
        # A module-level call: caller_symbol is None.
        CallSite("other.py", 10, None, "helper"),
        CallSite("service.py", 1, None, "os.path.join"),
    ],
)


@pytest.fixture
def indices_dir(tmp_path):
    db_path = tmp_path / REPO_ID / "structural.db"
    db_path.parent.mkdir(parents=True)

    conn = sqlite3.connect(db_path)
    try:
        create_schema(conn)
        with conn:
            store_structural_data(conn, SAMPLE_DATA)
    finally:
        conn.close()

    return tmp_path


def test_find_definition_exact_match(indices_dir):
    results = find_definition(REPO_ID, "Service", indices_dir=indices_dir)

    assert results == [
        {
            "file_path": "service.py",
            "start_line": 1,
            "end_line": 10,
            "symbol_type": "class",
            "enclosing_class": None,
        }
    ]


def test_find_definition_suffix_match_finds_all_qualified_matches(indices_dir):
    # No definition is literally named "prepare" -- only qualified ones.
    results = find_definition(REPO_ID, "prepare", indices_dir=indices_dir)

    assert len(results) == 2
    pairs = {(r["file_path"], r["enclosing_class"]) for r in results}
    assert pairs == {("service.py", "Service"), ("other.py", "Other")}
    assert all(r["symbol_type"] == "method" for r in results)


def test_find_definition_suffix_match_escapes_underscore_wildcard(indices_dir):
    # "get_all" must not match "Weird.getXall" via SQLite's "_" wildcard.
    results = find_definition(REPO_ID, "get_all", indices_dir=indices_dir)

    assert results == []


def test_find_definition_unknown_symbol_returns_empty_list(indices_dir):
    assert find_definition(REPO_ID, "NoSuchSymbol", indices_dir=indices_dir) == []


def test_find_definition_unindexed_repo_returns_empty_list_without_crashing(tmp_path):
    assert find_definition(REPO_ID, "Service", indices_dir=tmp_path) == []


def test_find_usages_exact_match_call(indices_dir):
    results = find_usages(REPO_ID, "helper", indices_dir=indices_dir)

    assert results == [
        {"file_path": "other.py", "line": 10, "caller_symbol": None, "kind": "call"}
    ]


def test_find_usages_suffix_match_finds_qualified_call_site(indices_dir):
    # Stored as callee_symbol "self.prepare", not "prepare".
    results = find_usages(REPO_ID, "prepare", indices_dir=indices_dir)

    assert results == [
        {
            "file_path": "service.py",
            "line": 3,
            "caller_symbol": "Service.run",
            "kind": "call",
        }
    ]


def test_find_usages_matches_imports_by_imported_name(indices_dir):
    results = find_usages(REPO_ID, "Optional", indices_dir=indices_dir)

    assert results == [
        {"file_path": "service.py", "line": 2, "caller_symbol": None, "kind": "import"}
    ]


def test_find_usages_unknown_symbol_returns_empty_list(indices_dir):
    assert find_usages(REPO_ID, "NoSuchSymbol", indices_dir=indices_dir) == []


def test_find_usages_unindexed_repo_returns_empty_list_without_crashing(tmp_path):
    assert find_usages(REPO_ID, "helper", indices_dir=tmp_path) == []
