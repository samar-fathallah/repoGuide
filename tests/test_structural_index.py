import sqlite3
from pathlib import Path

from repoguide.indexing.structural_index import build_index, extract_structural_data, index_file

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_PATH = FIXTURES_DIR / "structural_sample.py"


def _index_sample():
    return index_file(SAMPLE_PATH, repo_root=FIXTURES_DIR)


def test_definitions_have_correct_symbol_info_and_line_ranges():
    data = _index_sample()
    by_symbol = {d.symbol_name: d for d in data.definitions}

    assert set(by_symbol) == {
        "helper",
        "caller",
        "Service",
        "Service.run",
        "Service.prepare",
        "Service.Config",
        "Service.Config.describe",
        "App",
        "App.handler",
    }

    helper = by_symbol["helper"]
    assert helper.symbol_type == "function"
    assert helper.enclosing_class is None
    assert (helper.start_line, helper.end_line) == (9, 10)

    caller = by_symbol["caller"]
    assert caller.symbol_type == "function"
    assert caller.enclosing_class is None
    assert (caller.start_line, caller.end_line) == (13, 14)

    service = by_symbol["Service"]
    assert service.symbol_type == "class"
    assert service.enclosing_class is None
    # A class's line range spans its full body, including nested definitions.
    assert (service.start_line, service.end_line) == (17, 26)

    run = by_symbol["Service.run"]
    assert run.symbol_type == "method"
    assert run.enclosing_class == "Service"
    assert (run.start_line, run.end_line) == (18, 19)

    prepare = by_symbol["Service.prepare"]
    assert prepare.symbol_type == "method"
    assert prepare.enclosing_class == "Service"
    assert (prepare.start_line, prepare.end_line) == (21, 22)

    nested_class = by_symbol["Service.Config"]
    assert nested_class.symbol_type == "class"
    assert nested_class.enclosing_class == "Service"
    assert (nested_class.start_line, nested_class.end_line) == (24, 26)

    nested_method = by_symbol["Service.Config.describe"]
    assert nested_method.symbol_type == "method"
    assert nested_method.enclosing_class == "Service.Config"
    assert (nested_method.start_line, nested_method.end_line) == (25, 26)

    app = by_symbol["App"]
    assert app.symbol_type == "class"
    assert app.enclosing_class is None
    assert (app.start_line, app.end_line) == (29, 32)

    handler = by_symbol["App.handler"]
    assert handler.symbol_type == "method"
    assert handler.enclosing_class == "App"
    # start_line must include the decorator line (30), not the `def` line (31).
    assert (handler.start_line, handler.end_line) == (30, 32)

    for definition in data.definitions:
        # Labeled relative to repo_root, not the absolute filesystem path
        # the fixture happens to live at -- see repoguide.paths.
        assert definition.file_path == "structural_sample.py"


def test_imports_capture_module_imported_name_alias_and_relative_dots():
    data = _index_sample()
    by_line = {i.line: i for i in data.imports}

    plain_import = by_line[3]
    assert plain_import.module == "os"
    assert plain_import.imported_name is None
    assert plain_import.alias is None

    from_import = by_line[4]
    assert from_import.module == "collections"
    assert from_import.imported_name == "OrderedDict"
    assert from_import.alias is None

    aliased_import = by_line[5]
    assert aliased_import.module == "json"
    assert aliased_import.imported_name is None
    assert aliased_import.alias == "j"

    relative_import = by_line[6]
    assert relative_import.module == "."
    assert relative_import.imported_name == "sibling"
    assert relative_import.alias is None


def test_call_sites_have_correct_callee_and_caller_attribution():
    data = _index_sample()
    by_line = {c.line: c for c in data.calls}

    # `helper(x)` inside the top-level function `caller`.
    top_level_call = by_line[14]
    assert top_level_call.caller_symbol == "caller"
    assert top_level_call.callee_symbol == "helper"

    # `self.prepare()` inside the method `Service.run`.
    method_call = by_line[19]
    assert method_call.caller_symbol == "Service.run"
    assert method_call.callee_symbol == "self.prepare"

    # `caller(1)` at true module level, outside any function.
    module_level_call = by_line[35]
    assert module_level_call.caller_symbol is None
    assert module_level_call.callee_symbol == "caller"


def test_call_inside_decorator_is_attributed_to_enclosing_scope_not_decorated_method():
    data = _index_sample()
    decorator_call = next(c for c in data.calls if c.line == 30)

    assert decorator_call.callee_symbol == "app.route"
    # The decorator on `App.handler` runs while the surrounding class body
    # (`App`) is being executed, not inside `App.handler` itself.
    assert decorator_call.caller_symbol == "App"
    assert decorator_call.caller_symbol != "App.handler"


def test_extract_structural_data_matches_index_file():
    source = SAMPLE_PATH.read_text(encoding="utf-8")
    via_extract = extract_structural_data(source, "structural_sample.py")
    via_index_file = index_file(SAMPLE_PATH, repo_root=FIXTURES_DIR)

    assert via_extract.definitions == via_index_file.definitions
    assert via_extract.imports == via_index_file.imports
    assert via_extract.calls == via_index_file.calls


def test_reindexing_the_same_file_does_not_duplicate_rows(tmp_path):
    db_path = tmp_path / "structural.db"

    build_index([SAMPLE_PATH], db_path)
    conn = sqlite3.connect(db_path)
    counts_after_first_run = {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("definitions", "imports", "calls")
    }
    conn.close()

    assert all(count > 0 for count in counts_after_first_run.values())

    build_index([SAMPLE_PATH], db_path)
    conn = sqlite3.connect(db_path)
    counts_after_second_run = {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("definitions", "imports", "calls")
    }
    conn.close()

    assert counts_after_second_run == counts_after_first_run
