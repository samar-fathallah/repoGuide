from pathlib import Path

import pytest

from repoguide.indexing import repo_registry
from repoguide.tools.get_file_tree import SubpathNotFoundError, get_file_tree


@pytest.fixture(autouse=True)
def _isolate_registry(tmp_path, monkeypatch):
    """Redirect the repo registry under tmp_path for every test here, so
    nothing ever reads or writes the project's real data/indices folder."""
    monkeypatch.setattr(repo_registry, "DEFAULT_INDICES_DIR", tmp_path / "indices")


def _make_sample_repo(tmp_path) -> Path:
    repo_dir = tmp_path / "sample_repo"
    (repo_dir / "src").mkdir(parents=True)
    (repo_dir / "src" / "app.py").write_text("print('hi')\n", encoding="utf-8")
    (repo_dir / "src" / "utils.py").write_text("def helper():\n    pass\n", encoding="utf-8")
    (repo_dir / "tests").mkdir()
    (repo_dir / "tests" / "test_app.py").write_text("def test_x():\n    pass\n", encoding="utf-8")
    (repo_dir / "README.md").write_text("# Sample\n", encoding="utf-8")

    # These should be skipped entirely, not just left empty.
    (repo_dir / ".git").mkdir()
    (repo_dir / ".git" / "config").write_text("", encoding="utf-8")
    (repo_dir / "src" / "__pycache__").mkdir()
    (repo_dir / "src" / "__pycache__" / "app.cpython-39.pyc").write_text("", encoding="utf-8")
    (repo_dir / "node_modules" / "some_pkg").mkdir(parents=True)
    (repo_dir / ".venv" / "pyvenv.cfg").parent.mkdir(parents=True)
    (repo_dir / ".venv" / "pyvenv.cfg").write_text("", encoding="utf-8")

    return repo_dir


def test_get_file_tree_returns_expected_structure_and_skips_noise_dirs(tmp_path):
    repo_dir = _make_sample_repo(tmp_path)
    repo_registry.upsert_repo("sample-repo", repo_dir)

    tree = get_file_tree("sample-repo")

    assert tree["name"] == "sample_repo"
    assert tree["type"] == "directory"

    names = {child["name"] for child in tree["children"]}
    assert names == {"src", "tests", "README.md"}
    assert ".git" not in names
    assert "node_modules" not in names
    assert ".venv" not in names

    src = next(c for c in tree["children"] if c["name"] == "src")
    assert src["type"] == "directory"
    src_names = {c["name"] for c in src["children"]}
    assert src_names == {"app.py", "utils.py"}
    assert "__pycache__" not in src_names

    readme = next(c for c in tree["children"] if c["name"] == "README.md")
    assert readme["type"] == "file"
    assert "children" not in readme


def test_get_file_tree_scopes_to_subpath(tmp_path):
    repo_dir = _make_sample_repo(tmp_path)
    repo_registry.upsert_repo("sample-repo", repo_dir)

    tree = get_file_tree("sample-repo", subpath="src")

    assert tree["name"] == "src"
    assert tree["type"] == "directory"
    names = {c["name"] for c in tree["children"]}
    assert names == {"app.py", "utils.py"}


def test_get_file_tree_unknown_repo_id_raises_clear_error():
    with pytest.raises(repo_registry.UnknownRepoError, match="does-not-exist"):
        get_file_tree("does-not-exist")


def test_get_file_tree_nonexistent_subpath_raises_clear_error(tmp_path):
    repo_dir = _make_sample_repo(tmp_path)
    repo_registry.upsert_repo("sample-repo", repo_dir)

    with pytest.raises(SubpathNotFoundError, match="does/not/exist"):
        get_file_tree("sample-repo", subpath="does/not/exist")


def _make_deeply_nested_repo(tmp_path) -> Path:
    repo_dir = tmp_path / "deep_repo"
    deepest = repo_dir / "a" / "b" / "c" / "d"
    deepest.mkdir(parents=True)
    (deepest / "file.py").write_text("x = 1\n", encoding="utf-8")
    return repo_dir


def test_get_file_tree_truncates_directories_past_the_depth_limit(tmp_path):
    repo_dir = _make_deeply_nested_repo(tmp_path)
    repo_registry.upsert_repo("deep-repo", repo_dir)

    tree = get_file_tree("deep-repo")

    # root(0) -> a(1) -> b(2) -> c(3, truncated) -- d and file.py never appear.
    a = next(c for c in tree["children"] if c["name"] == "a")
    assert "truncated" not in a
    b = next(c for c in a["children"] if c["name"] == "b")
    assert "truncated" not in b
    c = next(c for c in b["children"] if c["name"] == "c")
    assert c == {"name": "c", "type": "directory", "truncated": True}
    assert "children" not in c


def test_get_file_tree_subpath_gets_a_fresh_depth_budget_past_a_truncated_directory(tmp_path):
    repo_dir = _make_deeply_nested_repo(tmp_path)
    repo_registry.upsert_repo("deep-repo", repo_dir)

    # "c" showed up truncated from the root; drilling into it directly
    # should expand it fully rather than truncating again immediately.
    tree = get_file_tree("deep-repo", subpath="a/b/c")

    assert tree["name"] == "c"
    assert tree["type"] == "directory"
    d = next(child for child in tree["children"] if child["name"] == "d")
    assert d["type"] == "directory"
    assert "truncated" not in d
    file_names = {child["name"] for child in d["children"]}
    assert file_names == {"file.py"}