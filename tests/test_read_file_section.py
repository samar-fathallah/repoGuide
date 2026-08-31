from pathlib import Path

import pytest

from repoguide.indexing import repo_registry
from repoguide.tools.read_file_section import (
    InvalidLineRangeError,
    PathEscapesRepoError,
    read_file_section,
)


@pytest.fixture(autouse=True)
def _isolate_registry(tmp_path, monkeypatch):
    """Redirect the repo registry under tmp_path for every test here, so
    nothing ever reads or writes the project's real data/indices folder."""
    monkeypatch.setattr(repo_registry, "DEFAULT_INDICES_DIR", tmp_path / "indices")


def _make_sample_repo(tmp_path) -> Path:
    repo_dir = tmp_path / "sample_repo"
    (repo_dir / "src").mkdir(parents=True)
    (repo_dir / "src" / "app.py").write_text(
        "line one\nline two\nline three\nline four\nline five\n", encoding="utf-8"
    )

    # Sits outside the repo root, used to test path-escape rejection.
    (tmp_path / "outside.txt").write_text("secret\n", encoding="utf-8")

    return repo_dir


def test_read_file_section_returns_expected_lines(tmp_path):
    repo_dir = _make_sample_repo(tmp_path)
    repo_registry.upsert_repo("sample-repo", repo_dir)

    result = read_file_section("sample-repo", "src/app.py", 2, 4)

    assert result == "line two\nline three\nline four\n"


def test_read_file_section_rejects_path_escaping_repo_root(tmp_path):
    repo_dir = _make_sample_repo(tmp_path)
    repo_registry.upsert_repo("sample-repo", repo_dir)

    with pytest.raises(PathEscapesRepoError):
        read_file_section("sample-repo", "../outside.txt", 1, 1)


def test_read_file_section_rejects_absolute_path_escaping_repo_root(tmp_path):
    repo_dir = _make_sample_repo(tmp_path)
    repo_registry.upsert_repo("sample-repo", repo_dir)
    outside_file = tmp_path / "outside.txt"

    with pytest.raises(PathEscapesRepoError):
        read_file_section("sample-repo", str(outside_file), 1, 1)


def test_read_file_section_rejects_nonexistent_file(tmp_path):
    repo_dir = _make_sample_repo(tmp_path)
    repo_registry.upsert_repo("sample-repo", repo_dir)

    with pytest.raises(FileNotFoundError):
        read_file_section("sample-repo", "src/does_not_exist.py", 1, 1)


def test_read_file_section_rejects_invalid_line_ranges(tmp_path):
    repo_dir = _make_sample_repo(tmp_path)
    repo_registry.upsert_repo("sample-repo", repo_dir)

    with pytest.raises(InvalidLineRangeError):
        read_file_section("sample-repo", "src/app.py", 4, 2)  # end < start

    with pytest.raises(InvalidLineRangeError):
        read_file_section("sample-repo", "src/app.py", 0, 1)  # start < 1

    with pytest.raises(InvalidLineRangeError):
        read_file_section("sample-repo", "src/app.py", 100, 100)  # start beyond EOF