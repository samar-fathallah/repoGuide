from repoguide.paths import relative_to_repo_root


def test_relative_to_repo_root_returns_posix_style_relative_path(tmp_path):
    repo_root = tmp_path / "repo"
    nested_file = repo_root / "pkg" / "module.py"

    assert relative_to_repo_root(nested_file, repo_root) == "pkg/module.py"


def test_relative_to_repo_root_handles_file_directly_under_root(tmp_path):
    repo_root = tmp_path / "repo"
    top_level_file = repo_root / "module.py"

    assert relative_to_repo_root(top_level_file, repo_root) == "module.py"


def test_relative_to_repo_root_accepts_plain_string_arguments(tmp_path):
    repo_root = tmp_path / "repo"
    nested_file = repo_root / "a.py"

    assert relative_to_repo_root(str(nested_file), str(repo_root)) == "a.py"
