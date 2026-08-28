import pytest
from fastapi.testclient import TestClient

import repoguide.api.main as main_module
from repoguide.api.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _isolate_indices_dir(tmp_path, monkeypatch):
    """Redirect structural-index output under tmp_path for every test here,
    so nothing ever writes into the project's real data/indices folder."""
    monkeypatch.setattr(main_module, "INDICES_DIR", tmp_path / "indices")


def _make_sample_repo(tmp_path):
    repo_dir = tmp_path / "sample_repo"
    repo_dir.mkdir()

    (repo_dir / "module_a.py").write_text(
        "import os\n"
        "\n"
        "\n"
        "def foo(x):\n"
        "    return x + 1\n",
        encoding="utf-8",
    )
    (repo_dir / "module_b.py").write_text(
        "class Bar:\n"
        "    def method(self):\n"
        "        return 1\n",
        encoding="utf-8",
    )
    return repo_dir


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_valid_repo_returns_populated_counts(tmp_path):
    repo_dir = _make_sample_repo(tmp_path)

    response = client.post("/index", json={"repo_path": str(repo_dir), "repo_id": "sample-repo"})

    assert response.status_code == 200
    body = response.json()

    assert set(body) == {"files_indexed", "chunks_created", "symbols_found", "elapsed_seconds"}

    assert isinstance(body["files_indexed"], int)
    assert isinstance(body["chunks_created"], int)
    assert isinstance(body["symbols_found"], int)
    assert isinstance(body["elapsed_seconds"], float)

    assert body["files_indexed"] == 2
    assert body["chunks_created"] > 0
    assert body["symbols_found"] > 0
    assert body["elapsed_seconds"] > 0


def test_index_nonexistent_repo_path_returns_404_not_500(tmp_path):
    missing_path = tmp_path / "does_not_exist"

    response = client.post("/index", json={"repo_path": str(missing_path), "repo_id": "missing-repo"})

    assert response.status_code == 404
    body = response.json()
    assert "detail" in body
    assert str(missing_path) in body["detail"]


def test_index_repo_path_pointing_at_a_file_is_handled_cleanly(tmp_path):
    file_path = tmp_path / "not_a_directory.py"
    file_path.write_text("x = 1\n", encoding="utf-8")

    response = client.post("/index", json={"repo_path": str(file_path), "repo_id": "file-repo"})

    assert response.status_code == 404
    body = response.json()
    assert "detail" in body
    assert str(file_path) in body["detail"]
