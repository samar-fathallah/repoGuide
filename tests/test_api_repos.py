from unittest.mock import patch

from fastapi.testclient import TestClient

import repoguide.api.main as main
from repoguide.indexing.repo_registry import UnknownRepoError
from repoguide.tools.get_file_tree import SubpathNotFoundError

client = TestClient(main.app)

FIXED_REGISTRY = {
    "sample-repo": {
        "repo_path": "/repos/sample",
        "last_indexed_at": "2026-09-02T00:00:00+00:00",
    },
    "other-repo": {
        "repo_path": "/repos/other",
        "last_indexed_at": "2026-08-30T12:34:56+00:00",
    },
}

FIXED_TREE = {
    "name": "sample-repo",
    "type": "directory",
    "children": [
        {"name": "service.py", "type": "file"},
        {"name": "tests", "type": "directory", "children": []},
    ],
}


def test_list_repos_returns_registry_contents_as_a_list():
    with patch.object(main.repo_registry, "load_registry", return_value=FIXED_REGISTRY):
        response = client.get("/repos")

    assert response.status_code == 200
    assert response.json() == [
        {
            "repo_id": "sample-repo",
            "repo_path": "/repos/sample",
            "last_indexed_at": "2026-09-02T00:00:00+00:00",
        },
        {
            "repo_id": "other-repo",
            "repo_path": "/repos/other",
            "last_indexed_at": "2026-08-30T12:34:56+00:00",
        },
    ]


def test_list_repos_empty_registry_returns_200_with_empty_list():
    with patch.object(main.repo_registry, "load_registry", return_value={}):
        response = client.get("/repos")

    assert response.status_code == 200
    assert response.json() == []


def test_repo_tree_returns_get_file_tree_result_unmodified():
    with patch.object(main, "get_file_tree", return_value=FIXED_TREE) as mock_get_file_tree:
        response = client.get("/repos/sample-repo/tree")

    assert response.status_code == 200
    assert response.json() == FIXED_TREE
    mock_get_file_tree.assert_called_once_with("sample-repo", subpath=None)


def test_repo_tree_forwards_path_query_param_as_subpath():
    with patch.object(main, "get_file_tree", return_value=FIXED_TREE) as mock_get_file_tree:
        response = client.get("/repos/sample-repo/tree", params={"path": "tests"})

    assert response.status_code == 200
    mock_get_file_tree.assert_called_once_with("sample-repo", subpath="tests")


def test_repo_tree_unknown_repo_returns_404_with_clean_message():
    with patch.object(
        main,
        "get_file_tree",
        side_effect=UnknownRepoError("No indexed repository found for repo_id 'ghost-repo'"),
    ):
        response = client.get("/repos/ghost-repo/tree")

    assert response.status_code == 404
    # Regression test: UnknownRepoError subclasses KeyError, whose __str__
    # re-wraps the message in repr(). The detail must be the clean message,
    # not '"No indexed repository found for repo_id \'ghost-repo\'"'.
    assert response.json() == {"detail": "No indexed repository found for repo_id 'ghost-repo'"}


def test_repo_tree_unknown_subpath_returns_404_with_clean_message():
    with patch.object(
        main,
        "get_file_tree",
        side_effect=SubpathNotFoundError(
            "subpath 'nope' does not exist under repo 'sample-repo' (root: '/repos/sample')"
        ),
    ):
        response = client.get("/repos/sample-repo/tree", params={"path": "nope"})

    assert response.status_code == 404
    assert response.json() == {
        "detail": "subpath 'nope' does not exist under repo 'sample-repo' (root: '/repos/sample')"
    }
