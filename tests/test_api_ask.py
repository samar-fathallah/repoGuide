from unittest.mock import patch

from fastapi.testclient import TestClient

import repoguide.api.main as main
from repoguide.indexing.repo_registry import UnknownRepoError

client = TestClient(main.app)

FAKE_AGENT_RESULT = {
    "answer": "Service is defined in service.py.",
    "citations": [{"file_path": "service.py", "start_line": 1, "end_line": 10}],
    "tool_calls": [
        {
            "tool": "find_definition",
            "arguments": {"symbol": "Service"},
            "result": [
                {
                    "file_path": "service.py",
                    "start_line": 1,
                    "end_line": 10,
                    "symbol_type": "class",
                    "enclosing_class": None,
                }
            ],
        },
        {
            "tool": "read_file_section",
            "arguments": {"path": "service.py", "start_line": 1, "end_line": 10},
            "result": "class Service:\n    ...\n",
        },
    ],
}


def test_ask_valid_repo_returns_200_matching_ask_response_schema():
    with patch.object(
        main, "run_agent", return_value=FAKE_AGENT_RESULT
    ) as mock_run_agent, patch.object(
        main.repo_registry, "get_repo_path", return_value="/some/repo"
    ) as mock_get_repo_path:
        response = client.post(
            "/ask", json={"question": "Where is Service defined?", "repo_id": "sample-repo"}
        )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"answer", "citations", "tool_calls"}
    assert body == FAKE_AGENT_RESULT

    mock_get_repo_path.assert_called_once_with("sample-repo")
    mock_run_agent.assert_called_once_with("sample-repo", "Where is Service defined?")


def test_ask_unknown_repo_id_returns_404_and_never_calls_run_agent():
    with patch.object(
        main.repo_registry,
        "get_repo_path",
        side_effect=UnknownRepoError("No indexed repository found for repo_id 'ghost-repo'"),
    ) as mock_get_repo_path, patch.object(main, "run_agent") as mock_run_agent:
        response = client.post("/ask", json={"question": "anything", "repo_id": "ghost-repo"})

    assert response.status_code == 404
    assert response.json() == {"detail": "No indexed repository found for repo_id 'ghost-repo'"}

    mock_get_repo_path.assert_called_once_with("ghost-repo")
    mock_run_agent.assert_not_called()


def test_ask_blank_question_returns_422_without_registry_lookup_or_run_agent():
    with patch.object(main.repo_registry, "get_repo_path") as mock_get_repo_path, patch.object(
        main, "run_agent"
    ) as mock_run_agent:
        response = client.post("/ask", json={"question": "   ", "repo_id": "sample-repo"})

    assert response.status_code == 422
    mock_get_repo_path.assert_not_called()
    mock_run_agent.assert_not_called()


def test_ask_empty_string_question_returns_422():
    with patch.object(main.repo_registry, "get_repo_path") as mock_get_repo_path, patch.object(
        main, "run_agent"
    ) as mock_run_agent:
        response = client.post("/ask", json={"question": "", "repo_id": "sample-repo"})

    assert response.status_code == 422
    mock_get_repo_path.assert_not_called()
    mock_run_agent.assert_not_called()


def test_ask_preserves_tool_calls_order_and_arguments_unmodified():
    with patch.object(main, "run_agent", return_value=FAKE_AGENT_RESULT), patch.object(
        main.repo_registry, "get_repo_path", return_value="/some/repo"
    ):
        response = client.post(
            "/ask", json={"question": "Where is Service defined?", "repo_id": "sample-repo"}
        )

    body = response.json()
    assert body["tool_calls"] == FAKE_AGENT_RESULT["tool_calls"]
    assert [call["tool"] for call in body["tool_calls"]] == ["find_definition", "read_file_section"]
