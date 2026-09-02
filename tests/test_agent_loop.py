"""Tests for the agent loop, with the Anthropic client entirely mocked out.

No real API calls are made: the Anthropic client's `.messages.create` is
replaced with a fake that returns pre-built, canned responses in sequence,
and the real tool functions in TOOL_FUNCTIONS are swapped for lightweight
stand-ins so the tests don't depend on a live indexed repository.
"""

from types import SimpleNamespace

from anthropic.types import ToolUseBlock

import repoguide.agent.loop as loop_module


class FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0

    def create(self, **kwargs):
        response = self._responses[self.call_count]
        self.call_count += 1
        return response


class FakeAnthropicClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


def _tool_use(name, input_, id_):
    return ToolUseBlock(type="tool_use", id=id_, name=name, input=input_)


def _fake_response(*blocks):
    return SimpleNamespace(content=list(blocks))


def _install_fake_client(monkeypatch, responses):
    fake_client = FakeAnthropicClient(responses)
    monkeypatch.setattr(loop_module.anthropic, "Anthropic", lambda **kwargs: fake_client)
    return fake_client


def test_single_tool_call_then_submit_answer(monkeypatch):
    responses = [
        _fake_response(_tool_use("find_definition", {"symbol": "Service"}, "call_1")),
        _fake_response(
            _tool_use(
                "submit_answer",
                {
                    "answer": "Service is defined in service.py.",
                    "citations": [{"file_path": "service.py", "start_line": 1, "end_line": 10}],
                },
                "call_2",
            )
        ),
    ]
    _install_fake_client(monkeypatch, responses)

    fake_definition_result = [
        {
            "file_path": "service.py",
            "start_line": 1,
            "end_line": 10,
            "symbol_type": "class",
            "enclosing_class": None,
        }
    ]
    monkeypatch.setitem(
        loop_module.TOOL_FUNCTIONS,
        "find_definition",
        lambda repo_id, symbol: fake_definition_result,
    )

    result = loop_module.run_agent("repo-1", "Where is Service defined?")

    assert result["answer"] == "Service is defined in service.py."
    assert result["citations"] == [{"file_path": "service.py", "start_line": 1, "end_line": 10}]
    assert result["tool_calls"] == [
        {
            "tool": "find_definition",
            "arguments": {"symbol": "Service"},
            "result": fake_definition_result,
        }
    ]


def test_multiple_tool_calls_in_sequence_then_submit_answer(monkeypatch):
    responses = [
        _fake_response(_tool_use("search_code", {"query": "parse config"}, "call_1")),
        _fake_response(
            _tool_use(
                "read_file_section",
                {"path": "config.py", "start_line": 1, "end_line": 20},
                "call_2",
            )
        ),
        _fake_response(
            _tool_use(
                "submit_answer",
                {
                    "answer": "Config parsing happens in config.py.",
                    "citations": [{"file_path": "config.py", "start_line": 1, "end_line": 20}],
                },
                "call_3",
            )
        ),
    ]
    _install_fake_client(monkeypatch, responses)

    search_result = [
        {
            "file_path": "config.py",
            "start_line": 1,
            "end_line": 20,
            "symbol_name": "load_config",
            "symbol_type": "function",
            "is_split": False,
            "code": "def load_config():\n    ...\n",
            "distance": 0.1,
        }
    ]
    section_result = "def load_config():\n    ...\n"

    monkeypatch.setitem(
        loop_module.TOOL_FUNCTIONS, "search_code", lambda repo_id, query, k=5: search_result
    )
    monkeypatch.setitem(
        loop_module.TOOL_FUNCTIONS,
        "read_file_section",
        lambda repo_id, path, start_line, end_line: section_result,
    )

    result = loop_module.run_agent("repo-1", "Where does config parsing happen?")

    # Order and arguments of the trace must match the sequence of calls made.
    assert result["tool_calls"] == [
        {
            "tool": "search_code",
            "arguments": {"query": "parse config"},
            "result": search_result,
        },
        {
            "tool": "read_file_section",
            "arguments": {"path": "config.py", "start_line": 1, "end_line": 20},
            "result": section_result,
        },
    ]
    assert result["answer"] == "Config parsing happens in config.py."
    assert result["citations"] == [{"file_path": "config.py", "start_line": 1, "end_line": 20}]


def test_submit_answer_citations_propagate_to_return_value(monkeypatch):
    citations = [
        {"file_path": "a.py", "start_line": 1, "end_line": 5},
        {"file_path": "b.py", "start_line": 10, "end_line": 12},
    ]
    responses = [
        _fake_response(
            _tool_use(
                "submit_answer",
                {"answer": "Two files are involved.", "citations": citations},
                "call_1",
            )
        )
    ]
    _install_fake_client(monkeypatch, responses)

    result = loop_module.run_agent("repo-1", "Which files are involved?")

    assert result["answer"] == "Two files are involved."
    assert result["citations"] == citations
    assert result["tool_calls"] == []


def test_exceeding_max_iterations_returns_incomplete_answer_without_looping_forever(monkeypatch):
    # The model calls a real tool on every single turn and never submits.
    responses = [
        _fake_response(_tool_use("search_code", {"query": f"query {i}"}, f"call_{i}"))
        for i in range(loop_module.MAX_ITERATIONS)
    ]
    fake_client = _install_fake_client(monkeypatch, responses)
    monkeypatch.setitem(loop_module.TOOL_FUNCTIONS, "search_code", lambda repo_id, query, k=5: [])

    result = loop_module.run_agent("repo-1", "Anything?")

    assert result["answer"] == loop_module.INCOMPLETE_ANSWER
    assert result["citations"] == []
    assert len(result["tool_calls"]) == loop_module.MAX_ITERATIONS
    assert fake_client.messages.call_count == loop_module.MAX_ITERATIONS
