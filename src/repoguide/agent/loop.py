"""The agent loop: model + tools, iterating until it calls submit_answer.

Each turn sends the conversation so far to the model along with the tool
schemas from tool_registry. If the model requests a real tool, repo_id is
injected server-side (the model never supplies or sees it), the matching
Python function is called directly, and its result is fed back as the next
message. If the model calls submit_answer instead, its arguments become
the final result and the loop stops. A hard iteration cap prevents an
uncooperative model from looping forever.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import anthropic

from repoguide.agent.tool_registry import ALL_SCHEMAS, SUBMIT_ANSWER_TOOL_NAME, TOOL_FUNCTIONS
import logging

logger = logging.getLogger(__name__)
MODEL_NAME = "claude-sonnet-5"
MAX_TOKENS = 4096
MAX_ITERATIONS = 8

SYSTEM_PROMPT = (
    "You are repoGuide, an assistant that answers questions about a single "
    "Python repository using the tools available to you. Investigate before "
    "answering: search for relevant code, inspect the repo's structure, "
    "look up where symbols are defined or used, and read exact source "
    "sections as needed. Ground every claim in what the tools actually "
    "return -- never guess or fall back on general knowledge about how "
    "similar code 'usually' works. If none of the tools turn up anything "
    "relevant to the question, say clearly that the information isn't in "
    "this repository rather than guessing an answer. When you have enough "
    "information to answer (or have determined the repository doesn't "
    "contain the answer), call submit_answer with your final answer and "
    "the citations that support it -- that is the only way to conclude; "
    "never respond with plain text instead."
)

INCOMPLETE_ANSWER = (
    "I couldn't complete this investigation within the allotted number of "
    "steps. Please try rephrasing the question or narrowing its scope."
)


def run_agent(repo_id: str, question: str) -> Dict[str, Any]:
    """Answer `question` about `repo_id` by running the agent loop.

    Returns a dict with:
      - "answer": str, the final answer text from submit_answer
      - "citations": list of {file_path, start_line, end_line} dicts, from
        submit_answer's citations argument
      - "tool_calls": ordered list of {tool, arguments, result} dicts, one
        per real tool invocation made along the way (submit_answer is the
        conclusion, not an investigation step, so it isn't included here)

    If the model doesn't call submit_answer within MAX_ITERATIONS turns,
    returns a clear "couldn't complete" answer instead of looping forever.
    """
    workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    client = anthropic.Anthropic(
        default_headers={"anthropic-workspace-id": workspace_id} if workspace_id else {},
    )

    messages: List[Dict[str, Any]] = [{"role": "user", "content": question}]
    tool_calls: List[Dict[str, Any]] = []

    for _ in range(MAX_ITERATIONS):
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=ALL_SCHEMAS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_use_blocks = [block for block in response.content if block.type == "tool_use"]

        submit_block = next(
            (block for block in tool_use_blocks if block.name == SUBMIT_ANSWER_TOOL_NAME),
            None,
        )
        if submit_block is not None:
            return {
                "answer": submit_block.input.get("answer", ""),
                "citations": submit_block.input.get("citations", []),
                "tool_calls": tool_calls,
            }

        if not tool_use_blocks:
            # The model responded without invoking any tool; nudge it back
            # toward the only valid way to conclude.
            messages.append(
                {
                    "role": "user",
                    "content": "Please call a tool to continue investigating, or call "
                    "submit_answer to provide your final answer.",
                }
            )
            continue

        tool_results = []
        for block in tool_use_blocks:
            arguments = dict(block.input)
            result = _call_tool(repo_id, block.name, arguments)
            tool_calls.append({"tool": block.name, "arguments": arguments, "result": result})
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": _serialize_result(result),
                }
            )
        messages.append({"role": "user", "content": tool_results})

    return {"answer": INCOMPLETE_ANSWER, "citations": [], "tool_calls": tool_calls}


def _call_tool(repo_id: str, name: str, arguments: Dict[str, Any]) -> Any:
    try:
        function = TOOL_FUNCTIONS[name]
        return function(repo_id=repo_id, **arguments)
    except Exception as exc:  # noqa: BLE001 -- surfaced to the model, not raised
        logger.exception("Tool %r failed with arguments %r", name, arguments)
        return {"error": str(exc)}


def _serialize_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result)
