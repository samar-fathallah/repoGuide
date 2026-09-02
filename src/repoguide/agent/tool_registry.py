"""Anthropic tool-use schemas for repoGuide's agent, plus the mapping from
tool name to the actual Python function each one dispatches to.

`repo_id` is deliberately absent from every input_schema below: the agent
loop (see loop.py) injects it server-side before calling the underlying
function, so the model never sees it and can never choose which repo it
operates on.

submit_answer is a pseudo-tool: it has a schema like the real tools so the
model "calls" it the same way, but the agent loop never dispatches it to a
Python function. Calling it is how the model produces a final, structured
answer instead of free text.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from repoguide.tools.find_definition import find_definition
from repoguide.tools.find_usages import find_usages
from repoguide.tools.get_file_tree import get_file_tree
from repoguide.tools.read_file_section import read_file_section
from repoguide.tools.search_code import search_code

SUBMIT_ANSWER_TOOL_NAME = "submit_answer"

SEARCH_CODE_SCHEMA: Dict[str, Any] = {
    "name": "search_code",
    "description": (
        "Semantic search over the repository's indexed code chunks. Use this "
        "to find code relevant to a natural-language description of "
        "functionality, even when you don't know exact symbol or file names. "
        "Results longer than ~20 lines come back as a truncated preview "
        "(flagged \"truncated\": true, with start_line/end_line still accurate) "
        "rather than the full body -- once you've found the right location, "
        "call read_file_section with those line numbers to get the complete code."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language description of the code you're looking for.",
            },
            "k": {
                "type": "integer",
                "description": "Number of results to return. Defaults to 5.",
            },
        },
        "required": ["query"],
    },
}

GET_FILE_TREE_SCHEMA: Dict[str, Any] = {
    "name": "get_file_tree",
    "description": (
        "Directory structure of the repository, optionally scoped to a "
        "subpath. Use this to orient yourself in the repo's layout. The "
        "default view (no subpath) is shallow: directories more than a few "
        "levels deep are listed with \"truncated\": true instead of being "
        "expanded. Pass subpath with the path of a directory you want to see "
        "more of -- including one that came back truncated -- to drill in "
        "and get a full view starting from there."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "subpath": {
                "type": "string",
                "description": "Path relative to the repo root to scope the tree to. "
                "Omit to list the tree from the repo root.",
            },
        },
        "required": [],
    },
}

FIND_DEFINITION_SCHEMA: Dict[str, Any] = {
    "name": "find_definition",
    "description": (
        "Where a symbol (function, class, or method) is defined. Accepts "
        "either a bare name (e.g. 'prepare') or a fully qualified name "
        "(e.g. 'Service.prepare'); a bare name can return more than one "
        "match if multiple classes define a method with that name."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "The symbol name to look up.",
            },
        },
        "required": ["symbol"],
    },
}

FIND_USAGES_SCHEMA: Dict[str, Any] = {
    "name": "find_usages",
    "description": (
        "Where a symbol is called or imported. Accepts either a bare name "
        "or a fully qualified name, same as find_definition."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "The symbol name to look up call sites and imports for.",
            },
        },
        "required": ["symbol"],
    },
}

READ_FILE_SECTION_SCHEMA: Dict[str, Any] = {
    "name": "read_file_section",
    "description": (
        "Read an exact range of source lines from a specific file in the "
        "repository. Use this to see the full context around a result "
        "surfaced by another tool before citing it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path relative to the repo root.",
            },
            "start_line": {
                "type": "integer",
                "description": "First line to read, 1-indexed, inclusive.",
            },
            "end_line": {
                "type": "integer",
                "description": "Last line to read, 1-indexed, inclusive.",
            },
        },
        "required": ["path", "start_line", "end_line"],
    },
}

SUBMIT_ANSWER_SCHEMA: Dict[str, Any] = {
    "name": SUBMIT_ANSWER_TOOL_NAME,
    "description": (
        "Conclude the investigation and submit the final answer to the "
        "user's question. This is the only way to conclude -- call it "
        "exactly once, when you have enough information to answer (or have "
        "determined the repository doesn't contain the answer), instead of "
        "responding with plain text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The final answer to the user's question, in plain language.",
            },
            "citations": {
                "type": "array",
                "description": "Source locations backing the answer. Empty if none apply, "
                "e.g. when answering that the repository doesn't contain the answer.",
                "items": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "start_line": {"type": "integer"},
                        "end_line": {"type": "integer"},
                    },
                    "required": ["file_path", "start_line", "end_line"],
                },
            },
        },
        "required": ["answer", "citations"],
    },
}

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    SEARCH_CODE_SCHEMA,
    GET_FILE_TREE_SCHEMA,
    FIND_DEFINITION_SCHEMA,
    FIND_USAGES_SCHEMA,
    READ_FILE_SECTION_SCHEMA,
]

ALL_SCHEMAS: List[Dict[str, Any]] = TOOL_SCHEMAS + [SUBMIT_ANSWER_SCHEMA]

# Real tools only -- submit_answer is handled specially by the agent loop
# and is never looked up here.
TOOL_FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "search_code": search_code,
    "get_file_tree": get_file_tree,
    "find_definition": find_definition,
    "find_usages": find_usages,
    "read_file_section": read_file_section,
}
