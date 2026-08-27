"""AST-based source chunking for embedding and retrieval.

Splits a Python source file into one chunk per top-level function, class,
and method (using the standard library `ast` module, per the project's
Python-only-parsing constraint), plus one chunk per contiguous block of
module-level code that sits outside any function or class. Chunks whose
body would be too large for a typical embedding context window are split
further into overlapping sub-chunks.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

FunctionNode = Union[ast.FunctionDef, ast.AsyncFunctionDef]

# Rough heuristic (~4 characters per token) used only to decide when a
# chunk needs to be split further; no tokenizer dependency is required.
CHARS_PER_TOKEN = 4
MAX_CHUNK_TOKENS = 1500
SPLIT_OVERLAP_LINES = 10


@dataclass
class Chunk:
    file_path: str
    start_line: int
    end_line: int
    symbol_name: Optional[str]
    symbol_type: str  # "module" | "class" | "function" | "method"
    class_name: Optional[str]
    text: str
    is_split: bool = False
    part_index: Optional[int] = None
    part_count: Optional[int] = None


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


def chunk_file(file_path: Union[str, Path]) -> List[Chunk]:
    path = Path(file_path)
    source = path.read_text(encoding="utf-8")
    return chunk_source(source, str(file_path))


def chunk_source(source: str, file_path: str) -> List[Chunk]:
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)

    chunks: List[Chunk] = []
    pending_module_stmts: List[ast.stmt] = []

    def flush_module_block() -> None:
        if not pending_module_stmts:
            return
        start_line = _start_line(pending_module_stmts[0])
        end_line = pending_module_stmts[-1].end_lineno
        chunks.append(
            Chunk(
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                symbol_name=None,
                symbol_type="module",
                class_name=None,
                text=_extract(lines, start_line, end_line),
            )
        )
        pending_module_stmts.clear()

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            flush_module_block()
            _chunk_class(node, file_path, lines, [], chunks)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            flush_module_block()
            _chunk_function(node, file_path, lines, [], is_method=False, chunks=chunks)
        else:
            pending_module_stmts.append(node)
    flush_module_block()

    chunks.sort(key=lambda c: c.start_line)
    return chunks


def _start_line(node: ast.AST) -> int:
    decorator_list = getattr(node, "decorator_list", None)
    if decorator_list:
        return min(d.lineno for d in decorator_list)
    return node.lineno


def _extract(lines: List[str], start_line: int, end_line: int) -> str:
    return "".join(lines[start_line - 1 : end_line])


def _qualified_name(class_stack: List[str], name: str) -> str:
    return ".".join(class_stack + [name]) if class_stack else name


def _chunk_class(
    node: ast.ClassDef,
    file_path: str,
    lines: List[str],
    class_stack: List[str],
    chunks: List[Chunk],
) -> None:
    start_line = _start_line(node)
    end_line = node.end_lineno
    chunks.append(
        Chunk(
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            symbol_name=_qualified_name(class_stack, node.name),
            symbol_type="class",
            class_name=".".join(class_stack) or None,
            text=_extract(lines, start_line, end_line),
        )
    )

    child_stack = class_stack + [node.name]
    for child in node.body:
        if isinstance(child, ast.ClassDef):
            _chunk_class(child, file_path, lines, child_stack, chunks)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _chunk_function(child, file_path, lines, child_stack, is_method=True, chunks=chunks)


def _chunk_function(
    node: FunctionNode,
    file_path: str,
    lines: List[str],
    class_stack: List[str],
    is_method: bool,
    chunks: List[Chunk],
) -> None:
    start_line = _start_line(node)
    end_line = node.end_lineno
    text = _extract(lines, start_line, end_line)
    symbol_name = _qualified_name(class_stack, node.name)
    symbol_type = "method" if is_method else "function"
    class_name = ".".join(class_stack) or None

    if estimate_tokens(text) <= MAX_CHUNK_TOKENS:
        chunks.append(
            Chunk(
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                symbol_name=symbol_name,
                symbol_type=symbol_type,
                class_name=class_name,
                text=text,
            )
        )
        return

    func_lines = lines[start_line - 1 : end_line]
    spans = _split_lines_with_overlap(func_lines, MAX_CHUNK_TOKENS, SPLIT_OVERLAP_LINES)
    for index, (span_start, span_end) in enumerate(spans):
        chunks.append(
            Chunk(
                file_path=file_path,
                start_line=start_line + span_start,
                end_line=start_line + span_end,
                symbol_name=symbol_name,
                symbol_type=symbol_type,
                class_name=class_name,
                text="".join(func_lines[span_start : span_end + 1]),
                is_split=True,
                part_index=index + 1,
                part_count=len(spans),
            )
        )


def _split_lines_with_overlap(
    lines: List[str], max_tokens: int, overlap_lines: int
) -> List[Tuple[int, int]]:
    """Return 0-indexed inclusive (start, end) line spans covering `lines`.

    Consecutive spans overlap by `overlap_lines` so that a sub-chunk near a
    split boundary still carries the neighboring context.
    """
    n = len(lines)
    if n == 0:
        return [(0, -1)]

    total_tokens = sum(estimate_tokens(line) for line in lines)
    avg_tokens_per_line = total_tokens / n
    lines_per_chunk = max(1, int(max_tokens / avg_tokens_per_line))

    if lines_per_chunk >= n:
        return [(0, n - 1)]

    step = max(1, lines_per_chunk - overlap_lines)
    spans: List[Tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(start + lines_per_chunk - 1, n - 1)
        spans.append((start, end))
        if end == n - 1:
            break
        start += step
    return spans
