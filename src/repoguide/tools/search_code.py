"""Agent-facing tool: semantic search over a repo's indexed chunks."""

from __future__ import annotations

from typing import List, Tuple

from repoguide.indexing import vector_store

# A result's code is only truncated above this size -- small chunks (the
# common case) are returned intact rather than always paying a truncation
# cost. Either threshold alone is enough to trigger truncation.
TRUNCATION_LINE_THRESHOLD = 20
TRUNCATION_CHAR_THRESHOLD = 800


def search_code(repo_id: str, query: str, k: int = 5) -> List[dict]:
    """Semantic search over repo_id's indexed chunks.

    Chunks larger than ~20 lines / ~800 characters come back as a
    truncated preview (with a "truncated": true flag and a note pointing
    at read_file_section) rather than their full body; smaller chunks are
    returned untouched. start_line/end_line always reflect the full
    chunk regardless of truncation, so a read_file_section follow-up call
    is always possible.
    """
    collection = vector_store.get_or_create_collection(repo_id)
    results = collection.query(query_texts=[query], n_results=k)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    output = []
    for metadata, document, distance in zip(metadatas, documents, distances):
        code, truncated = _maybe_truncate(
            document, metadata["file_path"], metadata["start_line"], metadata["end_line"]
        )
        result = {
            "file_path": metadata["file_path"],
            "start_line": metadata["start_line"],
            "end_line": metadata["end_line"],
            "symbol_name": metadata["symbol_name"],
            "symbol_type": metadata["symbol_type"],
            "is_split": metadata["is_split"],
            "code": code,
            "distance": distance,
        }
        if truncated:
            result["truncated"] = True
        output.append(result)
    return output


def _maybe_truncate(code: str, file_path: str, start_line: int, end_line: int) -> Tuple[str, bool]:
    lines = code.splitlines(keepends=True)
    if len(lines) <= TRUNCATION_LINE_THRESHOLD and len(code) <= TRUNCATION_CHAR_THRESHOLD:
        return code, False

    prefix_lines = lines[:TRUNCATION_LINE_THRESHOLD]
    remaining = len(lines) - len(prefix_lines)
    prefix = "".join(prefix_lines)
    if not prefix.endswith("\n"):
        prefix += "\n"
    note = (
        f'... {remaining} more lines, use read_file_section("{file_path}", '
        f"{start_line}, {end_line}) for the full body.\n"
    )
    return prefix + note, True