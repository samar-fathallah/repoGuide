"""Agent-facing tool: read an exact line range from a file in a repo."""

from __future__ import annotations

from pathlib import Path

from repoguide.indexing import repo_registry


class PathEscapesRepoError(ValueError):
    """Raised when a requested path resolves outside the repo's root."""


class InvalidLineRangeError(ValueError):
    """Raised when start_line/end_line don't describe a valid line range."""


def read_file_section(repo_id: str, path: str, start_line: int, end_line: int) -> str:
    """Return exact source lines start_line..end_line (1-indexed, inclusive)
    from `path`, relative to repo_id's root."""
    repo_root = Path(repo_registry.get_repo_path(repo_id)).resolve()
    target = (repo_root / path).resolve()

    try:
        target.relative_to(repo_root)
    except ValueError:
        raise PathEscapesRepoError(
            f"path '{path}' resolves outside repo '{repo_id}' root '{repo_root}'"
        ) from None

    if not target.is_file():
        raise FileNotFoundError(f"'{path}' does not exist in repo '{repo_id}'")

    if start_line < 1:
        raise InvalidLineRangeError(f"start_line must be >= 1, got {start_line}")
    if end_line < start_line:
        raise InvalidLineRangeError(
            f"end_line ({end_line}) must be >= start_line ({start_line})"
        )

    lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
    if start_line > len(lines):
        raise InvalidLineRangeError(
            f"start_line {start_line} is beyond '{path}''s length ({len(lines)} lines)"
        )

    return "".join(lines[start_line - 1 : end_line])