"""Agent-facing tool: where a symbol is defined."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Union

DEFAULT_INDICES_DIR = Path("data/indices")


def find_definition(
    repo_id: str, symbol: str, indices_dir: Union[str, Path] = DEFAULT_INDICES_DIR
) -> list[dict]:
    """Where a symbol is defined.

    Tries an exact match on symbol_name first (e.g. "Service.prepare"); if
    that finds nothing, falls back to symbol_name ending in "." + symbol
    (e.g. searching "prepare" also matches "Service.prepare"), so callers
    don't need to already know the fully qualified name.

    Returns a list of dicts (there can be more than one match, e.g. two
    unrelated classes with a same-named method), or an empty list if
    nothing matches or repo_id has never been structurally indexed.
    """
    db_path = Path(indices_dir) / repo_id / "structural.db"
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT file_path, start_line, end_line, symbol_type, enclosing_class
            FROM definitions
            WHERE symbol_name = ?
            ORDER BY file_path, start_line
            """,
            (symbol,),
        ).fetchall()

        if not rows:
            rows = conn.execute(
                """
                SELECT file_path, start_line, end_line, symbol_type, enclosing_class
                FROM definitions
                WHERE symbol_name LIKE ? ESCAPE '\\'
                ORDER BY file_path, start_line
                """,
                (f"%.{_escape_like(symbol)}",),
            ).fetchall()
    finally:
        conn.close()

    return [
        {
            "file_path": file_path,
            "start_line": start_line,
            "end_line": end_line,
            "symbol_type": symbol_type,
            "enclosing_class": enclosing_class,
        }
        for file_path, start_line, end_line, symbol_type, enclosing_class in rows
    ]


def _escape_like(value: str) -> str:
    """Escape SQLite LIKE wildcards so a symbol containing `%` or `_`
    (e.g. "get_all") is matched literally rather than as a pattern."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
