"""Agent-facing tool: where a symbol is called or imported."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Union

DEFAULT_INDICES_DIR = Path("data/indices")


def find_usages(
    repo_id: str, symbol: str, indices_dir: Union[str, Path] = DEFAULT_INDICES_DIR
) -> list[dict]:
    """Where a symbol is called or imported.

    Calls are matched against callee_symbol using the same exact-then-
    suffix strategy as find_definition: a call like `self.prepare()` is
    stored as callee_symbol "self.prepare", not "prepare", so a bare
    search for "prepare" needs the suffix fallback ("...ending in
    '.prepare'") to find it. Imports are matched by exact imported_name.

    Returns a list of dicts (there can be more than one match), or an
    empty list if nothing matches or repo_id has never been structurally
    indexed.
    """
    db_path = Path(indices_dir) / repo_id / "structural.db"
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    try:
        call_rows = conn.execute(
            """
            SELECT file_path, line, caller_symbol
            FROM calls
            WHERE callee_symbol = ?
            ORDER BY file_path, line
            """,
            (symbol,),
        ).fetchall()

        if not call_rows:
            call_rows = conn.execute(
                """
                SELECT file_path, line, caller_symbol
                FROM calls
                WHERE callee_symbol LIKE ? ESCAPE '\\'
                ORDER BY file_path, line
                """,
                (f"%.{_escape_like(symbol)}",),
            ).fetchall()

        import_rows = conn.execute(
            """
            SELECT file_path, line
            FROM imports
            WHERE imported_name = ?
            ORDER BY file_path, line
            """,
            (symbol,),
        ).fetchall()
    finally:
        conn.close()

    usages = [
        {"file_path": file_path, "line": line, "caller_symbol": caller_symbol, "kind": "call"}
        for file_path, line, caller_symbol in call_rows
    ]
    usages.extend(
        {"file_path": file_path, "line": line, "caller_symbol": None, "kind": "import"}
        for file_path, line in import_rows
    )
    return usages


def _escape_like(value: str) -> str:
    """Escape SQLite LIKE wildcards so a symbol containing `%` or `_`
    (e.g. "get_all") is matched literally rather than as a pattern."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
