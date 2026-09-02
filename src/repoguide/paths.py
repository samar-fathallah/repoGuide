"""Shared path utility used by both indices (semantic and structural).

Both the AST chunker and the structural indexer label each chunk/
definition/import/call with a `file_path`. That label needs to be
relative to the repo root, not the absolute filesystem path under which
the repo happened to be indexed -- an absolute path leaks local machine
structure into API responses and won't resolve if the same repo is
indexed again from a different machine, CI, or a container.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union


def relative_to_repo_root(file_path: Union[str, Path], repo_root: Union[str, Path]) -> str:
    """POSIX-style path of `file_path` relative to `repo_root`.

    Both sides are resolved first (matching the pattern already used in
    repoguide.tools.read_file_section), so this is robust to `repo_root`
    or `file_path` not already being in canonical form. `.as_posix()`
    normalizes Windows backslashes to forward slashes, so the same repo
    indexed on Windows or Linux stores identical path strings.
    """
    resolved_file = Path(file_path).resolve()
    resolved_root = Path(repo_root).resolve()
    return resolved_file.relative_to(resolved_root).as_posix()
