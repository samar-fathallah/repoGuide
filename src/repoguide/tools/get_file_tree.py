"""Agent-facing tool: directory structure for an indexed repository."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from repoguide.indexing import repo_registry

SKIPPED_DIR_NAMES = {".git", "__pycache__", "venv", ".venv", "node_modules"}

# Directories nested deeper than this beneath the requested target (repo
# root, or the given subpath) are listed but not expanded further, to
# avoid an unbounded whole-repo dump on large repositories. Depth is
# counted fresh from whatever target is requested, not from the repo's
# true root, so passing subpath to drill into a directory that showed up
# truncated gets a full few levels of its own rather than being
# immediately truncated again.
MAX_TREE_DEPTH = 3


class SubpathNotFoundError(FileNotFoundError):
    """Raised when a requested subpath doesn't exist under a repo's root."""


def get_file_tree(repo_id: str, subpath: Optional[str] = None) -> dict:
    """Directory structure for repo_id, optionally scoped to subpath."""
    repo_root = Path(repo_registry.get_repo_path(repo_id))
    target = (repo_root / subpath) if subpath else repo_root

    if not target.exists():
        raise SubpathNotFoundError(
            f"subpath '{subpath}' does not exist under repo '{repo_id}' (root: '{repo_root}')"
        )

    return _build_tree(target, depth=0)


def _build_tree(path: Path, depth: int) -> dict:
    if not path.is_dir():
        return {"name": path.name, "type": "file"}

    if depth >= MAX_TREE_DEPTH:
        return {"name": path.name, "type": "directory", "truncated": True}

    children: List[dict] = []
    for entry in sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
        if entry.is_dir() and entry.name in SKIPPED_DIR_NAMES:
            continue
        children.append(_build_tree(entry, depth + 1))

    return {"name": path.name, "type": "directory", "children": children}