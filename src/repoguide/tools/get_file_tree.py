"""Agent-facing tool: directory structure for an indexed repository."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from repoguide.indexing import repo_registry

SKIPPED_DIR_NAMES = {".git", "__pycache__", "venv", ".venv", "node_modules"}


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

    return _build_tree(target)


def _build_tree(path: Path) -> dict:
    if not path.is_dir():
        return {"name": path.name, "type": "file"}

    children: List[dict] = []
    for entry in sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
        if entry.is_dir() and entry.name in SKIPPED_DIR_NAMES:
            continue
        children.append(_build_tree(entry))

    return {"name": path.name, "type": "directory", "children": children}