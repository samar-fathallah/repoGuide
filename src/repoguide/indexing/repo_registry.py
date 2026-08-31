"""Registry of indexed repositories: repo_id -> {repo_path, last_indexed_at}.

Persisted as a single JSON file at `<indices_dir>/repos.json`, so anything
that only knows a repo_id (e.g. an agent tool) can resolve it back to the
repository's filesystem root without re-deriving it from index file
layout.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

DEFAULT_INDICES_DIR = Path("data/indices")
REGISTRY_FILENAME = "repos.json"


class UnknownRepoError(KeyError):
    """Raised when a repo_id has no entry in the repository registry."""


def _resolve_indices_dir(indices_dir: Optional[Union[str, Path]]) -> Path:
    return Path(indices_dir) if indices_dir is not None else DEFAULT_INDICES_DIR


def _registry_path(indices_dir: Optional[Union[str, Path]] = None) -> Path:
    return _resolve_indices_dir(indices_dir) / REGISTRY_FILENAME


def load_registry(indices_dir: Optional[Union[str, Path]] = None) -> dict:
    path = _registry_path(indices_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_registry(registry: dict, indices_dir: Optional[Union[str, Path]] = None) -> None:
    path = _registry_path(indices_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")


def upsert_repo(
    repo_id: str,
    repo_path: Union[str, Path],
    indices_dir: Optional[Union[str, Path]] = None,
) -> None:
    """Record that `repo_id`'s root is `repo_path`, refreshing last_indexed_at."""
    registry = load_registry(indices_dir)
    registry[repo_id] = {
        "repo_path": str(repo_path),
        "last_indexed_at": datetime.now(timezone.utc).isoformat(),
    }
    save_registry(registry, indices_dir)


def get_repo_path(repo_id: str, indices_dir: Optional[Union[str, Path]] = None) -> str:
    """Look up repo_id's root path.

    Raises UnknownRepoError if repo_id has never been indexed.
    """
    registry = load_registry(indices_dir)
    if repo_id not in registry:
        raise UnknownRepoError(f"No indexed repository found for repo_id '{repo_id}'")
    return registry[repo_id]["repo_path"]
