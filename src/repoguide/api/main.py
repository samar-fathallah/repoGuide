"""FastAPI app: health check plus the repository indexing endpoint."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from repoguide.chunking.ast_chunker import chunk_file
from repoguide.indexing.structural_index import build_index

app = FastAPI(
    title="repoGuide",
    description="An agent that answers questions about a Python repository "
    "using semantic retrieval and structural analysis.",
)

INDICES_DIR = Path("data/indices")
REPO_ID_PATTERN = r"^[A-Za-z0-9_-]+$"


class IndexRequest(BaseModel):
    repo_path: str = Field(
        ...,
        description="Local filesystem path to the root of the Python repository to index.",
        examples=["C:/repos/my-project"],
    )
    repo_id: str = Field(
        ...,
        pattern=REPO_ID_PATTERN,
        description="Short identifier for this repository. Used to name its structural "
        "index database file, at data/indices/<repo_id>/structural.db. "
        "Letters, digits, hyphens, and underscores only.",
        examples=["my-project"],
    )


class IndexResponse(BaseModel):
    files_indexed: int = Field(..., description="Number of .py files discovered under repo_path.")
    chunks_created: int = Field(
        ..., description="Total number of AST-aware semantic chunks produced across all files."
    )
    symbols_found: int = Field(
        ...,
        description="Total number of definitions (functions, classes, methods) recorded "
        "in the structural index.",
    )
    elapsed_seconds: float = Field(..., description="Wall-clock time spent indexing, in seconds.")


@app.get("/health")
def health() -> dict:
    """Liveness check. Returns {"status": "ok"} if the service is up."""
    return {"status": "ok"}


@app.post("/index", response_model=IndexResponse)
def index_repository(request: IndexRequest) -> IndexResponse:
    """Index a local Python repository, building both of repoGuide's indices.

    Walks `repo_path` for every `.py` file, then:

    - **Semantic index**: splits each file into AST-aware chunks (one per
      top-level function/class/method) using the Day 1 chunker
      (`repoguide.chunking.ast_chunker`).
    - **Structural index**: extracts definitions, imports, and call sites
      with `repoguide.indexing.structural_index` and persists them to a
      SQLite database at `data/indices/<repo_id>/structural.db`, isolated
      per repository so re-indexing one repo never touches another's data.

    Returns a 404 if `repo_path` does not exist or is not a directory.
    """
    repo_path = Path(request.repo_path)
    if not repo_path.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"repo_path '{request.repo_path}' does not exist or is not a directory",
        )

    start_time = time.perf_counter()

    python_files = sorted(repo_path.rglob("*.py"))

    chunks_created = 0
    for file_path in python_files:
        chunks_created += len(chunk_file(file_path))

    db_path = INDICES_DIR / request.repo_id / "structural.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    build_index(python_files, db_path)

    with sqlite3.connect(db_path) as conn:
        symbols_found = conn.execute("SELECT COUNT(*) FROM definitions").fetchone()[0]

    elapsed_seconds = time.perf_counter() - start_time

    return IndexResponse(
        files_indexed=len(python_files),
        chunks_created=chunks_created,
        symbols_found=symbols_found,
        elapsed_seconds=elapsed_seconds,
    )
