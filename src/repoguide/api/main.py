"""FastAPI app: health check, repository indexing, and the ask endpoint."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from repoguide.agent.loop import run_agent
from repoguide.chunking.ast_chunker import chunk_file
from repoguide.indexing import repo_registry
from repoguide.indexing.repo_registry import UnknownRepoError
from repoguide.indexing.structural_index import build_index
from repoguide.indexing.vector_store import add_chunks, get_or_create_collection
from repoguide.tools.get_file_tree import SubpathNotFoundError, get_file_tree

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


class AskRequest(BaseModel):
    question: str = Field(..., description="Natural-language question about the repository.")
    repo_id: str = Field(
        ..., description="Identifier of a previously indexed repository (see POST /index)."
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty or whitespace-only")
        return value


class Citation(BaseModel):
    file_path: str
    start_line: int
    end_line: int


class ToolCall(BaseModel):
    tool: str
    arguments: dict
    result: Any


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    tool_calls: list[ToolCall]


class RepoSummary(BaseModel):
    repo_id: str
    repo_path: str
    last_indexed_at: str


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
      (`repoguide.chunking.ast_chunker`), embeds them with
      `repoguide.indexing.vector_store`, and stores them in a persistent
      Chroma collection at `data/indices/<repo_id>/chroma/`, isolated per
      repository.
    - **Structural index**: extracts definitions, imports, and call sites
      with `repoguide.indexing.structural_index` and persists them to a
      SQLite database at `data/indices/<repo_id>/structural.db`, isolated
      per repository so re-indexing one repo never touches another's data.

    Also records/refreshes this repo's entry (repo_path, last_indexed_at)
    in the repo registry at `data/indices/repos.json`
    (`repoguide.indexing.repo_registry`), so tools that only know a
    repo_id can resolve it back to a filesystem path.

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

    collection = get_or_create_collection(request.repo_id, indices_dir=INDICES_DIR)
    chunks_created = 0
    for file_path in python_files:
        chunks = chunk_file(file_path)
        add_chunks(collection, str(file_path), chunks)
        chunks_created += len(chunks)

    db_path = INDICES_DIR / request.repo_id / "structural.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    build_index(python_files, db_path)

    with sqlite3.connect(db_path) as conn:
        symbols_found = conn.execute("SELECT COUNT(*) FROM definitions").fetchone()[0]

    repo_registry.upsert_repo(request.repo_id, repo_path, indices_dir=INDICES_DIR)

    elapsed_seconds = time.perf_counter() - start_time

    return IndexResponse(
        files_indexed=len(python_files),
        chunks_created=chunks_created,
        symbols_found=symbols_found,
        elapsed_seconds=elapsed_seconds,
    )


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    """Answer a natural-language question about a previously indexed repository.

    Resolves `repo_id` through the repo registry first -- the same
    `repo_registry.get_repo_path` lookup `get_file_tree` and
    `read_file_section` already use -- so an unindexed repo_id fails fast
    with a 404 instead of spending a model call on it. Once that resolves,
    the question is handed to the agent loop
    (`repoguide.agent.loop.run_agent`), which investigates the repository
    with its tools and returns a structured answer, its supporting
    citations, and the full ordered tool-call trace.

    Returns a 404 if repo_id was never indexed via POST /index.
    """
    try:
        repo_registry.get_repo_path(request.repo_id)
    except UnknownRepoError as exc:
        # UnknownRepoError subclasses KeyError, whose __str__ re-wraps the
        # message in repr() (e.g. '"message"' with literal quotes) -- use
        # the original arg directly so the 404 body stays human-readable.
        raise HTTPException(status_code=404, detail=exc.args[0]) from exc

    result = run_agent(request.repo_id, request.question)
    return AskResponse(**result)


@app.get("/repos", response_model=list[RepoSummary])
def list_repos() -> list[RepoSummary]:
    """All repositories currently in the repo registry (see POST /index).

    Reads the same registry `repo_registry.get_repo_path` resolves repo_id
    against, just returned as a list instead of the raw {repo_id: {...}}
    dict. Returns an empty list if nothing has been indexed yet -- that's
    a normal state, not an error.
    """
    registry = repo_registry.load_registry()
    return [RepoSummary(repo_id=repo_id, **entry) for repo_id, entry in registry.items()]


@app.get("/repos/{repo_id}/tree")
def repo_tree(repo_id: str, path: Optional[str] = None) -> dict:
    """Directory structure of an indexed repository, optionally scoped to a subdirectory.

    Delegates entirely to the get_file_tree tool -- the same function the
    agent calls internally -- so this route returns exactly what that tool
    returns, with one definition of "file tree" in the codebase. `path`,
    if given, is forwarded as get_file_tree's `subpath` argument to browse
    a subdirectory instead of the whole repo.

    Returns a 404 if repo_id was never indexed via POST /index, or if
    `path` doesn't exist under the repo's root.
    """
    try:
        return get_file_tree(repo_id, subpath=path)
    except UnknownRepoError as exc:
        # See ask_question's 404 handler: UnknownRepoError subclasses
        # KeyError, whose __str__ re-wraps the message in repr() -- use
        # the original arg directly so the 404 body stays human-readable.
        raise HTTPException(status_code=404, detail=exc.args[0]) from exc
    except SubpathNotFoundError as exc:
        raise HTTPException(status_code=404, detail=exc.args[0]) from exc
