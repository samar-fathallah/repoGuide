"""Semantic index: a Chroma vector store of embedded AST chunks.

Each repository gets its own persistent Chroma client rooted at
`data/indices/<repo_id>/chroma/`, with a single collection inside it, so
collections are never shared and a search against one repo can never
return another repo's chunks.

Chunks are embedded with the `jinaai/jina-embeddings-v2-base-code` model
(via `sentence-transformers`), chosen for its training on source code
rather than prose. Each chunk gets a deterministic ID derived from its
file path and line range (e.g. "file.py:10-25"), so re-indexing a file
is idempotent: `add_chunks` deletes that file's existing rows before
inserting the freshly computed ones instead of appending duplicates.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.api.models.Collection import Collection

from repoguide.chunking.ast_chunker import Chunk

logger = logging.getLogger(__name__)

DEFAULT_INDICES_DIR = Path("data/indices")
COLLECTION_NAME = "chunks"
EMBEDDING_MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"

# Conservative cap on the model's max sequence length. The model itself
# supports much longer sequences, but running a full forward pass on an
# enormous, un-truncated chunk risks an out-of-memory crash; capping it
# here makes sentence-transformers truncate instead.
MAX_SEQ_LENGTH = 2048


class JinaCodeEmbeddingFunction(EmbeddingFunction):
    """Wraps the jina-embeddings-v2-base-code sentence-transformers model."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._model = SentenceTransformer(model_name, trust_remote_code=True)
        self._model.max_seq_length = MAX_SEQ_LENGTH

    def __call__(self, input: Documents) -> Embeddings:
        for text in input:
            token_count = len(self._model.tokenizer.encode(text))
            if token_count > MAX_SEQ_LENGTH:
                logger.warning(
                    "Chunk has %d tokens, exceeding max_seq_length=%d; it will be "
                    "truncated before embedding.",
                    token_count,
                    MAX_SEQ_LENGTH,
                )
        return self._model.encode(list(input), convert_to_numpy=True).tolist()

    @staticmethod
    def name() -> str:
        return "jina-embeddings-v2-base-code"

    def get_config(self) -> Dict[str, Any]:
        return {"model_name": self._model_name}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "JinaCodeEmbeddingFunction":
        return JinaCodeEmbeddingFunction(model_name=config["model_name"])


@lru_cache(maxsize=1)
def _get_embedding_function() -> JinaCodeEmbeddingFunction:
    """Load the embedding model once and reuse it across collections."""
    return JinaCodeEmbeddingFunction()


def get_or_create_collection(
    repo_id: str,
    indices_dir: Union[str, Path] = DEFAULT_INDICES_DIR,
    embedding_function: Optional[EmbeddingFunction] = None,
) -> Collection:
    """Get (or create) the Chroma collection for `repo_id`.

    Backed by a persistent client rooted at `<indices_dir>/<repo_id>/chroma/`,
    a directory distinct per repo_id, so different repos never share a
    collection or return each other's chunks in search results.

    `embedding_function` defaults to the real jina-embeddings-v2-base-code
    model; tests can inject a lightweight fake instead to avoid loading it.
    """
    if embedding_function is None:
        embedding_function = _get_embedding_function()

    persist_dir = Path(indices_dir) / repo_id / "chroma"
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=embedding_function
    )


def chunk_id(chunk: Chunk) -> str:
    """Deterministic ID for a chunk, derived from its file path and line range."""
    return f"{chunk.file_path}:{chunk.start_line}-{chunk.end_line}"


def _metadata(chunk: Chunk) -> dict:
    return {
        "file_path": chunk.file_path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "symbol_name": chunk.symbol_name or "",
        "symbol_type": chunk.symbol_type,
        "enclosing_class": chunk.class_name or "",
        "is_split": chunk.is_split,
        "part_index": chunk.part_index or 0,
        "part_count": chunk.part_count or 0,
    }


def add_chunks(collection: Collection, file_path: str, chunks: Sequence[Chunk]) -> None:
    """Replace `file_path`'s chunks in `collection` with the given ones.

    Any chunks previously stored for `file_path` are deleted first, so
    re-indexing a file (e.g. re-running /index) never leaves stale or
    duplicate entries behind, even if that file's chunk boundaries changed
    since the last run.
    """
    collection.delete(where={"file_path": file_path})
    if not chunks:
        return

    collection.upsert(
        ids=[chunk_id(chunk) for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        metadatas=[_metadata(chunk) for chunk in chunks],
    )


def add_chunks_for_files(
    collection: Collection, chunks_by_file: dict[str, List[Chunk]]
) -> None:
    """Convenience wrapper: apply `add_chunks` for each file's chunk list."""
    for file_path, chunks in chunks_by_file.items():
        add_chunks(collection, file_path, chunks)
