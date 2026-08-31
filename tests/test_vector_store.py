from typing import Any, Dict

import pytest
from chromadb import Documents, EmbeddingFunction, Embeddings

from repoguide.chunking.ast_chunker import Chunk
from repoguide.indexing.vector_store import add_chunks, chunk_id, get_or_create_collection


class FakeEmbeddingFunction(EmbeddingFunction):
    """A trivial embedding function: same fixed short vector for any input,
    so tests never load the real model or run real inference."""

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim

    def __call__(self, input: Documents) -> Embeddings:
        return [[0.1] * self._dim for _ in input]

    @staticmethod
    def name() -> str:
        return "fake-embedding-function"

    def get_config(self) -> Dict[str, Any]:
        return {"dim": self._dim}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "FakeEmbeddingFunction":
        return FakeEmbeddingFunction(dim=config["dim"])


def _collection(tmp_path, repo_id: str):
    return get_or_create_collection(
        repo_id, indices_dir=tmp_path / "indices", embedding_function=FakeEmbeddingFunction()
    )


def _make_chunk(**overrides) -> Chunk:
    defaults = dict(
        file_path="pkg/module.py",
        start_line=1,
        end_line=2,
        symbol_name="foo",
        symbol_type="function",
        class_name=None,
        text="def foo():\n    pass\n",
    )
    defaults.update(overrides)
    return Chunk(**defaults)


def test_collections_are_isolated_per_repo_id(tmp_path):
    collection_a = _collection(tmp_path, "repo-a")
    collection_b = _collection(tmp_path, "repo-b")

    add_chunks(collection_a, "a.py", [_make_chunk(file_path="a.py")])

    assert collection_a.count() == 1
    assert collection_b.count() == 0
    assert collection_b.get()["ids"] == []


def test_add_chunks_stores_correct_ids_documents_and_metadata_for_mixed_chunks(tmp_path):
    collection = _collection(tmp_path, "sample-repo")

    plain_chunk = _make_chunk(
        file_path="pkg/module.py",
        start_line=1,
        end_line=2,
        symbol_name="foo",
        symbol_type="function",
        class_name=None,
        text="def foo():\n    pass\n",
    )
    split_chunk = _make_chunk(
        file_path="pkg/module.py",
        start_line=10,
        end_line=40,
        symbol_name="big_function",
        symbol_type="function",
        class_name="SomeClass",
        text="def big_function():\n    ...\n",
        is_split=True,
        part_index=2,
        part_count=5,
    )

    add_chunks(collection, "pkg/module.py", [plain_chunk, split_chunk])

    plain_id = chunk_id(plain_chunk)
    split_id = chunk_id(split_chunk)
    assert collection.count() == 2

    stored = collection.get(ids=[plain_id, split_id], include=["documents", "metadatas"])
    by_id = dict(zip(stored["ids"], zip(stored["documents"], stored["metadatas"])))

    plain_document, plain_metadata = by_id[plain_id]
    assert plain_document == plain_chunk.text
    assert plain_metadata == {
        "file_path": "pkg/module.py",
        "start_line": 1,
        "end_line": 2,
        "symbol_name": "foo",
        "symbol_type": "function",
        "enclosing_class": "",
        "is_split": False,
        "part_index": 0,
        "part_count": 0,
    }

    split_document, split_metadata = by_id[split_id]
    assert split_document == split_chunk.text
    assert split_metadata == {
        "file_path": "pkg/module.py",
        "start_line": 10,
        "end_line": 40,
        "symbol_name": "big_function",
        "symbol_type": "function",
        "enclosing_class": "SomeClass",
        "is_split": True,
        "part_index": 2,
        "part_count": 5,
    }


def test_add_chunks_replaces_old_entries_for_the_same_file_instead_of_duplicating(tmp_path):
    collection = _collection(tmp_path, "sample-repo")

    old_chunk = _make_chunk(file_path="a.py", start_line=1, end_line=2)
    add_chunks(collection, "a.py", [old_chunk])
    old_id = chunk_id(old_chunk)
    assert collection.count() == 1
    assert collection.get(ids=[old_id])["ids"] == [old_id]

    new_chunk = _make_chunk(file_path="a.py", start_line=5, end_line=6)
    add_chunks(collection, "a.py", [new_chunk])
    new_id = chunk_id(new_chunk)

    assert collection.count() == 1
    assert collection.get()["ids"] == [new_id]
    assert collection.get(ids=[old_id])["ids"] == []