"""Agent-facing tool: semantic search over a repo's indexed chunks."""

from __future__ import annotations

from typing import List

from repoguide.indexing import vector_store


def search_code(repo_id: str, query: str, k: int = 5) -> List[dict]:
    """Semantic search over repo_id's indexed chunks."""
    collection = vector_store.get_or_create_collection(repo_id)
    results = collection.query(query_texts=[query], n_results=k)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return [
        {
            "file_path": metadata["file_path"],
            "start_line": metadata["start_line"],
            "end_line": metadata["end_line"],
            "symbol_name": metadata["symbol_name"],
            "symbol_type": metadata["symbol_type"],
            "is_split": metadata["is_split"],
            "code": document,
            "distance": distance,
        }
        for metadata, document, distance in zip(metadatas, documents, distances)
    ]