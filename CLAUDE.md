# RepoGuide

An agent that answers questions about a Python repository using both
semantic retrieval (AST-aware chunking + vector search) and structural
analysis (an index of definitions, imports, and call sites built with
Python's `ast` module).

## Stack
Python 3.11+, FastAPI, Chroma (vector store), pytest.

## Hard constraints
- Python-only AST parsing — no tree-sitter.
- No graph database — structural index is JSON or SQLite.
- Tool selection must happen inside the LLM agent, not in routing code.