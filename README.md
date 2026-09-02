# repoGuide
An AI-assisted repository exploration agent

## Stack
Python 3.11+, FastAPI, Chroma (vector store), pytest.

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`. Originally
  `jinaai/jina-embeddings-v2-base-code`, chosen for its code-specific
  training, but that model hit a ~2.3GB single CPU memory allocation
  during smoke testing on real hardware. all-MiniLM-L6-v2 trades some
  code-specific embedding quality for a footprint small enough to
  actually run on that machine.
