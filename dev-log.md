## 2026-08-27 — Day 1: AST chunker + CI

**Asked for:** AST-aware chunker (one chunk per function/class/method,
with metadata) plus fixture files and tests for nested classes,
decorated functions, and oversized functions. Then a CI workflow.


**Verified by hand:** ran `pytest` locally, read through
ast_chunker.py, confirmed the GitHub Actions run went green.
