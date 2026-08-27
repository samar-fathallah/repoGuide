## 2026-08-27 — Day 1: AST chunker + CI

**Asked for:** AST-aware chunker (one chunk per function/class/method,
with metadata) plus fixture files and tests for nested classes,
decorated functions, and oversized functions. Then a CI workflow.

**What Claude Code did:** [fill in — did it get the metadata fields
right on the first try? Did it choose a sensible overlap size for the
oversized-function split? Did the CI file need Python version fixed,
or did it guess correctly?]

**Needed correction:** [anything you had to push back on, fix, or
reject — even small things count, like a wrong import or a fixture
that didn't actually trigger the edge case you wanted]

**Verified by hand:** ran `pytest` locally, read through
ast_chunker.py, confirmed the GitHub Actions run went green.