## 2026-08-27 — Day 1: AST chunker + CI

**Asked for:** AST-aware chunker (one chunk per function/class/method,
with metadata) plus fixture files and tests for nested classes,
decorated functions, and oversized functions. Then a CI workflow.


**Verified by hand:** ran `pytest` locally, read through
ast_chunker.py, confirmed the GitHub Actions run went green.


## 2026-08-28 — Day 2: Structural index + POST /index

**Asked for:** a structural indexer built with `ast` (definitions,
imports, call sites) persisted to SQLite; a fix so each repo gets its
own database file and re-indexing doesn't duplicate rows; a fixture
file plus tests covering nested classes, a decorated method, all four
import styles, and caller/callee attribution; a FastAPI app with
`/health` and `POST /index` tying together Day 1's chunker and today's
structural indexer; and automated API tests using FastAPI's TestClient.

**Claude did:** Designed a three-table schema (definitions / imports /
calls) with sensible indexes on the columns `find_definition` and
`find_usages` will actually query later. Correctly stored `calls`'
caller/callee as plain text rather than foreign keys, matching the
project's documented limitation that call resolution is approximate.
Got a subtle Python semantics detail right unprompted: decorator
expressions, base classes, and argument defaults are evaluated in the
*enclosing* scope, not the function/class being defined — so a call
inside a decorator is correctly attributed to the surrounding scope.

**Needed correction:** The first version of `build_index` wrote to a
single hardcoded database path, which would have silently overwritten
one repo's index with another's the moment a second repo was indexed —
a real gap, since the project indexes three separate repos. Asked for
`db_path` to become a parameter, plus deleting a file's existing rows
before re-inserting so re-indexing doesn't produce duplicate rows.

**Verified by hand:** Read the schema line by line rather than trusting
it looked reasonable. Manually recomputed every line number in the
fixture file against the test file's assertions to confirm they
actually matched a real file, not just each other. Ran the app and
exercised `POST /index` through the Swagger UI at `/docs` against a
real small repo before trusting anything downstream. Opened the
resulting SQLite file directly and inspected rows rather than assuming
the insert logic worked.

**Still open:** confirm `pytest tests/test_api.py` passes and that
GitHub Actions is green with today's additions before calling Day 2
closed.