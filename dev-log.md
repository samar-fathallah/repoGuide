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


## 2026-08-29 — Day 3: Vector store, search_code / get_file_tree / read_file_section

**Asked for:** wire Chroma into the indexing pipeline using a local,
code-aware embedding model (deliberately no API key, so the "clone and
run from the README" requirement still holds); implement and test
search_code, get_file_tree, and read_file_section; close a test-
coverage gap once it was found.

**Claude did:** built repo-scoped Chroma collections mirroring the
SQLite repo-scoping pattern from Day 2; used upsert with a delete-first
step for re-indexing, which turned out to be more robust than plain
add — it correctly handles a chunk's line range shifting between runs,
not just simple re-run duplication. Added a JSON repo registry
(repo_id -> original path) unprompted-but-necessary, to give
get_file_tree and read_file_section something to resolve paths
against, and split the "path escapes the repo root" test into
relative-traversal and absolute-path cases on its own initiative.

**Needed correction — three real bugs, all found by actually running
the system against a real repo (SparseDrive), not by reading code:**

1. `flush_module_block()` in the chunker had no size check at all,
   unlike the function-chunking path — so an oversized module-level
   block (a 776-line ML config file with no functions/classes) became
   one giant, unsplit chunk. Found by inspecting real search results
   and noticing a suspiciously huge single match.
2. Even after fixing #1, `vector_store.py`'s `_metadata()` silently
   dropped `is_split` / `part_index` / `part_count` on the way into
   Chroma — so there was no way to confirm a retrieved chunk was
   actually a fragment of something larger.
3. The chunker's token-size estimate (`len(text) / 4`) badly
   undercounted real tokens for dense, punctuation-heavy code. A
   single long chunk caused a CPU out-of-memory crash during
   embedding, since self-attention memory scales with sequence length
   squared. Confirmed the exact mechanism with a live token count
   against the real model's tokenizer rather than guessing. Fixed with
   a hard safety cap on the model's max sequence length (truncate with
   a logged warning, never crash) plus a more conservative chunk-size
   threshold.

**Also closed:** `vector_store.py` had zero automated test coverage,
despite being where two of the three bugs above actually lived —
added dependency injection (an optional embedding_function parameter)
so tests can use a lightweight fake embedder instead of downloading
and running the real model, keeping the suite fast and offline.

**Verified by hand:** ran raw queries directly against the real
indexed repo after each fix, not just unit tests in isolation; used
`model.tokenizer` to measure real token counts and confirm the OOM
root cause before proposing a fix; re-ran the full test suite and
confirmed all new and existing tests pass; checked GitHub Actions is
still green with today's additions.