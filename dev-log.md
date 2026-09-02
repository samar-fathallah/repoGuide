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

## 2026-09-02 — Structural tools: find_definition, find_usages

**Asked for:** Implement the two remaining agent tools, `find_definition`
and `find_usages`, querying the structural SQLite index from Day 2, plus
tests against a known structural index (same approach as the chunker and
structural-indexer tests).

**Claude did:** Both tools resolve a symbol name against the structural
index, matching either an exact top-level name or a qualified suffix —
querying `"prepare"` correctly matches both `Service.prepare` and
`Other.prepare` without needing to know which class it belongs to.
`find_usages` surfaces both call sites and import statements as "usages,"
distinguished by a `kind` field (`"call"` vs `"import"`). Both return `[]`
rather than raising when a `repo_id` has no structural index at all,
matching the "tool errors are recoverable, not crashes" pattern used
throughout the agent loop.

**Accelerated:** Reusing Day 2's schema and query patterns meant this was
mostly query-writing, not new plumbing.

**Needed correction:** —

**Verified by hand:** Read the suffix-matching tests closely, in
particular one built specifically to catch a SQL `LIKE`-wildcard bug —
SQLite treats `_` as a single-character wildcard, so a naive suffix query
for `"get_all"` could accidentally match a decoy symbol named
`"getXall"` if the underscore isn't escaped. Confirmed the test actually
exercises that failure mode via a purpose-built decoy row, not just an
assertion that happens to pass.

---

## 2026-09-02 — Agent loop: tool_registry.py, loop.py

**Asked for:** Build the agent loop — schemas for all five real tools
plus a `submit_answer` pseudo-tool for structured final output, the
actual send → check-for-tool-call → run-tool → feed-result-back loop
using the Anthropic SDK, `repo_id` hidden from every schema and injected
server-side only, a full ordered `tool_calls` trace with arguments (this
is what Study B grades), a refusal instruction for when nothing relevant
turns up, an 8-iteration cap, and fully mocked tests — no real API calls.

**Claude did:** `tool_registry.py` with schemas for `search_code`,
`get_file_tree`, `find_definition`, `find_usages`, `read_file_section`,
plus `submit_answer` — none expose `repo_id`. `TOOL_FUNCTIONS` maps tool
name to the real callable; `submit_answer` is deliberately excluded and
handled specially in the loop instead of dispatched. `loop.py`'s
`run_agent(repo_id, question)` injects `repo_id` only at the actual
function call (so the recorded trace matches exactly what the model saw
and asked for — no `repo_id` leakage either way), wraps each tool call in
try/except so a bad argument degrades to a recoverable result instead of
crashing the run, and caps at 8 iterations with a fixed "couldn't
complete" fallback rather than looping forever. `test_agent_loop.py`: 4
tests against a fully mocked Anthropic client — single tool call →
`submit_answer`; multiple sequential tool calls → `submit_answer` with
exact order/arguments asserted; citations propagating through untouched;
the iteration cap producing the fallback without exceeding 8 model calls.

**Accelerated:** Working out the loop's shape in conversation before
writing any code — `tool_use` → execute → `tool_result` → repeat,
`submit_answer` as the structured stop signal — meant the implementation
matched the design on the first pass instead of needing structural rework
later.

**Needed correction:** Two real bugs, caught by review rather than by the
tests themselves. First, the `try/except` around tool dispatch had no
logging — a genuine bug in a tool's implementation would look identical
to an ordinary "bad argument from the model" error, with nothing telling
me it happened; fixed with a module-level `logger` and
`logger.exception(...)` inside the `except`. Second, `TOOL_FUNCTIONS[name]`
was being looked up *before* the `try` block instead of inside it, so an
unregistered tool name (e.g. a future tool added to the schema list but
not to `TOOL_FUNCTIONS`) would raise an unhandled `KeyError` and crash the
whole run instead of degrading gracefully like every other dispatch
failure.

**Verified by hand:** Traced through what "hide `repo_id` from the model"
actually means in the code — confirmed the schemas never mention it and
it's injected only at the real function call, so the recorded
`tool_calls` trace is `repo_id`-free and reflects exactly what the model
saw, not a reconstruction.

---

## 2026-09-02 — Wiring POST /ask

**Asked for:** Wire `run_agent` into a real `POST /ask` per the API
contract (`{question, repo_id}` in, `{answer, citations, tool_calls}`
out), validating `repo_id` against the existing repo registry, 422 on an
empty question, and endpoint tests with `run_agent` and the registry
lookup mocked (the Anthropic client itself is already covered by
`test_agent_loop.py`, so no need to mock it twice).

**Claude did:** `AskRequest`/`Citation`/`ToolCall`/`AskResponse` models;
the route validates `repo_id` via `repo_registry.get_repo_path` — the
same function `get_file_tree`/`read_file_section` already use, not a
second lookup path — *before* calling `run_agent` at all, so an unknown
`repo_id` fails fast with a 404 and never spends a model call. 422 on an
empty/whitespace question via a Pydantic `field_validator`, without
touching either the registry or `run_agent`. 5 tests in
`test_api_ask.py`: happy path, 404 path (asserting `run_agent` was never
called), both whitespace-only and empty-string 422 cases, and
`tool_calls` order/argument preservation through the route.

**Accelerated:** —

**Needed correction:** Found and fixed a real bug before it shipped:
`UnknownRepoError` subclasses `KeyError`, and `KeyError.__str__` wraps
its message in `repr()` — using `str(exc)` for the 404 detail produced a
garbled, literally-quoted message instead of clean text. Fixed by reading
`exc.args[0]` directly, which bypasses `__str__` entirely.

**Verified by hand:** Confirmed against the real, unmocked registry that
the 404 body is actually clean text now — worth noting that a
status-code-only assertion in the mocked test wouldn't necessarily have
caught the `repr()`-wrapping bug in the first place; the fix included
asserting on the exact detail text going forward, not just the status
code.

---

## 2026-09-02 — GET /repos, GET /repos/{id}/tree

**Asked for:** The two remaining §3.4 endpoints, reusing
`repo_registry.load_registry()` and the existing `get_file_tree` tool
rather than building second implementations of either. `repo_path` is
deliberately exposed in `GET /repos` — a conscious decision, not an
oversight, given this is a local, single-user dev tool with no auth.

**Claude did:** `GET /repos` returns the registry's contents as a list
(200 with an empty list when nothing's indexed yet, not an error); `GET
/repos/{id}/tree` calls the same `get_file_tree` the agent already uses
internally, with an unknown `repo_id` caught and returned as a 404 using
the `exc.args[0]` fix from `/ask` (not `str(exc)` — the same
`KeyError`/`repr()` trap would otherwise have resurfaced here).
`test_api_repos.py` covers both endpoints, including a dedicated
regression test asserting the 404 body text is clean, not just the
status code.

**Accelerated:** Because both endpoints reuse existing, already-tested
logic (the registry, the `get_file_tree` tool) rather than introducing
new logic of their own, this was mostly routing and response-model work.

**Needed correction:** —

**Verified by hand:** —

---

## 2026-09-02 — End-to-end smoke test (TinyDB)

**Asked for:** Before trusting any of today's mocked tests, actually run
the live system against a real small repo (TinyDB, ~1,800 LOC, pure
Python) with a real Anthropic key — mocked tests prove the loop's
mechanics are correct given a canned response, not that the real system
behaves the way the system prompt asks it to.

**Claude did:** N/A — this was manual, hands-on verification, and it's
the most valuable hour of today precisely because of what it caught.

**Needed correction — five distinct issues found, none of them visible
from the mocked test suite alone:**

1. `POST /index` hung on first run. Root cause: the configured embedding
   model (`jinaai/jina-embeddings-v2-base-code`) was downloading for the
   first time, and separately, once running, blew CPU memory (~2.3GB
   single allocation) inside its attention layer on longer chunks — its
   8192-token context window is much larger than this machine can afford
   at any reasonable batch size. Resolved by switching to
   `sentence-transformers/all-MiniLM-L6-v2` (22M params, no long-context
   architecture) — a real trade-off (less code-specific embedding
   quality) made explicit rather than hidden, and documented in the
   README's embeddings line.
2. `POST /ask` failed with `Could not resolve authentication method` even
   with a real key in `.env` — nothing in the app's startup path actually
   called `load_dotenv()`, so the key never reached the process
   environment. `.env.example` documenting the variable isn't the same as
   the app loading it; the earlier prompt for the agent loop asked for
   the former and never explicitly asked for the latter, which is why
   this gap made it all the way to a live smoke test before surfacing.
3. `POST /ask` then failed with a 400 asking for an `anthropic-workspace-id`
   header — the generated API key was an identity-linked (personal) key
   rather than one scoped to a single workspace. Resolved by generating a
   workspace-scoped key instead, avoiding the extra header entirely.
4. `POST /ask` then failed with `'tuple' object has no attribute
   'messages'` — a stray trailing comma after the closing `)` of the
   `Anthropic(...)` client construction turned `client` into a 1-element
   tuple wrapping the real client, rather than the client itself. Classic
   Python footgun: a trailing comma outside a function call's parens
   creates a tuple even though the line still looks like an ordinary
   assignment.
5. Once `/ask` finally worked end to end, `search_code` results showed
   full absolute Windows filesystem paths
   (`C:\Users\samos\...\tinydb\utils.py`) in `file_path`, instead of
   paths relative to the indexed repo root. The structural index
   (`find_definition`/`find_usages`) already stores clean relative paths
   — this was an inconsistency between the two indices, not a new
   problem to design around. Real consequences beyond cosmetics: leaks
   local machine structure into API responses, and would break the
   moment the same repo is indexed on a different machine, in CI, or
   inside the Docker container §3.9 requires. Fix in progress: relativize
   `file_path` at chunk-creation time, reusing the structural index's
   existing (correct) approach instead of reimplementing it separately.

**Verified by hand:** Everything in this entry — none of it was caught
by an automated test, all of it by actually running the system and
reading logs/tracebacks closely. Also manually verified the refusal
path on a genuinely absent question ("is there a dog type class in this
repo?") — got a clean refusal grounded in real tool results (the model
correctly explained *why* its `search_code` hits weren't relevant, rather
than just returning a generic "not found"), `citations: []`, and evidence
in `tool_calls` that it actually checked (`search_code` then
`get_file_tree`) before concluding, not just guessed.

**Also noted, not urgent:** two efficiency findings queued for when
repos scale up in size — `get_file_tree` returns the entire recursive
tree in one call with no depth limit, and `search_code` returns full
chunk source regardless of size (an 80-line irrelevant chunk came back
in full on one query). Both fine at TinyDB's size; a depth limit and
snippet truncation (falling back to `read_file_section` for full content)
are queued as a follow-up before testing against a larger corpus.

---

## 2026-09-02 — Git hygiene: .env and cache files caught in tracking

**Asked for:** —

**Claude did:** —

**Needed correction:** `.env` and some cache/index output ended up
committed before a `.gitignore` existed to catch them. `.gitignore`
alone doesn't untrack a file git is already tracking — fixed with `git
rm --cached` for each, followed by a commit. Since `.env` had briefly
held a real API key, checked `git log --all --full-history -- .env` to
confirm whether it had ever actually been pushed before deciding whether
a key rotation was necessary.

**Verified by hand:** `git ls-files .env` to confirm it's untracked now;
`git status` after the fact to confirm neither reappears despite
`.gitignore` being in place.