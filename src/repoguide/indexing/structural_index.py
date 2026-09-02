"""Structural index of a Python codebase: definitions, imports, and calls.

Built entirely with the standard library `ast` module (per the project's
Python-only-parsing constraint) and persisted to SQLite (per the
no-graph-database constraint).

IMPORTANT LIMITATION: call-site resolution is by symbol name only, using
static lexical scoping — it does not perform real type inference. This
means:
  - dynamic dispatch (e.g. calling through a variable that could hold any
    of several callables, or via getattr/__getattr__) is not resolved;
  - aliasing (e.g. `f = some.module.func; f()`) records the callee as `f`,
    not `some.module.func`;
  - two unrelated classes that both define a method with the same name
    (e.g. `Dog.speak` and `Cat.speak`) are indistinguishable when a call
    site only says `.speak()` on some object — both would match a lookup
    for callee_symbol "speak".
Treat the `calls` table as a heuristic for navigation, not a precise
call graph.
"""

from __future__ import annotations

import ast
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from repoguide.paths import relative_to_repo_root

FunctionNode = Union[ast.FunctionDef, ast.AsyncFunctionDef]

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS definitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    symbol_name TEXT NOT NULL,
    symbol_type TEXT NOT NULL CHECK (symbol_type IN ('function', 'class', 'method')),
    enclosing_class TEXT,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_definitions_symbol_name ON definitions (symbol_name);
CREATE INDEX IF NOT EXISTS idx_definitions_file_path ON definitions (file_path);

CREATE TABLE IF NOT EXISTS imports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    line INTEGER NOT NULL,
    module TEXT NOT NULL,
    imported_name TEXT,
    alias TEXT
);
CREATE INDEX IF NOT EXISTS idx_imports_file_path ON imports (file_path);
CREATE INDEX IF NOT EXISTS idx_imports_module ON imports (module);

CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    line INTEGER NOT NULL,
    caller_symbol TEXT,
    callee_symbol TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_calls_callee_symbol ON calls (callee_symbol);
CREATE INDEX IF NOT EXISTS idx_calls_file_path ON calls (file_path);
"""


@dataclass
class Definition:
    file_path: str
    symbol_name: str
    symbol_type: str  # "function" | "class" | "method"
    enclosing_class: Optional[str]
    start_line: int
    end_line: int


@dataclass
class ImportRecord:
    file_path: str
    line: int
    module: str
    imported_name: Optional[str]  # None for `import module`; the name for `from module import name`
    alias: Optional[str]


@dataclass
class CallSite:
    file_path: str
    line: int
    caller_symbol: Optional[str]  # None means the call happens at module scope
    callee_symbol: str  # best-effort dotted name of the thing being called; see module docstring


@dataclass
class StructuralData:
    definitions: List[Definition] = field(default_factory=list)
    imports: List[ImportRecord] = field(default_factory=list)
    calls: List[CallSite] = field(default_factory=list)


def extract_structural_data(source: str, file_path: str) -> StructuralData:
    tree = ast.parse(source)
    visitor = _StructuralVisitor(file_path)
    visitor.visit(tree)
    return StructuralData(
        definitions=visitor.definitions,
        imports=visitor.imports,
        calls=visitor.calls,
    )


def index_file(
    file_path: Union[str, Path], repo_root: Optional[Union[str, Path]] = None
) -> StructuralData:
    """Extract structural data from the file at `file_path`.

    If `repo_root` is given, entries are labeled with `file_path` made
    relative to it (e.g. "pkg/module.py") instead of the raw path used to
    read the file -- callers indexing a whole repository should always
    pass this, so stored metadata doesn't leak the local filesystem
    layout. Omit it only for standalone use with no repo root concept.
    """
    path = Path(file_path)
    source = path.read_text(encoding="utf-8")
    label = relative_to_repo_root(path, repo_root) if repo_root is not None else str(file_path)
    return extract_structural_data(source, label)


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)


def store_structural_data(conn: sqlite3.Connection, data: StructuralData) -> None:
    conn.executemany(
        """
        INSERT INTO definitions
            (file_path, symbol_name, symbol_type, enclosing_class, start_line, end_line)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (d.file_path, d.symbol_name, d.symbol_type, d.enclosing_class, d.start_line, d.end_line)
            for d in data.definitions
        ],
    )
    conn.executemany(
        """
        INSERT INTO imports (file_path, line, module, imported_name, alias)
        VALUES (?, ?, ?, ?, ?)
        """,
        [(i.file_path, i.line, i.module, i.imported_name, i.alias) for i in data.imports],
    )
    conn.executemany(
        """
        INSERT INTO calls (file_path, line, caller_symbol, callee_symbol)
        VALUES (?, ?, ?, ?)
        """,
        [(c.file_path, c.line, c.caller_symbol, c.callee_symbol) for c in data.calls],
    )


def delete_rows_for_file(conn: sqlite3.Connection, file_path: str) -> None:
    """Remove any previously indexed rows for `file_path` from all three tables."""
    conn.execute("DELETE FROM definitions WHERE file_path = ?", (file_path,))
    conn.execute("DELETE FROM imports WHERE file_path = ?", (file_path,))
    conn.execute("DELETE FROM calls WHERE file_path = ?", (file_path,))


def build_index(
    file_paths: Sequence[Union[str, Path]],
    db_path: Union[str, Path],
    repo_root: Optional[Union[str, Path]] = None,
) -> None:
    """Index a set of source files into a SQLite database at `db_path`.

    Pass a distinct `db_path` per repository (e.g.
    `data/indices/<repo_id>/structural.db`) so repositories don't share
    and overwrite each other's index. Re-indexing a file that was already
    indexed into this database replaces its rows rather than duplicating
    them.

    Pass `repo_root` (the repository's root directory) when indexing a
    whole repository, so stored rows are labeled with paths relative to
    it instead of the raw filesystem paths in `file_paths` -- see
    index_file.
    """
    conn = sqlite3.connect(db_path)
    try:
        create_schema(conn)
        with conn:
            for file_path in file_paths:
                label = (
                    relative_to_repo_root(file_path, repo_root)
                    if repo_root is not None
                    else str(file_path)
                )
                delete_rows_for_file(conn, label)
                store_structural_data(conn, index_file(file_path, repo_root=repo_root))
    finally:
        conn.close()


class _StructuralVisitor(ast.NodeVisitor):
    def __init__(self, file_path: str):
        self.file_path = file_path
        # Stack of (kind, qualified_name) for enclosing function/class scopes.
        self._scope_stack: List[Tuple[str, str]] = []
        self.definitions: List[Definition] = []
        self.imports: List[ImportRecord] = []
        self.calls: List[CallSite] = []

    def _qualified_name(self, name: str) -> str:
        prefix = self._scope_stack[-1][1] if self._scope_stack else None
        return f"{prefix}.{name}" if prefix else name

    def _nearest_enclosing_class(self) -> Optional[str]:
        for kind, qualified_name in reversed(self._scope_stack):
            if kind == "class":
                return qualified_name
        return None

    def _current_caller_symbol(self) -> Optional[str]:
        return self._scope_stack[-1][1] if self._scope_stack else None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qualified_name = self._qualified_name(node.name)
        self.definitions.append(
            Definition(
                file_path=self.file_path,
                symbol_name=qualified_name,
                symbol_type="class",
                enclosing_class=self._nearest_enclosing_class(),
                start_line=_start_line(node),
                end_line=node.end_lineno,
            )
        )
        # Decorators, base classes, and keyword arguments (e.g. `metaclass=`)
        # are evaluated in the *enclosing* scope when the `class` statement
        # runs, not inside the class body's own scope.
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)

        self._scope_stack.append(("class", qualified_name))
        for stmt in node.body:
            self.visit(stmt)
        self._scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: FunctionNode) -> None:
        qualified_name = self._qualified_name(node.name)
        parent_kind = self._scope_stack[-1][0] if self._scope_stack else None
        symbol_type = "method" if parent_kind == "class" else "function"
        self.definitions.append(
            Definition(
                file_path=self.file_path,
                symbol_name=qualified_name,
                symbol_type=symbol_type,
                enclosing_class=self._nearest_enclosing_class(),
                start_line=_start_line(node),
                end_line=node.end_lineno,
            )
        )
        # Decorators, parameter defaults/annotations, and the return
        # annotation are evaluated in the *enclosing* scope when the `def`
        # statement runs, not inside the function's own scope.
        for decorator in node.decorator_list:
            self.visit(decorator)
        self._visit_eager_argument_expressions(node.args)
        if node.returns is not None:
            self.visit(node.returns)

        self._scope_stack.append(("function", qualified_name))
        for stmt in node.body:
            self.visit(stmt)
        self._scope_stack.pop()

    def _visit_eager_argument_expressions(self, args: ast.arguments) -> None:
        for default in (*args.defaults, *args.kw_defaults):
            if default is not None:
                self.visit(default)
        all_args = (*args.posonlyargs, *args.args, *args.kwonlyargs)
        if args.vararg is not None:
            all_args = (*all_args, args.vararg)
        if args.kwarg is not None:
            all_args = (*all_args, args.kwarg)
        for arg in all_args:
            if arg.annotation is not None:
                self.visit(arg.annotation)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(
                ImportRecord(
                    file_path=self.file_path,
                    line=node.lineno,
                    module=alias.name,
                    imported_name=None,
                    alias=alias.asname,
                )
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        # Relative imports (`from . import x`, `from ..pkg import y`) can't be
        # resolved to an absolute module path from the AST alone, since that
        # requires knowing the importing file's package location. We record
        # the level as leading dots so the relationship is still visible.
        module = ("." * node.level) + (node.module or "")
        for alias in node.names:
            self.imports.append(
                ImportRecord(
                    file_path=self.file_path,
                    line=node.lineno,
                    module=module,
                    imported_name=alias.name,
                    alias=alias.asname,
                )
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # See the "IMPORTANT LIMITATION" note in the module docstring: this
        # is a name-based approximation, not a resolved call graph.
        self.calls.append(
            CallSite(
                file_path=self.file_path,
                line=node.lineno,
                caller_symbol=self._current_caller_symbol(),
                callee_symbol=_callee_name(node.func),
            )
        )
        self.generic_visit(node)


def _start_line(node: ast.AST) -> int:
    decorator_list = getattr(node, "decorator_list", None)
    if decorator_list:
        return min(d.lineno for d in decorator_list)
    return node.lineno


def _callee_name(node: ast.expr) -> str:
    """Best-effort dotted name for a call's target expression.

    Handles the common `name(...)`, `obj.method(...)`, and `a.b.c(...)`
    shapes exactly; anything more dynamic (e.g. `get_handler()(...)`,
    `handlers[key](...)`) falls back to an unparsed source snippet.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_callee_name(node.value)}.{node.attr}"
    try:
        return ast.unparse(node)
    except Exception:
        return "<unknown>"
