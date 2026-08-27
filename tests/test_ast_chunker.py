from pathlib import Path

from repoguide.chunking.ast_chunker import chunk_file

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_nested_class_produces_chunks_for_both_class_levels_and_their_methods():
    chunks = chunk_file(FIXTURES_DIR / "nested_class.py")

    by_symbol = {c.symbol_name: c for c in chunks if c.symbol_type in ("class", "method")}

    assert set(by_symbol) == {
        "Outer",
        "Outer.outer_method",
        "Outer.Inner",
        "Outer.Inner.inner_method",
        "Outer.Inner.another_inner_method",
        "Outer.after_nested",
    }

    outer = by_symbol["Outer"]
    assert outer.symbol_type == "class"
    assert outer.class_name is None
    assert 'class Outer:' in outer.text
    # The class chunk preserves the full body, including nested definitions.
    assert "class Inner" in outer.text
    assert "def after_nested" in outer.text

    outer_method = by_symbol["Outer.outer_method"]
    assert outer_method.symbol_type == "method"
    assert outer_method.class_name == "Outer"

    inner = by_symbol["Outer.Inner"]
    assert inner.symbol_type == "class"
    assert inner.class_name == "Outer"
    assert "def inner_method" in inner.text
    assert "def another_inner_method" in inner.text

    inner_method = by_symbol["Outer.Inner.inner_method"]
    assert inner_method.symbol_type == "method"
    assert inner_method.class_name == "Outer.Inner"
    assert 'return "inner"' in inner_method.text

    another_inner_method = by_symbol["Outer.Inner.another_inner_method"]
    assert another_inner_method.class_name == "Outer.Inner"

    after_nested = by_symbol["Outer.after_nested"]
    assert after_nested.symbol_type == "method"
    assert after_nested.class_name == "Outer"

    for chunk in chunks:
        assert chunk.file_path == str(FIXTURES_DIR / "nested_class.py")
        assert chunk.start_line <= chunk.end_line
        assert not chunk.is_split


def test_decorated_function_and_method_include_decorator_lines():
    chunks = chunk_file(FIXTURES_DIR / "decorated_function.py")
    by_symbol = {c.symbol_name: c for c in chunks}

    top_level = by_symbol["decorated_top_level"]
    assert top_level.symbol_type == "function"
    assert top_level.class_name is None
    assert top_level.text.startswith("@simple_decorator")
    assert "@functools.lru_cache(maxsize=None)" in top_level.text
    assert "def decorated_top_level(n):" in top_level.text
    assert "return n * n" in top_level.text

    method = by_symbol["WithDecoratedMethod.decorated_method"]
    assert method.symbol_type == "method"
    assert method.class_name == "WithDecoratedMethod"
    assert method.text.lstrip().startswith("@staticmethod")
    assert "@simple_decorator" in method.text
    assert "def decorated_method(x):" in method.text

    plain_function = by_symbol["simple_decorator"]
    assert plain_function.symbol_type == "function"
    # The nested `wrapper` function is part of simple_decorator's body and
    # is not chunked separately, since only top-level defs and class
    # methods are chunked on their own.
    assert "wrapper" not in by_symbol
    assert "def wrapper(*args, **kwargs):" in plain_function.text

    module_chunk = next(c for c in chunks if c.symbol_type == "module")
    assert "import functools" in module_chunk.text


def test_oversized_function_is_split_into_overlapping_flagged_subchunks():
    chunks = chunk_file(FIXTURES_DIR / "long_function.py")
    parts = [c for c in chunks if c.symbol_name == "process_large_batch"]

    assert len(parts) > 1
    assert all(c.symbol_type == "function" for c in parts)
    assert all(c.is_split for c in parts)
    assert all(c.part_count == len(parts) for c in parts)
    assert [c.part_index for c in parts] == list(range(1, len(parts) + 1))

    # Parts are contiguous/overlapping and cover the whole function body.
    assert parts[0].start_line == 4
    assert parts[-1].end_line == 807
    for earlier, later in zip(parts, parts[1:]):
        assert later.start_line <= earlier.end_line
        assert later.start_line > earlier.start_line

    # Each individual part fits under the token budget the split was meant
    # to enforce.
    from repoguide.chunking.ast_chunker import MAX_CHUNK_TOKENS, estimate_tokens

    assert all(estimate_tokens(c.text) <= MAX_CHUNK_TOKENS * 1.1 for c in parts)

    module_chunk = next(c for c in chunks if c.symbol_type == "module")
    assert not module_chunk.is_split
