import repoguide.tools.search_code as search_code_module
from repoguide.tools.search_code import search_code


class FakeCollection:
    """A stand-in for a Chroma collection that returns a fixed query response."""

    def __init__(self, response):
        self._response = response
        self.last_query_texts = None
        self.last_n_results = None

    def query(self, query_texts, n_results):
        self.last_query_texts = query_texts
        self.last_n_results = n_results
        return self._response


def _patch_collection(monkeypatch, collection):
    monkeypatch.setattr(
        search_code_module.vector_store,
        "get_or_create_collection",
        lambda repo_id: collection,
    )


def test_search_code_reshapes_query_results_into_expected_dicts(monkeypatch):
    response = {
        "ids": [["file_a.py:1-5", "file_b.py:10-20"]],
        "documents": [["def foo():\n    pass\n", "class Bar:\n    pass\n"]],
        "metadatas": [
            [
                {
                    "file_path": "file_a.py",
                    "start_line": 1,
                    "end_line": 5,
                    "symbol_name": "foo",
                    "symbol_type": "function",
                    "enclosing_class": "",
                    "is_split": False,
                    "part_index": 0,
                    "part_count": 0,
                },
                {
                    "file_path": "file_b.py",
                    "start_line": 10,
                    "end_line": 20,
                    "symbol_name": "Bar",
                    "symbol_type": "class",
                    "enclosing_class": "",
                    "is_split": True,
                    "part_index": 1,
                    "part_count": 2,
                },
            ]
        ],
        "distances": [[0.12, 0.45]],
    }
    fake_collection = FakeCollection(response)
    _patch_collection(monkeypatch, fake_collection)

    results = search_code("sample-repo", "how does foo work", k=2)

    assert fake_collection.last_query_texts == ["how does foo work"]
    assert fake_collection.last_n_results == 2

    assert results == [
        {
            "file_path": "file_a.py",
            "start_line": 1,
            "end_line": 5,
            "symbol_name": "foo",
            "symbol_type": "function",
            "is_split": False,
            "code": "def foo():\n    pass\n",
            "distance": 0.12,
        },
        {
            "file_path": "file_b.py",
            "start_line": 10,
            "end_line": 20,
            "symbol_name": "Bar",
            "symbol_type": "class",
            "is_split": True,
            "code": "class Bar:\n    pass\n",
            "distance": 0.45,
        },
    ]


def test_search_code_returns_fewer_than_k_when_collection_has_fewer_chunks(monkeypatch):
    response = {
        "ids": [["only_one.py:1-2"]],
        "documents": [["x = 1\n"]],
        "metadatas": [
            [
                {
                    "file_path": "only_one.py",
                    "start_line": 1,
                    "end_line": 2,
                    "symbol_name": "",
                    "symbol_type": "module",
                    "enclosing_class": "",
                    "is_split": False,
                    "part_index": 0,
                    "part_count": 0,
                }
            ]
        ],
        "distances": [[0.9]],
    }
    fake_collection = FakeCollection(response)
    _patch_collection(monkeypatch, fake_collection)

    results = search_code("sample-repo", "anything", k=5)

    assert fake_collection.last_n_results == 5
    assert len(results) == 1
    assert results[0] == {
        "file_path": "only_one.py",
        "start_line": 1,
        "end_line": 2,
        "symbol_name": "",
        "symbol_type": "module",
        "is_split": False,
        "code": "x = 1\n",
        "distance": 0.9,
    }