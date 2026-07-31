"""Unit tests for the working-set (fetch/read) behavior of `SearchCorpusTool` /
`GrepCorpusTool`: the `_build_metadata_where` inclusion/exclusion composition, the
per-mode `RetrievalState` updates, and the `SearchAgent` rendering of read payloads
(ChunkBlocks) vs fetch digests (TextBlock summaries).

No LLM / Chroma. Runs under pytest if installed, or standalone:
`python3 tests/test_working_set_tools.py`.
"""

from __future__ import annotations

from types import SimpleNamespace

from skunk.config import SearchAgentConfig
from skunk.multi_turn_agent import ChunkBlock, TextBlock
from skunk.sandbox.local_python_executor import CodeOutput
from skunk.search_agent.retrieval_state import RetrievalState
from skunk.search_agent.search_agent import SearchAgent
from skunk.search_agent.search_tools import (
    EMPTY_RESULT_MESSAGE,
    GREP_RESULT_TAG,
    SEARCH_RESULT_TAG,
    GrepCorpusTool,
    SearchCorpusTool,
    _build_metadata_where,
)


# ---- fakes ------------------------------------------------------------------------


class FakeLLMClient:
    def __init__(self):
        self.config = SimpleNamespace(llm_model="agent-model", llm_context_limits={})
        self.embed_calls: list[str] = []

    def embed_query(self, text, *, ctx=None, usage_key="default"):
        self.embed_calls.append(text)
        return [0.0, 0.0]


class FakeChroma:
    """Backed by a flat chunk list; records every query()/get() kwargs. The `where` /
    `where_document` filters are NOT applied (tests only assert what was passed)."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.query_kwargs: list[dict] = []
        self.get_kwargs: list[dict] = []

    def _meta(self, c: dict) -> dict:
        return {"doc_id": c["doc_id"], "element_id": c.get("element_id", 0), "type": c.get("type", "text")}

    def query(self, **kwargs):
        self.query_kwargs.append(kwargs)
        rows = self.chunks[: kwargs["n_results"]]
        return {
            "ids": [[c["chunk_id"] for c in rows]],
            "documents": [[c["text"] for c in rows]],
            "metadatas": [[self._meta(c) for c in rows]],
            "distances": [[0.1 * i for i, _ in enumerate(rows)]],
        }

    def get(self, where=None, where_document=None, include=None, limit=None):
        self.get_kwargs.append(
            {"where": where, "where_document": where_document, "include": include, "limit": limit}
        )
        rows = self.chunks[:limit] if limit is not None else self.chunks
        return {
            "ids": [c["chunk_id"] for c in rows],
            "documents": [c["text"] for c in rows],
            "metadatas": [self._meta(c) for c in rows],
        }


_CHUNKS = [
    {"chunk_id": "d1_0", "doc_id": "d1", "element_id": 0, "text": "apple pie recipe"},
    {"chunk_id": "d2_0", "doc_id": "d2", "element_id": 0, "text": "banana bread"},
    {"chunk_id": "d1_1", "doc_id": "d1", "element_id": 1, "text": "more apples"},
]


def _search_tool(state: RetrievalState | None = None) -> tuple[SearchCorpusTool, FakeChroma]:
    collection = FakeChroma(list(_CHUNKS))
    return SearchCorpusTool(collection, FakeLLMClient(), state), collection


# ---- _build_metadata_where --------------------------------------------------------


def test_where_none_when_no_clauses():
    assert _build_metadata_where(
        metadata_filter=None, in_chunk_ids=None, in_doc_ids=None,
        not_in_chunk_ids=set(), not_in_doc_ids=None,
    ) is None


def test_where_single_clause_is_unwrapped():
    where = _build_metadata_where(
        metadata_filter=None, in_chunk_ids={"c1"}, in_doc_ids=None,
        not_in_chunk_ids=None, not_in_doc_ids=None,
    )
    assert where == {"chunk_id": {"$in": ["c1"]}}


def test_where_inclusion_sets_are_unioned():
    # A chunk is in the working set if ITS id was fetched OR its whole doc was — the two
    # inclusion sets must combine with $or, never $and (their intersection is ~empty).
    where = _build_metadata_where(
        metadata_filter={"year": "1946"},
        in_chunk_ids={"c1"}, in_doc_ids={"d9"},
        not_in_chunk_ids={"px"}, not_in_doc_ids={"pd"},
    )
    assert {"$or": [{"doc_id": {"$in": ["d9"]}}, {"chunk_id": {"$in": ["c1"]}}]} in where["$and"]
    assert {"year": "1946"} in where["$and"]
    assert {"doc_id": {"$nin": ["pd"]}} in where["$and"]
    assert {"chunk_id": {"$nin": ["px"]}} in where["$and"]


# ---- search_corpus modes ----------------------------------------------------------


def test_search_fetch_only_excludes_fetched_and_pruned_and_updates_state():
    state = RetrievalState(
        fetched_chunk_ids={"old"}, fetched_doc_ids={"fd"},
        pruned_chunk_ids={"px"}, pruned_doc_ids={"pd"},
    )
    tool, collection = _search_tool(state)
    out = tool("q", top_k=3, fetch=True)
    where = collection.query_kwargs[0]["where"]
    assert {"chunk_id": {"$nin": ["old", "px"]}} in where["$and"]
    assert {"doc_id": {"$nin": ["fd", "pd"]}} in where["$and"]
    # Payload: digest side only; nothing marked read.
    assert out[SEARCH_RESULT_TAG] is True and out["read_chunks"] == []
    assert [c["chunk_id"] for c in out["fetched_chunks"]] == ["d1_0", "d2_0", "d1_1"]
    assert state.fetched_chunk_ids == {"old", "d1_0", "d2_0", "d1_1"}
    assert not state.read_chunk_ids


def test_search_fetch_and_read_excludes_pruned_and_read():
    state = RetrievalState(
        pruned_chunk_ids={"px"}, pruned_doc_ids={"pd"},
        read_chunk_ids={"sx"}, read_doc_ids={"sd"},
    )
    tool, collection = _search_tool(state)
    out = tool("q", top_k=3, fetch=True, read=True)
    where = collection.query_kwargs[0]["where"]
    assert {"chunk_id": {"$nin": ["px", "sx"]}} in where["$and"]
    assert {"doc_id": {"$nin": ["pd", "sd"]}} in where["$and"]
    # Both payload sides carry the chunks; both state sets are updated.
    assert [c["chunk_id"] for c in out["read_chunks"]] == [c["chunk_id"] for c in out["fetched_chunks"]]
    assert {"d1_0", "d2_0", "d1_1"} <= state.fetched_chunk_ids
    assert {"d1_0", "d2_0", "d1_1"} <= state.read_chunk_ids


def test_search_read_only_queries_working_set_and_excludes_read():
    state = RetrievalState(fetched_chunk_ids={"d1_0", "d1_1"}, read_chunk_ids={"sx"})
    tool, collection = _search_tool(state)
    out = tool("q", top_k=2, read=True)
    where = collection.query_kwargs[0]["where"]
    assert {"chunk_id": {"$in": ["d1_0", "d1_1"]}} in where["$and"]
    assert {"chunk_id": {"$nin": ["sx"]}} in where["$and"]
    assert out["fetched_chunks"] == [] and out["read_chunks"]
    assert {c["chunk_id"] for c in out["read_chunks"]} <= state.read_chunk_ids


def test_search_read_only_rejects_empty_working_set():
    tool, _ = _search_tool(RetrievalState())
    try:
        tool("q", top_k=2, read=True)
    except AssertionError as e:
        assert "empty working set" in str(e)
    else:
        raise AssertionError("expected AssertionError on read from an empty working set")


# ---- grep_corpus modes ------------------------------------------------------------


def test_grep_fetch_only_groups_and_updates_state():
    state = RetrievalState()
    collection = FakeChroma(list(_CHUNKS))
    out = GrepCorpusTool(collection, state)("apple", fetch=True)
    assert out[GREP_RESULT_TAG] is True and out["read_groups"] == []
    # Groups in doc_id order; chunks in element order within each doc.
    assert [g["doc_id"] for g in out["fetched_groups"]] == ["d1", "d2"]
    assert [c["chunk_id"] for c in out["fetched_groups"][0]["fetched_chunks"]] == ["d1_0", "d1_1"]
    assert state.fetched_chunk_ids == {"d1_0", "d1_1", "d2_0"} and not state.read_chunk_ids


def test_grep_read_truncation_marks_only_rendered_read():
    # A tiny cap admits only the very first chunk into the read (rendered) side, while the
    # fetch side always carries every hit; the note tells the agent what was omitted.
    state = RetrievalState()
    collection = FakeChroma(list(_CHUNKS))
    out = GrepCorpusTool(collection, state)("apple", read=True, fetch=True, max_output_tokens=1)
    read_ids = [c["chunk_id"] for g in out["read_groups"] for c in g["read_chunks"]]
    fetched_ids = [c["chunk_id"] for g in out["fetched_groups"] for c in g["fetched_chunks"]]
    assert read_ids == ["d1_0"] and set(fetched_ids) == {"d1_0", "d1_1", "d2_0"}
    assert "truncation_note" in out and "showing 1 of 3" in out["truncation_note"]
    # Only the rendered chunk is read; everything stays fetchable/readable later.
    assert state.read_chunk_ids == {"d1_0"} and state.fetched_chunk_ids == {"d1_0", "d1_1", "d2_0"}


# ---- SearchAgent rendering --------------------------------------------------------


def _agent() -> SearchAgent:
    return SearchAgent(
        config=SearchAgentConfig(name="t"), document_map={},
        chroma_collection=FakeChroma([]), llm_client=FakeLLMClient(),
    )


def _blocks(payload: dict) -> list:
    return _agent()._blocks_from_output(CodeOutput(output=payload, logs=""))


def _chunk(i: int, doc_id: str = "d1") -> dict:
    return {
        "chunk_id": f"{doc_id}_{i}", "doc_id": doc_id, "est_num_tokens": i,
        "header": f"[h{i}]", "text": f"chunk text {i}",
    }


def test_render_search_read_chunks_as_chunkblocks():
    blocks = _blocks({SEARCH_RESULT_TAG: True, "read_chunks": [_chunk(1), _chunk(2)], "fetched_chunks": []})
    assert all(isinstance(b, ChunkBlock) for b in blocks)
    assert [b.chunk_id for b in blocks] == ["d1_1", "d1_2"]


def test_render_search_fetch_digest_caps_listed_headers():
    n = SearchAgent.chunks_per_summary + 2
    blocks = _blocks({SEARCH_RESULT_TAG: True, "read_chunks": [], "fetched_chunks": [_chunk(i) for i in range(n)]})
    [digest] = blocks
    assert isinstance(digest, TextBlock)
    total = sum(range(n))
    assert f"Retrieved {n} chunks with {total:,} est. tokens" in digest.text
    # Top-N headers by est. token count — the two smallest are elided.
    assert digest.text.count("[h") == SearchAgent.chunks_per_summary
    assert "[h0]" not in digest.text and "[h1]" not in digest.text and f"[h{n - 1}]" in digest.text


def test_render_search_empty():
    [empty] = _blocks({SEARCH_RESULT_TAG: True, "read_chunks": [], "fetched_chunks": []})
    assert empty.text == EMPTY_RESULT_MESSAGE


def test_render_grep_read_groups_and_truncated_marker():
    payload = {
        GREP_RESULT_TAG: True,
        "read_groups": [
            {"doc_id": "d1", "header": "# doc_id=d1", "read_chunks": [_chunk(1)], "fetched_chunks": [_chunk(1)]},
            {"doc_id": "d2", "header": "# doc_id=d2", "read_chunks": [], "fetched_chunks": [_chunk(2, "d2")]},
        ],
        "fetched_groups": [],
        "truncation_note": "[grep_corpus output truncated: showing 1 of 2 matching chunk(s)]",
    }
    blocks = _blocks(payload)
    # doc header, its chunk, second doc header, truncated marker, then the note.
    assert isinstance(blocks[0], TextBlock) and blocks[0].text == "# doc_id=d1"
    assert isinstance(blocks[1], ChunkBlock) and blocks[1].chunk_id == "d1_1"
    assert isinstance(blocks[2], TextBlock) and blocks[2].text == "# doc_id=d2"
    assert isinstance(blocks[3], TextBlock) and "truncated" in blocks[3].text
    assert "truncated" in blocks[-1].text


def test_render_grep_fetch_digest():
    payload = {
        GREP_RESULT_TAG: True,
        "read_groups": [],
        "fetched_groups": [
            {"doc_id": "d1", "header": "# doc_id=d1", "read_chunks": [], "fetched_chunks": [_chunk(1), _chunk(2)]},
            {"doc_id": "d2", "header": "# doc_id=d2", "read_chunks": [], "fetched_chunks": [_chunk(3, "d2")]},
        ],
    }
    [digest] = _blocks(payload)
    assert isinstance(digest, TextBlock)
    assert "Retrieved 3 chunks from 2 documents with 6 est. tokens" in digest.text


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ok")
