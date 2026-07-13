"""Unit tests for `SemanticFilterTool` (doc_ids + corpus modes), its `bind_retrieval_state`
wiring, and the `SEMFILTER_RESULT_TAG` rendering in `SearchAgent._blocks_from_output`.

No LLM / Chroma. Runs under pytest if installed, or standalone:
`python3 tests/test_semantic_filter_tool.py`.
"""

from __future__ import annotations

from skunk.config import SearchAgentConfig
from skunk.multi_turn_agent import ChunkBlock, TextBlock, Tool
from skunk.sandbox.local_python_executor import CodeOutput
from skunk.search_agent.search_agent import SearchAgent
from skunk.search_agent.search_tools import (
    EMPTY_RESULT_MESSAGE,
    SEMFILTER_RESULT_TAG,
    _JUDGE_CHARS_PER_TOKEN,
    _JUDGE_OUTPUT_TOKENS,
    _JUDGE_TRUNC_MARKER,
    RetrievalState,
    SemanticFilterTool,
    _judge_doc_char_budget,
)


# ---- fakes ------------------------------------------------------------------------


class _Resp:
    def __init__(self, text: str):
        self.text = text


class FakeLLMClient:
    """Judge verdict = `verdict_fn(document_text)`; records judge + embed calls."""

    def __init__(self, verdict_fn=None):
        self.verdict_fn = verdict_fn or (lambda text: True)
        self.judged_texts: list[str] = []
        self.judge_max_output_tokens: list[int | None] = []
        self.embed_calls: list[tuple[str, str | None]] = []

    def call(self, *, system, user, temperature, model, ctx, call_site, provider_order=None,
             max_output_tokens=None):
        doc_text = user.split("\n\nDocument:\n", 1)[1]
        self.judged_texts.append(doc_text)
        self.judge_max_output_tokens.append(max_output_tokens)
        return _Resp("TRUE" if self.verdict_fn(doc_text) else "FALSE")

    def embed_query(self, text, *, model=None, ctx=None):
        self.embed_calls.append((text, model))
        return [0.0, 0.0]


class FakeChroma:
    """Backed by a flat chunk list; records every query()/get() kwargs. The metadata
    `where` filter is NOT applied (tests only assert what was passed), but paging is."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.query_kwargs: list[dict] = []
        self.get_kwargs: list[dict] = []

    def query(self, **kwargs):
        self.query_kwargs.append(kwargs)
        rows = self.chunks[: kwargs["n_results"]]
        return {
            "ids": [[c["chunk_id"] for c in rows]],
            "documents": [[c["text"] for c in rows]],
            "metadatas": [[{"doc_id": c["doc_id"]} for c in rows]],
        }

    def get(self, ids=None, where=None, include=None, limit=None, offset=None):
        self.get_kwargs.append({"ids": ids, "where": where, "include": include, "limit": limit, "offset": offset})
        if ids is not None:
            wanted = set(ids)
            rows = [c for c in self.chunks if c["chunk_id"] in wanted]
            return {"ids": [c["chunk_id"] for c in rows], "documents": [c["text"] for c in rows]}
        rows = self.chunks[offset or 0:]
        if limit is not None:
            rows = rows[:limit]
        return {"ids": [c["chunk_id"] for c in rows], "metadatas": [{"doc_id": c["doc_id"]} for c in rows]}


class FakeCtx:
    def __init__(self):
        self.events: list[tuple[str, dict | None]] = []

    def emit(self, message, kind=None, data=None):
        self.events.append((message, data))


_CHUNKS = [
    {"chunk_id": "d1_0", "doc_id": "d1", "text": "apple pie recipe"},
    {"chunk_id": "d2_0", "doc_id": "d2", "text": "banana bread"},
    {"chunk_id": "d1_1", "doc_id": "d1", "text": "more apples"},
    {"chunk_id": "d3_0", "doc_id": "d3", "text": "cherry cake"},
]
_DOC_MAP = {"d1": "apple document", "d2": "banana document", "d3": "cherry document"}


def _tool(verdict_fn=None, *, chroma=True, ctx=None, **kwargs) -> tuple[SemanticFilterTool, FakeLLMClient, FakeChroma | None]:
    client = FakeLLMClient(verdict_fn)
    collection = FakeChroma(list(_CHUNKS)) if chroma else None
    tool = SemanticFilterTool(
        client, dict(_DOC_MAP), "judge-model",
        chroma_collection=collection,
        emb_model_id="emb-model" if chroma else None,
        ctx=ctx,
        **kwargs,
    )
    return tool, client, collection


# ---- validation -------------------------------------------------------------------


def test_validation_errors():
    tool, client, collection = _tool()
    cases = [
        (dict(predicate="  "), "`predicate` is required"),
        (dict(predicate="p", doc_ids=["d1"], metadata_filter={"year": "1946"}), "EITHER doc_ids OR corpus-mode"),
        (dict(predicate="p", search_str="q"), "search_str requires top_k"),
        (dict(predicate="p"), "provide doc_ids, or a metadata_filter and/or top_k"),
    ]
    for kwargs, needle in cases:
        out = tool(**kwargs)
        assert out[SEMFILTER_RESULT_TAG] is True and needle in out["error"], (kwargs, out)
    # No judge / chroma traffic on validation failures.
    assert not client.judged_texts and not collection.query_kwargs and not collection.get_kwargs


def test_corpus_mode_requires_wiring():
    tool, _, _ = _tool(chroma=False)
    out = tool("p", metadata_filter={"year": "1946"})
    assert "corpus mode is not available" in out["error"]

    tool2 = SemanticFilterTool(FakeLLMClient(), dict(_DOC_MAP), "m", chroma_collection=FakeChroma(list(_CHUNKS)))
    out2 = tool2("p", top_k=5)
    assert "top_k is not available" in out2["error"]


# ---- doc_ids mode (legacy path) ---------------------------------------------------


def test_doc_ids_mode_regression():
    tool, client, collection = _tool(lambda text: "apple" in text)
    out = tool("about apples", doc_ids=["d1", "d2"])
    assert out == {SEMFILTER_RESULT_TAG: True, "kept_doc_ids": ["d1"], "n_in": 2, "n_out": 1}
    assert "chunks" not in out and "summary" not in out
    assert not collection.query_kwargs and not collection.get_kwargs  # no corpus traffic

    out2 = tool("about bananas", doc_ids="d2")  # bare-string coercion
    assert out2["kept_doc_ids"] == []
    assert out2["n_in"] == 1


# ---- corpus mode: vector prefilter ------------------------------------------------


def test_vector_mode_embeds_search_str_or_predicate():
    tool, client, collection = _tool()
    tool("a long verbose predicate", top_k=3, search_str="short query")
    assert client.embed_calls[-1] == ("short query", "emb-model")
    assert collection.query_kwargs[-1]["n_results"] == 3
    assert "where" not in collection.query_kwargs[-1]  # nothing to filter or exclude

    tool("a long verbose predicate", top_k=2)
    assert client.embed_calls[-1] == ("a long verbose predicate", "emb-model")


def test_vector_mode_result_payload():
    tool, client, _ = _tool(lambda text: "apple" in text)
    state = RetrievalState()
    tool.bind_retrieval_state(state)
    out = tool("about apples", top_k=4)
    # Judged the deduped parent docs' FULL text (from document_map), not chunk text.
    assert sorted(client.judged_texts) == ["apple document", "banana document", "cherry document"]
    assert out["kept_doc_ids"] == ["d1"] and out["n_in"] == 3 and out["n_out"] == 1
    # Snippets: only kept docs' candidate chunks, in relevance (candidate) order, marked seen.
    assert [c["chunk_id"] for c in out["chunks"]] == ["d1_0", "d1_1"]
    assert "apple pie recipe" in out["chunks"][0]["text"]
    assert state.seen_chunk_ids == {"d1_0", "d1_1"}
    assert "kept 1/3" in out["summary"] and "d1" in out["summary"]


def test_vector_mode_over_doc_cap():
    tool, client, _ = _tool(max_candidate_docs=2)
    out = tool("p", top_k=4)  # 4 chunks → 3 distinct docs > cap 2
    assert "over the 2-document cap" in out["error"]
    assert not client.judged_texts  # cap trips before any judge spend


# ---- corpus mode: metadata-only --------------------------------------------------


def test_metadata_mode_where_has_pruned_but_not_seen():
    tool, _, collection = _tool()
    state = RetrievalState(
        pruned_chunk_ids={"px"}, pruned_doc_ids={"pd"},
        seen_chunk_ids={"sx"}, seen_doc_ids={"sd"},
    )
    tool.bind_retrieval_state(state)
    tool("p", metadata_filter={"year": "1946"})
    where = collection.get_kwargs[0]["where"]
    assert {"year": "1946"} in where["$and"]
    assert {"doc_id": {"$nin": ["pd"]}} in where["$and"]
    assert {"chunk_id": {"$nin": ["px"]}} in where["$and"]
    assert "sx" not in str(where) and "sd" not in str(where)


def test_metadata_mode_two_phase_fetch():
    tool, client, collection = _tool(lambda text: "banana" in text)
    out = tool("about bananas", metadata_filter={"kind": "any"})
    # Phase 1: ids/metadatas-only paged scan — no texts pulled for the full match set.
    assert collection.get_kwargs[0]["include"] == ["metadatas"]
    assert collection.get_kwargs[0]["limit"] == SemanticFilterTool._GET_PAGE_SIZE
    # Every matched doc judged (thread-pool order is arbitrary); phase 2 fetches texts
    # ONLY for the kept doc's chunks.
    assert sorted(client.judged_texts) == ["apple document", "banana document", "cherry document"]
    assert collection.get_kwargs[1]["ids"] == ["d2_0"]
    assert out["kept_doc_ids"] == ["d2"]
    assert [c["chunk_id"] for c in out["chunks"]] == ["d2_0"]
    assert "banana bread" in out["chunks"][0]["text"]


def test_metadata_mode_over_doc_cap_before_judging():
    tool, client, collection = _tool(max_candidate_docs=2)
    out = tool("p", metadata_filter={"kind": "any"})
    assert "matched more than 2 candidate documents" in out["error"]
    assert not client.judged_texts
    assert len(collection.get_kwargs) == 1  # no phase-2 text fetch either


def test_output_cap_truncates_and_marks_only_rendered_seen():
    # Cap of 4 tokens = 16 chars: the first chunk always renders, the second is dropped.
    tool, _, _ = _tool(lambda text: "apple" in text, max_output_tokens=4)
    state = RetrievalState()
    tool.bind_retrieval_state(state)
    out = tool("about apples", top_k=4)
    assert [c["chunk_id"] for c in out["chunks"]] == ["d1_0"]
    assert "showing 1 of 2 chunk(s)" in out["truncation_note"]
    assert state.seen_chunk_ids == {"d1_0"}  # the omitted chunk stays fetchable later


def test_zero_candidates_payload():
    tool, _, collection = _tool()
    collection.chunks = []
    out = tool("p", metadata_filter={"year": "3000"})
    assert out["kept_doc_ids"] == [] and out["chunks"] == [] and not out["summary"]


# ---- trace event ------------------------------------------------------------------


def test_trace_event_records_mode_and_inputs():
    ctx = FakeCtx()
    tool, _, _ = _tool(ctx=ctx)
    tool("about fruit", top_k=4, search_str="fruit", metadata_filter={"year": "1946"})
    message, data = ctx.events[-1]
    assert message.startswith("semantic_filter n_in=")  # load-bearing: tool_metrics keys on this
    assert data["mode"] == "vector" and data["top_k"] == 4 and data["search_str"] == "fruit"
    assert data["metadata_filter"] == {"year": "1946"}
    assert data["n_candidate_chunks"] == 4 and data["n_candidate_docs"] == 3

    tool("about fruit", doc_ids=["d1"])
    assert ctx.events[-1][1]["mode"] == "doc_ids"


# ---- SearchAgent integration: bind_retrieval_state + rendering -------------------------------


def _agent(extra_tools=()) -> SearchAgent:
    config = SearchAgentConfig(
        name="t", emb_provider="openrouter", emb_model_id="emb", llm_provider="openrouter",
        llm_model="m", llm_max_retries=0, llm_retry_initial_delay_s=0.0,
        llm_model_rpm={}, llm_default_rpm=1e9, llm_model_tpm={}, llm_default_tpm=None, llm_prices={},
        llm_context_limits={},
    )
    return SearchAgent(
        config=config, document_map={}, chroma_collection=FakeChroma([]),
        llm_client=FakeLLMClient(), extra_tools=extra_tools,
    )


def test_bind_retrieval_state_hook():
    bound = SemanticFilterTool(FakeLLMClient(), {}, "m")
    class NoBind(Tool):
        name = "nobind"
        doc = "### nobind()\nDoes nothing."
        def __call__(self):
            return None
    agent = _agent(extra_tools=(bound, NoBind()))  # tool without bind_retrieval_state must not break init
    assert bound._state is agent._state


def _blocks(payload: dict) -> list:
    return _agent()._blocks_from_output(CodeOutput(output=payload, logs=""))


def test_render_corpus_payload():
    blocks = _blocks({
        SEMFILTER_RESULT_TAG: True,
        "summary": "[semantic_filter] kept 1/3",
        "chunks": [{"chunk_id": "c1", "doc_id": "d1", "text": "snippet"}],
        "truncation_note": "[semantic_filter output truncated]",
        "kept_doc_ids": ["d1"], "n_in": 3, "n_out": 1,
    })
    assert isinstance(blocks[0], TextBlock) and "kept 1/3" in blocks[0].text
    assert isinstance(blocks[1], ChunkBlock) and blocks[1].chunk_id == "c1" and blocks[1].doc_id == "d1"
    assert isinstance(blocks[2], TextBlock) and "truncated" in blocks[2].text


def test_render_error_and_empty_and_doc_ids_payloads():
    [err] = _blocks({SEMFILTER_RESULT_TAG: True, "error": "semantic_filter error: boom"})
    assert err.text == "[error]\nsemantic_filter error: boom"

    [empty] = _blocks({SEMFILTER_RESULT_TAG: True, "summary": "", "chunks": [],
                       "kept_doc_ids": [], "n_in": 0, "n_out": 0})
    assert empty.text == EMPTY_RESULT_MESSAGE

    [ids] = _blocks({SEMFILTER_RESULT_TAG: True, "kept_doc_ids": ["d1", "d2"], "n_in": 5, "n_out": 2})
    assert "kept 2 of 5 document(s)" in ids.text and "['d1', 'd2']" in ids.text


def test_rendered_chunks_are_redactable():
    agent = _agent()
    agent._state.pruned_chunk_ids.add("c1")
    block = ChunkBlock(chunk_id="c1", doc_id="d1", text="snippet")
    assert agent._block_is_visible(block) is False


# ---- judge output cap + context-limit truncation ---------------------------------


def test_judge_call_caps_output_at_256():
    tool, client, _ = _tool(lambda t: True)
    tool("about apples", doc_ids=["d1", "d2"])
    # Every judge call carries the 256-token output cap (the verdict is one word).
    assert client.judge_max_output_tokens == [_JUDGE_OUTPUT_TOKENS, _JUDGE_OUTPUT_TOKENS]


def test_context_limit_truncates_only_oversized_docs():
    big, small = "z" * 500_000, "tiny doc"
    ctx = FakeCtx()
    client = FakeLLMClient(lambda t: True)
    limit = 8000  # tokens
    tool = SemanticFilterTool(
        client, {"big": big, "small": small}, "judge-model",
        ctx=ctx, context_limits={"judge-model": limit},
    )
    tool("some predicate", doc_ids=["big", "small"])
    budget = _judge_doc_char_budget(limit, "some predicate")
    # The oversized doc is head-truncated to exactly the budget and carries the marker;
    # the small doc is sent verbatim.
    assert len(client.judged_texts[0]) == budget
    assert client.judged_texts[0].endswith(_JUDGE_TRUNC_MARKER)
    assert client.judged_texts[1] == small
    # The truncation is estimated to fit under the limit (with the safety margin).
    assert budget / _JUDGE_CHARS_PER_TOKEN < limit
    # The trace event reports exactly one truncation.
    assert ctx.events[-1][1]["n_truncated"] == 1


def test_no_context_limit_sends_full_text():
    big = "z" * 100_000
    client = FakeLLMClient(lambda t: True)
    tool = SemanticFilterTool(client, {"big": big}, "judge-model")  # no context_limits
    assert tool._context_limit is None
    tool("p", doc_ids=["big"])
    assert client.judged_texts[0] == big  # untouched


def test_context_limit_resolved_by_substring_match():
    # The tool resolves its judge model's limit via the shared exact-then-substring matcher.
    tool = SemanticFilterTool(
        FakeLLMClient(), dict(_DOC_MAP), "qwen/qwen3.6-35b-a3b",
        context_limits={"qwen3.6-35b-a3b": 262144},
    )
    assert tool._context_limit == 262144


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ok")
