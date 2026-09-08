"""Unit tests for `SemanticFilterTool` (candidate selection + judging in read/fetch modes), its
shared `RetrievalState` wiring, and the `SEMFILTER_RESULT_TAG` rendering in
`SearchAgent._blocks_from_output`.

No LLM / Chroma. Runs under pytest if installed, or standalone:
`python3 tests/test_semantic_filter_tool.py`.
"""

from __future__ import annotations

from types import SimpleNamespace

from skunk.config import SearchAgentConfig
from skunk.multi_turn_agent import ChunkBlock, TextBlock
from skunk.sandbox.local_python_executor import CodeOutput
from skunk.search_agent.search_agent import SearchAgent
from skunk.search_agent.search_tools import (
    EMPTY_RESULT_MESSAGE,
    SEMFILTER_RESULT_TAG,
    RetrievalState,
    SemanticFilterTool,
)


# ---- fakes ------------------------------------------------------------------------


class _Resp:
    def __init__(self, text: str):
        self.text = text


class FakeLLMClient:
    """Judge verdict = `verdict_fn(document_text)`; records judge + embed calls. `config`
    mirrors `LLMClient.config` — the tool resolves its default judge model and the judge
    model's context limit from it."""

    def __init__(self, verdict_fn=None, *, llm_model="agent-model", context_limits=None):
        self.verdict_fn = verdict_fn or (lambda text: True)
        self.config = SimpleNamespace(llm_model=llm_model, llm_context_limits=context_limits or {})
        self.judged_texts: list[str] = []
        self.judge_max_output_tokens: list[int | None] = []
        self.judge_disable_reasoning: list[bool] = []
        self.embed_calls: list[str] = []

    def call(self, *, system, messages, temperature, model, ctx, call_site, provider_order=None,
             max_output_tokens=None, disable_reasoning=False, usage_key="default"):
        doc_text = messages[0]["content"].split("\n\nDocument:\n", 1)[1]
        self.judged_texts.append(doc_text)
        self.judge_max_output_tokens.append(max_output_tokens)
        self.judge_disable_reasoning.append(disable_reasoning)
        return _Resp("TRUE" if self.verdict_fn(doc_text) else "FALSE")

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

    def query(self, **kwargs):
        self.query_kwargs.append(kwargs)
        rows = self.chunks[: kwargs["n_results"]]
        return {
            "ids": [[c["chunk_id"] for c in rows]],
            "documents": [[c["text"] for c in rows]],
            "metadatas": [[{"doc_id": c["doc_id"]} for c in rows]],
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
            "metadatas": [{"doc_id": c["doc_id"]} for c in rows],
        }


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


def _tool(verdict_fn=None, *, ctx=None, context_limits=None, **kwargs) -> tuple[SemanticFilterTool, FakeLLMClient, FakeChroma]:
    client = FakeLLMClient(verdict_fn, context_limits=context_limits)
    collection = FakeChroma(list(_CHUNKS))
    tool = SemanticFilterTool(
        collection, client, dict(_DOC_MAP), "judge-model",
        ctx=ctx,
        **kwargs,
    )
    return tool, client, collection


# ---- validation -------------------------------------------------------------------


def test_validation_errors():
    tool, client, collection = _tool()
    cases = [
        (dict(predicate="  ", fetch=True), "`predicate` is required"),
        # A selector is present (pattern), so the search_str/top_k pairing check is reached.
        (dict(predicate="p", fetch=True, pattern="x", search_str="q"), "search_str requires top_k"),
        (dict(predicate="p", fetch=True), "must provide one of"),
        (dict(predicate="p", fetch=True, metadata_filter={"y": "1"}, limit=5), "limit requires pattern"),
    ]
    for kwargs, needle in cases:
        out = tool(**kwargs)
        assert out[SEMFILTER_RESULT_TAG] is True and needle in out["error"], (kwargs, out)
    # No judge / chroma traffic on validation failures.
    assert not client.judged_texts and not collection.query_kwargs and not collection.get_kwargs


def test_read_only_requires_nonempty_working_set():
    tool, _, _ = _tool()  # fresh private state → empty working set
    try:
        tool("p", read=True)
    except AssertionError as e:
        assert "empty working set" in str(e)
    else:
        raise AssertionError("expected AssertionError on read from an empty working set")


def test_default_judge_model_is_client_default():
    # No explicit judge model => the client's `llm_model` (the agent model).
    tool = SemanticFilterTool(FakeChroma(list(_CHUNKS)), FakeLLMClient(), dict(_DOC_MAP))
    assert tool._model == "agent-model"


# ---- corpus mode: vector prefilter ------------------------------------------------


def test_vector_mode_embeds_search_str_or_predicate():
    tool, client, collection = _tool()
    tool("a long verbose predicate", fetch=True, top_k=3, search_str="short query")
    assert client.embed_calls[-1] == "short query"
    assert collection.query_kwargs[-1]["n_results"] == 3
    assert "where" not in collection.query_kwargs[-1]  # nothing fetched/pruned yet → no filter

    tool2, client2, collection2 = _tool()
    tool2("a long verbose predicate", fetch=True, top_k=2)
    assert client2.embed_calls[-1] == "a long verbose predicate"


def test_vector_mode_read_and_fetch_payload():
    state = RetrievalState()
    tool, client, _ = _tool(lambda text: "apple" in text, state=state)
    out = tool("about apples", read=True, fetch=True, top_k=4)
    # Judged the deduped parent docs' FULL text (from document_map), not chunk text.
    assert sorted(client.judged_texts) == ["apple document", "banana document", "cherry document"]
    assert out["kept_doc_ids"] == ["d1"] and out["n_in"] == 3 and out["n_out"] == 1
    # Kept docs' candidate chunks in (doc_id, chunk_id) order, present under BOTH keys.
    assert [c["chunk_id"] for c in out["read_chunks"]] == ["d1_0", "d1_1"]
    assert [c["chunk_id"] for c in out["fetched_chunks"]] == ["d1_0", "d1_1"]
    assert "apple pie recipe" in out["read_chunks"][0]["text"]
    assert "est_num_tokens" in out["read_chunks"][0]
    # State: kept chunks AND their docs are marked read + fetched.
    assert state.read_chunk_ids == {"d1_0", "d1_1"} and state.fetched_chunk_ids == {"d1_0", "d1_1"}
    assert state.read_doc_ids == {"d1"} and state.fetched_doc_ids == {"d1"}
    assert "kept 1/3" in out["summary"] and "d1" in out["summary"]


def test_fetch_only_populates_working_set_not_read():
    state = RetrievalState()
    tool, _, _ = _tool(lambda text: "apple" in text, state=state)
    out = tool("about apples", fetch=True, top_k=4)
    assert out["read_chunks"] == [] and [c["chunk_id"] for c in out["fetched_chunks"]] == ["d1_0", "d1_1"]
    assert state.fetched_chunk_ids == {"d1_0", "d1_1"} and state.fetched_doc_ids == {"d1"}
    assert not state.read_chunk_ids and not state.read_doc_ids


def test_vector_mode_over_doc_cap():
    tool, client, _ = _tool()
    tool._MAX_CANDIDATE_DOCS = 2  # instance attr shadows the class constant
    out = tool("p", fetch=True, top_k=4)  # 4 chunks → 3 distinct docs > cap 2
    assert "over the 2-document cap" in out["error"]
    assert not client.judged_texts  # cap trips before any judge spend


# ---- where-clause construction per mode ------------------------------------------


def test_fetch_mode_excludes_fetched_and_pruned():
    state = RetrievalState(
        pruned_chunk_ids={"px"}, pruned_doc_ids={"pd"},
        fetched_chunk_ids={"fx"}, fetched_doc_ids={"fd"},
    )
    tool, _, collection = _tool(state=state)
    tool("p", fetch=True, metadata_filter={"year": "1946"})
    where = collection.get_kwargs[0]["where"]
    assert {"year": "1946"} in where["$and"]
    assert {"doc_id": {"$nin": ["fd", "pd"]}} in where["$and"]
    assert {"chunk_id": {"$nin": ["fx", "px"]}} in where["$and"]


def test_fetch_and_read_mode_excludes_pruned_and_fetched():
    # Dedup is on the FETCH axis: fetch+read excludes working-set (fetched) + pruned material,
    # but NOT read material — already-read docs outside the working set don't exist (read marks
    # fetched too), and re-reads are always allowed.
    state = RetrievalState(
        pruned_chunk_ids={"px"}, pruned_doc_ids={"pd"},
        read_chunk_ids={"sx"}, read_doc_ids={"sd"},
        fetched_chunk_ids={"fx"}, fetched_doc_ids={"fd"},
    )
    tool, _, collection = _tool(state=state)
    tool("p", read=True, fetch=True, metadata_filter={"year": "1946"})
    where = collection.get_kwargs[0]["where"]
    assert {"year": "1946"} in where["$and"]
    assert {"doc_id": {"$nin": ["fd", "pd"]}} in where["$and"]
    assert {"chunk_id": {"$nin": ["fx", "px"]}} in where["$and"]


def test_read_mode_includes_working_set_as_union():
    # Read-only mode restricts candidates to the working set: chunk ∈ fetched_chunks OR
    # doc ∈ fetched_docs (a doc opened via read_document exposes ALL its chunks), minus
    # pruned material only — read chunks stay eligible, so the same doc/chunk can be
    # re-judged under a different predicate.
    state = RetrievalState(
        fetched_chunk_ids={"fx"}, fetched_doc_ids={"fd"},
        pruned_chunk_ids={"px"}, read_chunk_ids={"sx"},
    )
    tool, _, collection = _tool(state=state)
    tool("p", read=True)  # no selector needed: the working set IS the candidate pool
    where = collection.get_kwargs[0]["where"]
    assert {"$or": [{"doc_id": {"$in": ["fd"]}}, {"chunk_id": {"$in": ["fx"]}}]} in where["$and"]
    assert {"chunk_id": {"$nin": ["px"]}} in where["$and"]


def test_read_mode_allows_rereading_same_docs():
    # Two read-only calls over the same working set: the second call's filter must not
    # exclude what the first call read — different predicates over the same docs re-judge
    # them and return them again.
    state = RetrievalState(fetched_chunk_ids={"d1_0", "d1_1"}, fetched_doc_ids={"d1"})
    tool, _, collection = _tool(lambda text: "apple" in text, state=state)
    first = tool("about apples", read=True)
    assert first["kept_doc_ids"] == ["d1"] and state.read_doc_ids == {"d1"}
    second = tool("still about apples", read=True)
    assert second["kept_doc_ids"] == ["d1"]  # re-read allowed despite read marks
    # the second call's exclusions contain no read ids
    where = collection.get_kwargs[1]["where"]
    clauses = where["$and"] if "$and" in where else [where]
    nin_values = [v for c in clauses for f, spec in c.items() if isinstance(spec, dict) and "$nin" in spec for v in spec["$nin"]]
    assert "d1" not in nin_values and "d1_0" not in nin_values and "d1_1" not in nin_values


# ---- corpus mode: grep prefilter --------------------------------------------------


def test_grep_mode_passes_pattern_and_limit():
    tool, client, collection = _tool()
    tool("p", fetch=True, pattern="(?i)apple", limit=7)
    got = collection.get_kwargs[0]
    assert got["where_document"] == {"$regex": "(?i)apple"} and got["limit"] == 7
    # The fake ignores the regex, so all 3 parent docs get judged.
    assert sorted(client.judged_texts) == ["apple document", "banana document", "cherry document"]


# ---- zero candidates + trace event ------------------------------------------------


def test_zero_candidates_payload():
    tool, _, collection = _tool()
    collection.chunks = []
    out = tool("p", fetch=True, metadata_filter={"year": "3000"})
    assert out["kept_doc_ids"] == [] and out["read_chunks"] == [] and out["fetched_chunks"] == []
    assert not out["summary"] and out["n_in"] == 0 and out["n_out"] == 0


def test_trace_event_records_mode_and_inputs():
    ctx = FakeCtx()
    tool, _, _ = _tool(ctx=ctx)
    tool("about fruit", fetch=True, top_k=4, search_str="fruit", metadata_filter={"year": "1946"})
    message, data = ctx.events[-1]
    assert message.startswith("semantic_filter n_in=")  # load-bearing: tool_metrics keys on this
    assert data["mode"] == "vector" and data["top_k"] == 4 and data["search_str"] == "fruit"
    assert data["metadata_filter"] == {"year": "1946"}
    assert data["n_candidate_chunks"] == 4 and data["n_candidate_docs"] == 3

    tool2, _, _ = _tool(ctx=ctx)
    tool2("about fruit", fetch=True, metadata_filter={"year": "1946"})
    assert ctx.events[-1][1]["mode"] == "metadata"


# ---- SearchAgent integration: first-class tool wiring + rendering ----------------


def _agent(**kwargs) -> SearchAgent:
    config = SearchAgentConfig(name="t", agent_id="t")
    return SearchAgent(
        config=config, document_map={}, chroma_collection=FakeChroma([]),
        llm_client=FakeLLMClient(), **kwargs,
    )


def test_semantic_filter_is_first_class_and_shares_agent_state():
    # `include_semantic_filter=True` makes the agent construct the tool itself, sharing its
    # per-question RetrievalState by reference, like search/grep/read/prune.
    agent = _agent(include_semantic_filter=True)
    tool = next(t for t in agent._tools if t.name == "semantic_filter")
    assert tool._state is agent._state
    # Off by default: the vanilla agent has no semantic_filter tool.
    assert all(t.name != "semantic_filter" for t in _agent()._tools)


def _blocks(payload: dict) -> list:
    return _agent()._blocks_from_output(CodeOutput(output=payload, logs=""))


def test_render_read_payload():
    blocks = _blocks({
        SEMFILTER_RESULT_TAG: True,
        "summary": "[semantic_filter] kept 1/3",
        "read_chunks": [{"chunk_id": "c1", "doc_id": "d1", "est_num_tokens": 2, "header": "h", "text": "snippet"}],
        "fetched_chunks": [],
        "kept_doc_ids": ["d1"], "n_in": 3, "n_out": 1,
    })
    assert isinstance(blocks[0], TextBlock) and "kept 1/3" in blocks[0].text
    assert isinstance(blocks[1], ChunkBlock) and blocks[1].chunk_id == "c1" and blocks[1].doc_id == "d1"


def test_render_fetched_payload_is_a_digest():
    chunk = {"chunk_id": "c1", "doc_id": "d1", "est_num_tokens": 7, "header": "[c1 header]", "text": "snippet"}
    blocks = _blocks({
        SEMFILTER_RESULT_TAG: True,
        "summary": "[semantic_filter] kept 1/3",
        "read_chunks": [],
        "fetched_chunks": [chunk],
        "kept_doc_ids": ["d1"], "n_in": 3, "n_out": 1,
    })
    # Fetch-only renders the summary plus a digest TextBlock (no ChunkBlocks → no chunk text
    # enters the agent's context).
    assert isinstance(blocks[0], TextBlock) and "kept 1/3" in blocks[0].text
    assert isinstance(blocks[1], TextBlock)
    assert "Retrieved 1 chunks from 1 documents" in blocks[1].text and "[c1 header]" in blocks[1].text
    assert all(not isinstance(b, ChunkBlock) for b in blocks)


def test_render_summary_only_when_all_rejected():
    # kept 0/N: no chunks under either key, but the summary must still reach the agent.
    [summary] = _blocks({
        SEMFILTER_RESULT_TAG: True,
        "summary": "[semantic_filter] kept 0/3 candidate document(s) matching the predicate. kept_doc_ids=[].",
        "read_chunks": [], "fetched_chunks": [],
        "kept_doc_ids": [], "n_in": 3, "n_out": 0,
    })
    assert isinstance(summary, TextBlock) and "kept 0/3" in summary.text


def test_render_error_and_empty_payloads():
    [err] = _blocks({SEMFILTER_RESULT_TAG: True, "error": "semantic_filter error: boom"})
    assert err.text == "[error]\nsemantic_filter error: boom"

    [empty] = _blocks({SEMFILTER_RESULT_TAG: True, "summary": "", "read_chunks": [],
                       "fetched_chunks": [], "kept_doc_ids": [], "n_in": 0, "n_out": 0})
    assert empty.text == EMPTY_RESULT_MESSAGE


def test_rendered_chunks_are_redactable():
    agent = _agent()
    agent._state.pruned_chunk_ids.add("c1")
    block = ChunkBlock(chunk_id="c1", doc_id="d1", text="snippet")
    assert agent._block_is_visible(block) is False
    # Trim-time redaction (redacted_* sets) hides blocks the same way.
    agent._make_block_invisible(chunk_id="c2")
    assert agent._block_is_visible(ChunkBlock(chunk_id="c2", doc_id="d1", text="s")) is False
    agent._make_block_invisible(doc_id="d9")
    assert agent._block_is_visible(ChunkBlock(chunk_id=None, doc_id="d9", text="s")) is False


# ---- judge output cap + context-limit truncation ---------------------------------


def test_judge_call_caps_output():
    tool, client, _ = _tool(lambda t: True)
    tool("about apples", fetch=True, top_k=2)  # top_k vector prefilter → 2 candidate docs (d1, d2)
    # Every judge call carries the tool's default output cap (the verdict is one word).
    cap = SemanticFilterTool._JUDGE_MAX_OUTPUT_TOKENS
    assert client.judge_max_output_tokens == [cap, cap]

    tool2, client2, _ = _tool(lambda t: True, judge_max_output_tokens=2048)
    tool2("about apples", fetch=True, top_k=2)
    # A reasoning judge gets its configured thinking headroom instead.
    assert client2.judge_max_output_tokens == [2048, 2048]


def test_judge_reasoning_disabled_by_default():
    tool, client, _ = _tool(lambda t: True)
    tool("about apples", fetch=True, top_k=2)
    # Judge calls disable thinking by default (the verdict is one token)...
    assert client.judge_disable_reasoning == [True, True]

    tool2, client2, _ = _tool(lambda t: True, disable_judge_reasoning=False)
    tool2("about apples", fetch=True, top_k=2)
    # ...but a reasoning-mandated judge model can opt back in.
    assert client2.judge_disable_reasoning == [False, False]


def test_context_limit_truncates_only_oversized_docs():
    big, small = "z" * 500_000, "tiny doc"
    ctx = FakeCtx()
    limit = 8000  # tokens
    # The limit is resolved from the client's config (`llm_context_limits`), not a tool arg.
    client = FakeLLMClient(lambda t: True, context_limits={"judge-model": limit})
    # Metadata-mode candidates come from the collection (big, small); the judged text is
    # pulled from the document_map, so that's what the context-limit truncation acts on.
    chroma = FakeChroma([
        {"chunk_id": "big_0", "doc_id": "big", "text": big},
        {"chunk_id": "small_0", "doc_id": "small", "text": small},
    ])
    tool = SemanticFilterTool(chroma, client, {"big": big, "small": small}, "judge-model", ctx=ctx)
    assert tool._context_limit == limit
    tool("some predicate", fetch=True, metadata_filter={"any": "x"})
    # The oversized doc is head-truncated (marker appended) to fit the judge's window;
    # the small doc is sent verbatim.
    marker = SemanticFilterTool._JUDGE_TRUNC_MARKER
    assert client.judged_texts[0].endswith(marker) and len(client.judged_texts[0]) < len(big)
    assert client.judged_texts[1] == small
    # The truncated text honors the tool's own token budget for this predicate.
    from skunk.common import estimate_tokens
    budget = tool._judge_doc_token_budget("some predicate")
    assert estimate_tokens(client.judged_texts[0][: -len(marker)]) <= budget
    # The trace event reports exactly one truncation.
    assert ctx.events[-1][1]["n_truncated"] == 1


def test_no_context_limit_sends_full_text():
    big = "z" * 100_000
    client = FakeLLMClient(lambda t: True)  # no llm_context_limits entry for the judge model
    chroma = FakeChroma([{"chunk_id": "big_0", "doc_id": "big", "text": big}])
    tool = SemanticFilterTool(chroma, client, {"big": big}, "judge-model")
    assert tool._context_limit is None
    tool("p", fetch=True, metadata_filter={"any": "x"})
    assert client.judged_texts[0] == big  # untouched


def test_context_limit_resolved_by_substring_match():
    # The tool resolves its judge model's limit via the shared exact-then-substring matcher.
    tool = SemanticFilterTool(
        FakeChroma(list(_CHUNKS)),
        FakeLLMClient(context_limits={"qwen3.6-35b-a3b": 262144}),
        dict(_DOC_MAP), "qwen/qwen3.6-35b-a3b",
    )
    assert tool._context_limit == 262144


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ok")
