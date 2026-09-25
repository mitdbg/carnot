"""Unit tests for the collection-building agents' wiring (qatfd.agents.collection_agent).

Covers `CollectionAgent._build_tools` (the tool set the Bootstrap / Enrich agents get) and
`_handle_tool_call` (how each tool's payload is rendered into observation blocks + traced).
The LLM client is faked; chroma is two PersistentClient shards behind a MergedClient.

Run from the qatfd dir: python3 -m pytest tests/test_collection_agent.py
"""

from __future__ import annotations

from types import SimpleNamespace

import chromadb
import pytest
from skunk.agents.multi_turn_agent import ChunkBlock, TextBlock
from skunk.config import StorageConfig
from skunk.sandbox.local_python_executor import CodeOutput

from qatfd.agents.bootstrap_agent import BootstrapAgent
from qatfd.agents.enrich_agent import EnrichAgent
from qatfd.config import BootstrapConfig, EnrichConfig
from qatfd.merged_chroma import MergedClient, Shard
from qatfd.tools import MapResult, ResultSet, SearchResult

BASE = "corpus"


@pytest.fixture
def client(tmp_path):
    a = chromadb.PersistentClient(path=str(tmp_path / "a"))
    b = chromadb.PersistentClient(path=str(tmp_path / "b"))
    for cl, prefix, x0 in ((a, "a", 0.0), (b, "b", 10.0)):
        c = cl.create_collection(f"{BASE}_r{prefix == 'b'}".replace("True", "1").replace("False", "0"), metadata={"fields": "  - doc_id (str): id\n"})
        c.add(
            ids=[f"{prefix}{i}" for i in range(4)],
            embeddings=[[x0 + i, 0.0] for i in range(4)],
            documents=[f"{prefix} text {i} federal reserve" for i in range(4)],
            metadatas=[{"doc_id": f"D{prefix}{i}", "type": "text"} for i in range(4)],
        )
    return MergedClient(BASE, [Shard(a, f"{BASE}_r0", "A"), Shard(b, f"{BASE}_r1", "B")])


class _Tracer:
    def __init__(self):
        self.events = []

    def emit(self, id, level="info", **kw):
        self.events.append((id, kw.get("data")))


def _fake_llm_client():
    return SimpleNamespace(config=SimpleNamespace(llm_model="fake/model", llm_context_limits={"fake": 8000}))


def _agent(client, cls=BootstrapAgent, cfg_cls=BootstrapConfig, **cfg_kw):
    config = cfg_cls(name="collection_agent", agent_id="agent_0", search_preview_chunks=2, search_preview_chars=12, map_preview_samples=1, **cfg_kw)
    storage = StorageConfig(collection_name=BASE, chroma_server_host="h", chroma_server_port=1)
    return cls(config, document_map={}, chroma_client=client, llm_client=_fake_llm_client(), storage_config=storage, additional_notes="notes")


def _tools(agent) -> dict:
    return {t.name: t for t in agent._tools}


def _render(agent, output, logs=""):
    tracer = _Tracer()
    blocks = agent._handle_tool_call(SimpleNamespace(tracer=tracer), CodeOutput(output=output, logs=logs))
    return blocks, tracer.events


def _texts(blocks) -> str:
    return "\n".join(b.text for b in blocks)


# ---- construction ----------------------------------------------------------------------------------


@pytest.mark.parametrize("cls,cfg_cls", [(BootstrapAgent, BootstrapConfig), (EnrichAgent, EnrichConfig)])
def test_build_tools_and_prompt(client, cls, cfg_cls):
    agent = _agent(client, cls, cfg_cls)
    names = set(_tools(agent))
    assert names == {
        "read_document", "search_corpus", "grep_corpus", "semantic_map", "map",
        "create_collection", "add_to_collection", "copy_collection", "merge_collections", "delete_collection", "list_collections",
    }
    tools = _tools(agent)
    assert tools["search_corpus"]._include_embeddings is False and tools["grep_corpus"]._include_embeddings is False
    assert tools["create_collection"]._base_collection_name == BASE and tools["create_collection"]._agent_id == "agent_0"
    assert tools["create_collection"]._max_copy_chunks == cfg_cls.max_copy_chunks
    # the system prompt carries every tool docstring, the collection-building guidance, and the notes
    assert "### create_collection(" in agent._system_prompt and "## Building Collections" in agent._system_prompt
    assert "notes" in agent._system_prompt


def test_tools_operate_end_to_end_through_the_agent(client):
    tools = _tools(_agent(client))
    hits = tools["grep_corpus"]([BASE], "federal", limit=3)
    assert isinstance(hits, ResultSet) and len(hits) == 3 and hits.results[BASE][0].embedding is None
    out = tools["create_collection"]("ws_fed", "fed chunks", hits)
    assert out["num_chunks"] == 3
    assert [s["name"] for s in tools["list_collections"]()["collections"]] == [BASE, "ws_fed"]


# ---- rendering -------------------------------------------------------------------------------------


def _hits(collection, ids):
    return [SearchResult(i, f"D{i}", f"[hdr {i}]", "x" * 40, {"doc_id": f"D{i}"}, None) for i in ids]


def test_render_search_results_as_summary(client):
    agent = _agent(client)
    rs = ResultSet({"tool": "search_corpus", "tool_kwargs": {"q": 1}, "results": {BASE: _hits(BASE, ["a0", "a1", "a2"]), "ws": _hits("ws", ["b0"])}})
    blocks, events = _render(agent, rs, logs="printed")
    assert isinstance(blocks[0], TextBlock) and blocks[0].text == "[stdout]\nprinted"
    text = _texts(blocks)
    assert "search_corpus: 4 chunk(s) from 4 doc(s) across 2 collection(s)" in text
    assert f"collection={BASE}: 3 chunk(s) / 3 doc(s) (showing the first 2)" in text
    assert "collection=ws: 1 chunk(s) / 1 doc(s)\n" in text
    chunk_blocks = [b for b in blocks if isinstance(b, ChunkBlock)]
    assert [b.chunk_id for b in chunk_blocks] == ["a0", "a1", "b0"]  # preview capped per collection
    assert chunk_blocks[0].text.startswith("[hdr a0]\nxxxxxxxxxxxx …[28 more chars]")
    assert events == [("search_corpus_tool_call", {"tool": "search_corpus", "tool_kwargs": {"q": 1}, "results": [
        {"collection": BASE, "chunk_id": "a0", "doc_id": "Da0"}, {"collection": BASE, "chunk_id": "a1", "doc_id": "Da1"},
        {"collection": BASE, "chunk_id": "a2", "doc_id": "Da2"}, {"collection": "ws", "chunk_id": "b0", "doc_id": "Db0"},
    ], "error": None})]

    # set-algebra results, empty results, and errors
    combined = rs | ResultSet({"tool": "grep_corpus", "tool_kwargs": {}, "results": {}})
    assert "result_set: 4 chunk(s)" in _texts(_render(agent, combined)[0])
    assert _texts(_render(agent, ResultSet({"tool": "grep_corpus", "tool_kwargs": {}, "results": {}}))[0]) == "No results found."
    err = ResultSet({"tool": "grep_corpus", "tool_kwargs": {}, "results": {}, "error": "bad regex"})
    assert _texts(_render(agent, err)[0]) == "[error]\nbad regex"


def test_render_map_results(client):
    agent = _agent(client)
    out = {
        "tool": "map",
        "tool_kwargs": {},
        "results": {
            "ws_a": [MapResult(samples=[("a0", "Da0", "t", {"k": 1}), ("a1", "Da1", "t", {"map_error": "boom"})], error=None)],
            "ws_b": [MapResult(samples=[], error="too large")],
        },
        "warning": "undeclared: ['z']",
    }
    blocks, events = _render(agent, out)
    text = _texts(blocks)
    assert "collection=ws_a: mapped 2 chunk(s) (1 with map_error)" in text
    assert "chunk_id=a0 | doc_id=Da0 | fields={'k': 1}" in text and "chunk_id=a1" not in text  # one sample
    assert "[error]\ncollection=ws_b: too large" in text and "[warning]\nundeclared: ['z']" in text
    assert events[0][1]["results"] == {"ws_a": {"n_mapped": 2, "n_failed": 1}, "ws_b": {"error": "too large"}}
    assert _texts(_render(agent, {"tool": "semantic_map", "tool_kwargs": {}, "error": "bad fields"})[0]) == "[error]\nbad fields"


def test_render_collection_tools(client):
    agent = _agent(client)
    tools = _tools(agent)
    out = tools["create_collection"]("ws_fed", "fed chunks", from_collections=[BASE])
    text = _texts(_render(agent, out)[0])
    assert text.startswith("[result]\nCreated collection 'ws_fed': now 8 chunk(s). This call: 8 new, 0 already present, 0 not found, 8 distinct doc(s)")
    assert "Metadata fields:\n  - doc_id (str): id" in text

    out = tools["add_to_collection"]("ws_fed", from_collections=[BASE])
    assert "Added to collection 'ws_fed': now 8 chunk(s). This call: 0 new, 8 already present" in _texts(_render(agent, out)[0])

    err = tools["add_to_collection"](BASE, from_collections=["ws_fed"])
    assert _texts(_render(agent, err)[0]).startswith("[error]\nadd_to_collection error:")

    listing = tools["list_collections"]()
    text = _texts(_render(agent, listing)[0])
    assert text.startswith("[result]\n2 collection(s):") and 'name="corpus"' in text and "Description: the base collection" in text
    assert 'name="ws_fed"' in text and "Description: fed chunks" in text and "create_collection(" in text

    out = tools["delete_collection"]("ws_fed")
    blocks, events = _render(agent, out)
    assert _texts(blocks) == "[result]\nDeleted collection 'ws_fed'." and events[0][0] == "delete_collection_tool_call"


def test_render_fallbacks(client):
    agent = _agent(client)
    assert _texts(_render(agent, None)[0]) == "[no output]"
    assert _texts(_render(agent, 42)[0]) == "[result]\n42"
    assert _texts(_render(agent, {"tool": "read_document", "tool_kwargs": {}, "docs": [{"doc_id": "Da0", "text": "hello"}]})[0]) == "hello"

