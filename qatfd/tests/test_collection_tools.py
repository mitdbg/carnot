"""Unit tests for the collection-management tools in qatfd.tools (ResultSet + create/add/copy/merge/
delete/list) over a sharded base corpus served through a MergedClient.

Run from the qatfd dir: python3 -m pytest tests/test_collection_tools.py
"""

from __future__ import annotations

import chromadb
import pytest

from types import SimpleNamespace

from qatfd.constants import METADATA_LIST_DELIMITER
from qatfd.merged_chroma import MergedClient, Shard
from qatfd.tools import (
    COLLECTION_FLAG_KEY,
    AddToCollectionTool,
    CopyCollectionTool,
    CreateCollectionTool,
    DeleteCollectionTool,
    GrepCorpusTool,
    ListCollectionsTool,
    MapTool,
    MergeCollectionsTool,
    ResultSet,
    SearchResult,
    append_action,
    format_metadata_fields,
    SemanticMapTool,
    merge_metadata_fields,
    normalize_metadata_fields,
    validate_metadata_fields,
)
import qatfd.tools as tools_mod

BASE = "corpus"
BASE_FIELDS = "  - doc_id (str): the document id\n  - year (int): publication year\n"


def _fill(collection, prefix: str, n: int, x0: float, year: int) -> None:
    collection.add(
        ids=[f"{prefix}{i}" for i in range(n)],
        embeddings=[[x0 + i, 0.0] for i in range(n)],
        documents=[f"{prefix} text {i} " + ("federal reserve" if i % 2 == 0 else "treasury") for i in range(n)],
        metadatas=[{"doc_id": f"D{prefix}{i // 2}", "type": "text" if i % 3 else "table", "year": year} for i in range(n)],
    )


@pytest.fixture
def client(tmp_path):
    a = chromadb.PersistentClient(path=str(tmp_path / "a"))
    b = chromadb.PersistentClient(path=str(tmp_path / "b"))
    _fill(a.create_collection(f"{BASE}_r0", metadata={"fields": BASE_FIELDS}), "a", 6, 0.0, 1946)
    _fill(b.create_collection(f"{BASE}_r1", metadata={"fields": BASE_FIELDS}), "b", 6, 10.0, 1947)
    return MergedClient(BASE, [Shard(a, f"{BASE}_r0", "A"), Shard(b, f"{BASE}_r1", "B")])


@pytest.fixture
def tools(client):
    kw = dict(agent_id="bootstrap_0", max_copy_chunks=1000, batch_size=4)
    return {
        "create": CreateCollectionTool(client, BASE, **kw),
        "add": AddToCollectionTool(client, BASE, **kw),
        "copy": CopyCollectionTool(client, BASE, **kw),
        "merge": MergeCollectionsTool(client, BASE, **kw),
        "delete": DeleteCollectionTool(client, BASE, **kw),
        "list": ListCollectionsTool(client, BASE, **kw),
        "grep": GrepCorpusTool(client),
    }


def _rs(tool: str, collection: str, ids: list[str], year: int = 1946) -> ResultSet:
    hits = [SearchResult(i, f"D{i}", None, f"text {i}", {"doc_id": f"D{i}", "year": year}, None) for i in ids]
    return ResultSet({"tool": tool, "tool_kwargs": {}, "results": {collection: hits}})


# ---- schema helpers ----------------------------------------------------------------------------


def test_format_and_merge_metadata_fields():
    fields = [{"name": "topic", "type": "str", "desc": "main topic"}, {"name": "n", "type": "int", "desc": "count"}]
    assert format_metadata_fields(fields) == "  - topic (str): main topic\n  - n (int): count\n"
    merged = merge_metadata_fields(BASE_FIELDS, format_metadata_fields(fields), "  - year (int): dup\n", None, "")
    assert merged == BASE_FIELDS + "  - topic (str): main topic\n  - n (int): count\n"
    assert validate_metadata_fields(fields) is None
    assert validate_metadata_fields([]) and validate_metadata_fields([{"field": "x", "type": "str", "desc": "d"}])
    assert validate_metadata_fields([{"name": "x", "type": "str", "desc": ""}])


def test_normalize_metadata_fields_accepts_python_types():
    """The tool docs call `type` "the Python type of the field", so agents pass `str` / `float | None` /
    `list[str]` as often as "str"; normalization renders them as text and leaves the input untouched."""
    from typing import Optional

    fields = [
        {"name": "a", "type": str, "desc": "d"},
        {"name": "b", "type": float | None, "desc": "d"},
        {"name": "c", "type": list[str], "desc": "d"},
        {"name": "d", "type": Optional[int], "desc": "d"},
        {"name": "e", "type": "bool", "desc": "d"},
    ]
    norm = normalize_metadata_fields(fields)
    assert [f["type"] for f in norm] == ["str", "float | None", "list[str]", "int | None", "bool"]  # py3.14 renders Optional[int] as int | None
    assert validate_metadata_fields(norm) is None
    assert fields[0]["type"] is str  # copies, no mutation of the agent's objects
    assert format_metadata_fields(norm).startswith("  - a (str): d\n  - b (float | None): d\n")
    # still rejected after normalization: a missing / None type, and non-list input
    assert validate_metadata_fields(normalize_metadata_fields([{"name": "x", "type": None, "desc": "d"}]))
    assert validate_metadata_fields(normalize_metadata_fields([{"name": "x", "desc": "d"}]))
    assert normalize_metadata_fields("nope") == "nope" and validate_metadata_fields(normalize_metadata_fields(None))


def test_append_action_uses_list_delimiter(client):
    c = client.create_collection("ws_log", metadata={"actions": ""})
    append_action(c, "a()")
    append_action(client.get_collection("ws_log"), "b()")
    assert client.get_collection("ws_log").metadata["actions"] == f"a(){METADATA_LIST_DELIMITER}b()"


def test_map_tool_records_fields(tools, client):
    tools["create"]("ws_map", "to map", from_collections=[BASE], metadata_filter={"year": 1946})
    tool = MapTool(client, {}, SimpleNamespace(map_max_workers=2))
    fields = [{"name": "mentions_fed", "type": "bool", "desc": "text mentions the Fed"}]
    out = tool(["ws_map"], lambda text: {"mentions_fed": "federal" in text, "extra": 1}, fields)
    assert "error" not in out and "extra" in out["warning"]
    c = client.get_collection("ws_map")
    assert c.metadata["fields"] == BASE_FIELDS + "  - mentions_fed (bool): text mentions the Fed\n"
    assert c.get(ids=["a0"])["metadatas"][0]["mentions_fed"] is True
    assert "must match" not in str(tool(["ws_map"], lambda t: {}, [{"name": "x"}])["error"]) and "error" in tool(["ws_map"], lambda t: {}, [])
    # a Python type for `type` is accepted and recorded as text (in the schema and in the tool_kwargs the trace / actions ledger keep)
    out = tool(["ws_map"], lambda text: {"n_words": len(text.split())}, [{"name": "n_words", "type": int, "desc": "word count"}])
    assert "error" not in out and out["tool_kwargs"]["fields"] == [{"name": "n_words", "type": "int", "desc": "word count"}]
    assert client.get_collection("ws_map").metadata["fields"].endswith("  - n_words (int): word count\n")


def test_map_tool_passes_metadata(tools, client):
    """map_fn(text, metadata): the metadata carries chunk_id / doc_id / earlier fields, so a field can be
    derived from the ids; doc_level maps once per document on the shared metadata; ids cannot be overwritten."""
    tools["create"]("ws_meta", "to map", from_collections=[BASE])
    tool = MapTool(client, {"Da0": "doc text a0", "Db1": "doc text b1"}, SimpleNamespace(map_max_workers=2))
    out = tool(["ws_meta"], lambda text, meta: {"doc_num": int(meta["doc_id"][2:]), "cid": meta["chunk_id"], "doc_id": "HACK"},
               [{"name": "doc_num", "type": int, "desc": "n"}, {"name": "cid", "type": str, "desc": "c"}])
    assert "error" not in out
    row = client.get_collection("ws_meta").get(ids=["a3"])["metadatas"][0]
    assert row["doc_num"] == 1 and row["cid"] == "a3" and row["doc_id"] == "Da1"  # reserved key ignored, not overwritten
    # doc_level: one call per document with the metadata its chunks share (chunk_id drops out), applied to every chunk
    seen: list[dict] = []
    def per_doc(text, meta):
        seen.append(meta)
        return {"doc_words": len(text.split())}
    out = tool(["ws_meta"], per_doc, [{"name": "doc_words", "type": int, "desc": "w"}], doc_level=True)
    assert "error" not in out and all("chunk_id" not in m and m["doc_id"] for m in seen)
    assert len(seen) == len({m["doc_id"] for m in seen})  # once per document
    got = client.get_collection("ws_meta").get(ids=["a0", "a1"])["metadatas"]
    assert got[0]["doc_words"] == got[1]["doc_words"] == 3  # both chunks of Da0 got the document's value
    # earlier fields are visible to later maps
    out = tool(["ws_meta"], lambda t, m: {"twice": m["doc_num"] * 2}, [{"name": "twice", "type": int, "desc": "2n"}])
    assert client.get_collection("ws_meta").get(ids=["b5"])["metadatas"][0]["twice"] == 4


class _JudgeStub:
    """An LLMClient stand-in for the semantic map: replies with a bare JSON object built from the metadata block
    of the prompt (proving the metadata reached the judge), in whichever wrapping `style` says."""

    def __init__(self, style: str = "bare"):
        self.style = style
        self.config = SimpleNamespace(llm_context_limits={})
        self.prompts: list[str] = []

    def call(self, messages, **kwargs):
        import json as _json
        user = messages[1]["content"]
        self.prompts.append(user)
        meta = _json.loads(user.split("Metadata:\n", 1)[1].split("\n\nFields:", 1)[0])
        body = _json.dumps({"seen_doc": meta["doc_id"], "seen_chunk": meta.get("chunk_id"), "chunk_id": "HACK"})
        text = {"bare": body, "fenced": f"```json\n{body}\n```", "prose": f"Sure, here it is:\n{body}\nDone."}[self.style]
        return SimpleNamespace(text=text)


def _semmap_config():
    return SimpleNamespace(
        semantic_map_max_workers=2, semantic_map_max_candidate_chunks=100, semantic_map_context_frac=0.9,
        semantic_map_max_output_tokens=64, semantic_map_disable_reasoning=True, semantic_map_provider_order=None,
        request_timeout_s=None,
    )


def test_semantic_map_parses_bare_json_and_shows_metadata(tools, client):
    tools["create"]("ws_sem", "to judge", from_collections=[BASE], metadata_filter={"year": 1947})
    for style in ("bare", "fenced", "prose"):
        stub = _JudgeStub(style)
        tool = SemanticMapTool(client, stub, {}, _semmap_config(), "stub-model")  # type: ignore[arg-type]
        out = tool(["ws_sem"], [{"name": "seen_doc", "type": str, "desc": "d"}, {"name": "seen_chunk", "type": str, "desc": "c"}])
        assert "error" not in out, style
        res = out["results"]["ws_sem"]
        assert all(p.error is None for p in res) and not any(s.error or "map_error" in s.fields for p in res for s in p.samples), style
        row = client.get_collection("ws_sem").get(ids=["b2"])["metadatas"][0]
        assert row["seen_doc"] == "Db1" and row["seen_chunk"] == "b2" and "chunk_id" not in row, style  # injected for the judge, reserved on write
    # doc_level: the judge sees the document's shared metadata (no chunk_id) once per document
    stub = _JudgeStub()
    tool = SemanticMapTool(client, stub, {"Db1": "full doc", "Db0": "full doc", "Db2": "full doc"}, _semmap_config(), "stub-model")  # type: ignore[arg-type]
    out = tool(["ws_sem"], [{"name": "seen_doc", "type": str, "desc": "d"}, {"name": "seen_chunk", "type": str, "desc": "c"}], doc_level=True)
    assert "error" not in out and len(stub.prompts) == 3 and all('"chunk_id"' not in p.split("Metadata:")[1].split("Fields:")[0] for p in stub.prompts)
    assert client.get_collection("ws_sem").get(ids=["b2"])["metadatas"][0].get("seen_chunk") is None  # chroma drops None-valued keys


def test_semantic_map_parse_json():
    parse = SemanticMapTool._parse_json
    assert parse('{"a": 1, "b": null}') == {"a": 1, "b": None}
    assert parse('```json\n{"a": "x"}\n```') == {"a": "x"}
    assert parse('Here you go: {"a": [1, 2]} thanks') == {"a": "[1, 2]"}  # non-scalars are json-encoded
    assert "map_error" in parse("") and "map_error" in parse("no json here") and "map_error" in parse("[1, 2]")
    assert "map_error" in parse('{"chunk_id": "x"}')  # only reserved keys => empty output


def test_grep_pages_large_match_sets(tools, client, monkeypatch):
    """A grep with no limit (or one over the SQL-variable cap) scans ids then fetches rows in batches; the
    result matches the single-get path exactly and embeddings are only fetched when the tool keeps them."""
    single = tools["grep"]([BASE], "(?i)text", limit=100)
    assert len(single) == 12 and single.results[BASE][0].embedding is None
    monkeypatch.setattr(tools_mod, "_GREP_MAX_LIMIT", 2)
    monkeypatch.setattr(tools_mod, "_GREP_ROW_BATCH_TEXT_ONLY", 5)
    paged = tools["grep"]([BASE], "(?i)text")
    assert paged.chunk_ids == single.chunk_ids and "error" not in paged
    assert [h.text for h in paged.results[BASE]] == [h.text for h in single.results[BASE]]
    over_cap = tools["grep"]([BASE], "(?i)text", limit=7)  # limit above the (patched) cap => paged, limit honoured
    assert len(over_cap) == 7 and over_cap.chunk_ids == single.chunk_ids[:7]
    with_emb = GrepCorpusTool(client, include_embeddings=True)([BASE], "(?i)text")
    assert len(with_emb) == 12 and with_emb.results[BASE][0].embedding is not None


def test_resultset_algebra_keeps_errors_and_provenance(tools):
    ok = tools["grep"]([BASE], "(?i)federal reserve", limit=3)
    bad = ResultSet({"tool": "grep_corpus", "tool_kwargs": {"pattern": "boom"}, "results": {}, "error": "grep_corpus error: too many SQL variables"})
    diff = bad - ok
    assert len(diff) == 0 and "too many SQL variables" in diff["error"]
    assert diff["tool_kwargs"]["op"] == "-" and diff["tool_kwargs"]["left"]["tool_kwargs"]["pattern"] == "boom"
    assert diff["tool_kwargs"]["right"]["tool_kwargs"]["pattern"] == "(?i)federal reserve" and diff["tool_kwargs"]["right"]["num_chunks"] == 3
    assert "error" in (ok | bad) and "error" in bad.where(lambda m: True) and "error" not in (ok - ok)
    # create refuses a composed set that carries a failure, and records the searches behind a good one
    out = tools["create"]("ws_bad", "x", diff)
    assert "error" in out and "too many SQL variables" in out["error"]
    out = tools["create"]("ws_prov", "x", ok - bad.where(lambda m: False) if False else ok | ok)
    assert "error" not in out
    actions = tools["list"]()["collections"]
    action = next(c for c in actions if c["name"] == "ws_prov")["actions"][0]
    assert "'pattern': '(?i)federal reserve'" in action and "'limit': 3" in action


# ---- ResultSet ----------------------------------------------------------------------------------


def test_resultset_algebra_and_where():
    a = _rs("search_corpus", BASE, ["a0", "a1", "a2"])
    b = _rs("grep_corpus", "ws_x", ["a2", "b0"], year=1947)
    with pytest.raises(TypeError):
        a | ["a0"]  # type: ignore[operator]
    assert len(a) == 3 and a.chunk_ids == ["a0", "a1", "a2"] and a.doc_ids == ["Da0", "Da1", "Da2"]
    union, inter, diff = a | b, a & b, a - b
    assert union.chunk_ids == ["a0", "a1", "a2", "b0"] and union["tool"] == "result_set"
    assert union.by_source() == {BASE: ["a0", "a1", "a2"], "ws_x": ["b0"]}  # a2 attributed to its first source
    assert inter.chunk_ids == ["a2"] and diff.chunk_ids == ["a0", "a1"]
    assert union.where(lambda m: m["year"] == 1947).chunk_ids == ["b0"]
    assert "chunks=4" in repr(union) and union["tool_kwargs"]["op"] == "|"
    # still a dict payload for the rendering / tracing code
    assert isinstance(union, dict) and union["results"][BASE][0].chunk_id == "a0"


def test_grep_returns_resultset(tools):
    rs = tools["grep"]([BASE], "(?i)federal reserve", metadata_filter={"year": 1946})
    assert isinstance(rs, ResultSet) and rs["tool"] == "grep_corpus"
    assert sorted(rs.chunk_ids) == ["a0", "a2", "a4"]


# ---- create / add ---------------------------------------------------------------------------------


def test_create_from_results_and_add(tools, client):
    hits = _rs("search_corpus", BASE, ["a0", "a1", "b0", "zz"])  # zz: not in the corpus
    out = tools["create"]("ws_fed", "Fed policy chunks", hits)
    assert "error" not in out, out
    assert out["num_chunks"] == 3 and out["stats"]["n_new"] == 3 and out["stats"]["n_missing"] == 1
    assert out["stats"]["n_docs"] == 2 and out["stats"]["sources"] == {BASE: 3}
    c = client.get_collection("ws_fed")
    assert c.metadata[COLLECTION_FLAG_KEY] is True and c.metadata["description"] == "Fed policy chunks"
    assert c.metadata["created_by"] == "bootstrap_0" and c.metadata["fields"] == BASE_FIELDS
    assert c.metadata["actions"].startswith("create_collection(") and METADATA_LIST_DELIMITER not in c.metadata["actions"]
    got = c.get(ids=["b0"], include=["embeddings", "documents", "metadatas"])
    assert list(got["embeddings"][0]) == [10.0, 0.0] and got["metadatas"][0]["year"] == 1947

    # re-adding overlapping results is idempotent; new ones are added
    out = tools["add"]("ws_fed", _rs("grep_corpus", BASE, ["a1", "a2"]))
    assert out["stats"]["n_new"] == 1 and out["stats"]["n_existing"] == 1 and out["num_chunks"] == 4
    actions = client.get_collection("ws_fed").metadata["actions"].split(METADATA_LIST_DELIMITER)
    assert len(actions) == 2 and actions[1].startswith("add_to_collection(")

    # a list of result sets, and results sourced from a managed collection
    out = tools["add"]("ws_fed", [_rs("grep_corpus", "ws_fed", ["a0"]), _rs("grep_corpus", BASE, ["b1"])])
    assert out["num_chunks"] == 5 and out["stats"]["sources"] == {"ws_fed": 1, BASE: 1}


def test_create_from_filtered_collection(tools):
    out = tools["create"]("ws_tables", "table chunks", from_collections=[BASE], metadata_filter={"type": "table"})
    assert "error" not in out, out
    assert out["num_chunks"] == 4  # a0, a3, b0, b3
    # combine results + a filtered copy in one call
    out = tools["create"]("ws_mix", "mix", _rs("search_corpus", BASE, ["a1"]), from_collections=["ws_tables"], metadata_filter={"year": 1947})
    assert out["num_chunks"] == 3 and out["stats"]["sources"] == {BASE: 1, "ws_tables": 2}


def test_create_errors(tools, client):
    assert "base collection" in tools["create"](BASE, "x")["error"]
    assert "invalid collection name" in tools["create"]("ab", "x")["error"]
    assert "description" in tools["create"]("ws_ok", "  ")["error"]
    assert "nothing to add" not in str(tools["create"]("ws_ok", "fine"))  # empty collection is allowed
    assert "already exists" in tools["create"]("ws_ok", "again")["error"]
    assert "metadata_filter" in tools["create"]("ws_bad", "x", metadata_filter={"a": 1})["error"]
    assert "must be the value returned" in tools["create"]("ws_bad", "x", ["a0"])["error"]
    err = ResultSet({"tool": "grep_corpus", "tool_kwargs": {}, "results": {}, "error": "boom"})
    assert "carries an error" in tools["create"]("ws_bad", "x", err)["error"]
    # a failed populate does not leave the collection behind
    assert "ws_bad" not in {c.name for c in client.list_collections()}


def test_copy_cap_rolls_back_the_new_collection(tools, client):
    out = CreateCollectionTool(client, BASE, max_copy_chunks=5)("ws_big", "too big", from_collections=[BASE])
    assert "more than 5 chunks" in out["error"]
    out = CopyCollectionTool(client, BASE, max_copy_chunks=5)(BASE, "ws_big2")
    assert "more than 5 chunks" in out["error"]
    out = MergeCollectionsTool(client, BASE, max_copy_chunks=5)([BASE], "ws_big3", "x")
    assert "more than 5 chunks" in out["error"]
    assert not {"ws_big", "ws_big2", "ws_big3"} & {c.name for c in client.list_collections()}


def test_add_errors(tools):
    assert "cannot be modified" in tools["add"](BASE, _rs("search_corpus", BASE, ["a0"]))["error"]
    assert "does not exist" in tools["add"]("nope", _rs("search_corpus", BASE, ["a0"]))["error"]
    assert "not an agent-managed" in tools["add"](f"{BASE}_r0", _rs("search_corpus", BASE, ["a0"]))["error"]
    tools["create"]("ws_a", "a")
    assert "nothing to add" in tools["add"]("ws_a")["error"]
    assert "to itself" in tools["add"]("ws_a", from_collections=["ws_a"])["error"]


# ---- copy / merge / delete / list -------------------------------------------------------------------


def test_copy_merge_delete_list(tools, client):
    tools["create"]("ws_1946", "1946 chunks", from_collections=[BASE], metadata_filter={"year": 1946})
    tools["create"]("ws_tables", "tables", from_collections=[BASE], metadata_filter={"type": "table"})
    client.get_collection("ws_tables").modify(metadata={**client.get_collection("ws_tables").metadata, "fields": BASE_FIELDS + "  - caption (str): table caption\n"})

    out = tools["copy"]("ws_1946", "ws_1946_tables", metadata_filter={"type": "table"})
    assert out["num_chunks"] == 2 and "Copy of 'ws_1946'" in client.get_collection("ws_1946_tables").metadata["description"]

    out = tools["merge"](["ws_1946", "ws_tables"], "ws_union", "1946 or tables")
    assert "error" not in out, out
    assert out["num_chunks"] == 8  # 6 (1946) + 4 (tables) - 2 (1946 tables)
    assert out["stats"]["n_existing"] == 2 and out["fields"].endswith("  - caption (str): table caption\n")

    listing = tools["list"]()["collections"]
    assert [d["name"] for d in listing] == [BASE, "ws_1946", "ws_1946_tables", "ws_tables", "ws_union"]
    assert listing[0]["is_base"] and listing[0]["num_chunks"] == 12
    assert listing[1]["description"] == "1946 chunks" and listing[1]["actions"][0].startswith("create_collection(")

    assert tools["delete"]("ws_1946_tables")["deleted"] is True
    assert "base collection" in tools["delete"](BASE)["error"]
    assert "not an agent-managed" in tools["delete"](f"{BASE}_r1")["error"]
    assert "does not exist" in tools["delete"]("ws_1946_tables")["error"]
    assert "ws_1946_tables" not in [d["name"] for d in tools["list"]()["collections"]]

    assert "already exists" in tools["merge"](["ws_1946"], "ws_union", "dup")["error"]
    assert "does not exist" in tools["merge"](["ws_1946", "ghost"], "ws_union2", "x")["error"]
