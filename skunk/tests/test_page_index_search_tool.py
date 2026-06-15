"""Unit tests for the optional `page_index_search` tool that lets the SearchAgent query the
PageIndex from inside its loop (env `SKUNK_SEARCH_AGENT_PAGEINDEX=1`).

No LLM and no real artifacts: a stub retriever stands in for `PageIndexRetriever`, so these
cover the wiring — page-key round-trip, the synthetic-branch / document-scope plumbing, the
output cap, and the `SearchAgent._blocks_from_output` rendering branch.

Runs under pytest if installed, or standalone: `python3 tests/test_page_index_search_tool.py`.
"""

from __future__ import annotations

from skunk.common import (
    BlockRef,
    PageRef,
    SemPoolEntry,
    page_key_to_pageref,
    pageref_to_page_key,
)
from skunk.local_python_executor import CodeOutput
from skunk.multi_turn_agent import ChunkBlock, TextBlock
from skunk.search_agent.search_agent import SearchAgent
from skunk.search_agent.search_tools import PAGE_INDEX_RESULT_TAG, PageIndexSearchTool


class _StubRetriever:
    """Stands in for `PageIndexRetriever`: records the call and returns canned blocks."""

    def __init__(self, blocks):
        self._blocks = blocks
        self.calls: list[tuple] = []

    async def retrieve_all(self, ctx, branches, document_scopes=None):
        self.calls.append((branches[0].key, branches[0].period, document_scopes))
        return [self._blocks]

    def pool_for_blocks(self, blocks, pdf_dir):
        return [
            SemPoolEntry(
                ref=b,
                interval=("2013-10", "2014-09"),
                kind="table",
                title="Public Debt",
                summary="monthly debt subject to limit",
                cols=("Month", "Amount"),
                rows_tail=("Sep",),
                rows=("Jan", "Sep"),
            )
            for b in blocks
        ]


def _block(month="2014-06", page=12) -> BlockRef:
    ref = PageRef(month=month, page=page)
    return BlockRef(page=ref, block_index=0, member_refs=(ref,))


def test_page_key_round_trip():
    for key in ("2002_12_25", "1946_11_41", "2014_06_12"):
        assert pageref_to_page_key(page_key_to_pageref(key)) == key


def test_tool_returns_tagged_results_with_page_keys():
    stub = _StubRetriever([_block()])
    tool = PageIndexSearchTool(stub, "/tmp/pdfs", lambda: object(), max_output_tokens=100_000)
    out = tool("monthly public debt", period="2013-10..2014-09")
    assert out[PAGE_INDEX_RESULT_TAG] is True
    assert out["results"][0]["page_key"] == "2014_06_12"
    assert "monthly debt subject to limit" in out["results"][0]["text"]
    # Synthetic branch carries the query + period; no scope when no required_bulletins.
    assert stub.calls == [("monthly public debt", "2013-10..2014-09", None)]


def test_required_bulletins_become_document_scopes():
    stub = _StubRetriever([_block()])
    tool = PageIndexSearchTool(
        stub, "/tmp/pdfs", lambda: object(), max_output_tokens=100_000,
        required_bulletins=["2014-06"],
    )
    tool("q", period=None)
    assert stub.calls == [("q", None, [["2014-06"]])]


def test_retriever_error_is_carried_in_payload():
    class _Boom(_StubRetriever):
        async def retrieve_all(self, ctx, branches, document_scopes=None):
            raise RuntimeError("kaboom")

    tool = PageIndexSearchTool(_Boom([]), "/tmp", lambda: object(), max_output_tokens=1000)
    out = tool("q")
    assert out[PAGE_INDEX_RESULT_TAG] is True
    assert out["results"] == []
    assert "kaboom" in out["error"]


def test_output_cap_truncates_and_notes():
    stub = _StubRetriever([_block(page=p) for p in range(1, 51)])
    # Tiny cap so all-but-the-first block is dropped.
    tool = PageIndexSearchTool(stub, "/tmp", lambda: object(), max_output_tokens=20)
    out = tool("q")
    assert 0 < len(out["results"]) < 50
    assert "truncation_note" in out


def test_blocks_from_output_renders_page_index_payload():
    payload = {
        PAGE_INDEX_RESULT_TAG: True,
        "results": [{"page_key": "2014_06_12", "text": "some block summary"}],
    }
    blocks = SearchAgent._blocks_from_output(object(), CodeOutput(output=payload, logs=""))
    chunk = [b for b in blocks if isinstance(b, ChunkBlock)]
    assert len(chunk) == 1 and chunk[0].doc_id == "2014_06_12" and chunk[0].chunk_id is None


def test_blocks_from_output_empty_and_error():
    empty = SearchAgent._blocks_from_output(
        object(), CodeOutput(output={PAGE_INDEX_RESULT_TAG: True, "results": []}, logs="")
    )
    assert all(isinstance(b, TextBlock) for b in empty) and empty
    err = SearchAgent._blocks_from_output(
        object(),
        CodeOutput(output={PAGE_INDEX_RESULT_TAG: True, "results": [], "error": "boom"}, logs=""),
    )
    assert any("boom" in b.text for b in err if isinstance(b, TextBlock))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok: {name}")
    print("all passed")
