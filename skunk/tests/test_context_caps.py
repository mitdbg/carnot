"""Unit tests for `read_document` (full/partial reads + RetrievalState bookkeeping — the
per-call output cap was deliberately removed: the agent manages its own window via the
working set and prune) and the `input_values_desc` axis-label elision.

No LLM / PDFs / Chroma. Runs under pytest if installed, or standalone:
`python3 tests/test_context_caps.py`.
"""

from __future__ import annotations

from skunk.common import AnnotatedValue, _MAX_RENDERED_LABELS, input_values_desc
from skunk.search_agent.retrieval_state import RetrievalState
from skunk.search_agent.search_tools import READ_DOCUMENT_RESULT_TAG, ReadDocumentTool


# ---- read_document ---------------------------------------------------------------


def test_read_document_returns_full_docs():
    docs = {"d1": "hello", "d2": "world"}
    out = ReadDocumentTool(docs)(["d1", "d2"])
    assert out[READ_DOCUMENT_RESULT_TAG] is True
    assert len(out["docs"]) == 2
    # Header carries the doc id + size hints; the full body follows uncapped.
    assert out["docs"][0]["doc_id"] == "d1" and out["docs"][0]["text"].endswith("===\nhello")
    assert "doc_id=d1" in out["docs"][0]["text"] and "total chars: 5" in out["docs"][0]["text"]
    assert out["docs"][1]["doc_id"] == "d2" and out["docs"][1]["text"].endswith("===\nworld")


def test_read_document_missing_doc_note():
    state = RetrievalState()
    out = ReadDocumentTool({}, state)("nope")
    assert "no such document" in out["docs"][0]["text"]
    # A missing doc is NOT recorded as read/fetched.
    assert not state.read_doc_ids and not state.fetched_doc_ids


def test_read_document_char_ranges_and_scalar_broadcast():
    docs = {"d1": "abcdefghij", "d2": "hello"}
    tool = ReadDocumentTool(docs)
    # A None start/end broadcasts across a list of doc_ids (non-None scalars with a list are
    # rejected by the tool's contract — indices must then be same-length lists).
    # Per-doc ranges; None entries mean "no bound" for that doc (the docstring's own example).
    out = tool(["d1", "d2"], start_char_idx=[-3, None])
    assert out["docs"][0]["text"].endswith("===\nhij")
    assert out["docs"][1]["text"].endswith("===\nhello")
    # end_char_idx past the doc length truncates to the actual length.
    out = tool("d2", start_char_idx=1, end_char_idx=1000)
    assert out["docs"][0]["text"].endswith("===\nello")


def test_read_document_updates_state_and_unprunes():
    state = RetrievalState(pruned_doc_ids={"d1"})
    tool = ReadDocumentTool({"d1": "hello"}, state)
    tool("d1")
    # Reading marks the doc read + fetched, and re-reading a pruned doc un-prunes it so the
    # new blocks are visible again.
    assert state.read_doc_ids == {"d1"} and state.fetched_doc_ids == {"d1"}
    assert "d1" not in state.pruned_doc_ids


# ---- input_values_desc label elision ---------------------------------------------


def _vector(labels: list[str]) -> AnnotatedValue:
    return AnnotatedValue(
        description="series",
        kind="vector",
        index_name="period",
        value={lbl: float(i) for i, lbl in enumerate(labels)},
        unit="usd",
    )


def test_input_values_desc_elides_long_index():
    labels = [f"2{n:06d}" for n in range(5000)]  # 5000 distinct labels
    desc = input_values_desc([_vector(labels)])
    assert "labels elided" in desc
    # The rendered description must be far smaller than dumping all 5000 labels.
    assert len(desc) < 50_000
    # Edges are present; a deep-middle label is not.
    assert labels[0] in desc and labels[-1] in desc
    assert labels[2500] not in desc


def test_input_values_desc_small_index_is_full():
    labels = [f"19{n:02d}" for n in range(_MAX_RENDERED_LABELS)]  # exactly at the threshold
    desc = input_values_desc([_vector(labels)])
    assert "elided" not in desc
    assert all(lbl in desc for lbl in labels)


if __name__ == "__main__":
    test_read_document_returns_full_docs()
    test_read_document_missing_doc_note()
    test_read_document_char_ranges_and_scalar_broadcast()
    test_read_document_updates_state_and_unprunes()
    test_input_values_desc_elides_long_index()
    test_input_values_desc_small_index_is_full()
    print("ok")
