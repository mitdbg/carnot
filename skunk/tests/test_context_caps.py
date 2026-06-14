"""Unit tests for the context-size caps that keep a single LLM request under the model's
~1.05M input-token ceiling: the `read_document` per-call output cap and the
`input_values_desc` axis-label elision.

No LLM / PDFs / Chroma. Runs under pytest if installed, or standalone:
`python3 tests/test_context_caps.py`.
"""

from __future__ import annotations

from skunk.common import AnnotatedValue, _MAX_RENDERED_LABELS, input_values_desc
from skunk.search_agent.search_tools import READ_DOCUMENT_RESULT_TAG, ReadDocumentTool


# ---- read_document output cap ----------------------------------------------------


def _doc_text_len(out: dict) -> int:
    return sum(len(d["text"]) for d in out["docs"])


def test_read_document_truncates_and_drops_overflow():
    docs = {"d1": "a" * 1000, "d2": "b" * 1000, "d3": "c" * 1000}
    tool = ReadDocumentTool(docs, max_pages=20, max_output_chars=500)
    out = tool(["d1", "d2", "d3"])
    assert out[READ_DOCUMENT_RESULT_TAG] is True
    # Only the first doc is (partly) returned; d2/d3 are dropped.
    assert len(out["docs"]) == 1
    text = out["docs"][0]["text"]
    assert "[truncated:" in text and "2 more requested doc(s) not shown" in text
    # The doc body honored the cap (the appended note is the only allowed overflow).
    assert len(text.replace("[truncated:", "")) <= 500 + 400  # body ≤ cap + bounded note


def test_read_document_small_read_is_unchanged():
    docs = {"d1": "hello", "d2": "world"}
    tool = ReadDocumentTool(docs, max_pages=20, max_output_chars=400_000)
    out = tool(["d1", "d2"])
    assert len(out["docs"]) == 2
    assert "[truncated:" not in _join_all(out)
    assert out["docs"][0]["text"] == "=== doc_id=d1 ===\nhello"
    assert out["docs"][1]["text"] == "=== doc_id=d2 ===\nworld"


def test_read_document_missing_doc_note():
    tool = ReadDocumentTool({}, max_pages=20, max_output_chars=400_000)
    out = tool("nope")
    assert "no such document" in out["docs"][0]["text"]


def _join_all(out: dict) -> str:
    return "\n".join(d["text"] for d in out["docs"])


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
    test_read_document_truncates_and_drops_overflow()
    test_read_document_small_read_is_unchanged()
    test_read_document_missing_doc_note()
    test_input_values_desc_elides_long_index()
    test_input_values_desc_small_index_is_full()
    print("ok")
