"""Unit tests for `inherited_column_headers` / `continuation_chain` — the column grammar a
header-less continuation page inherits from its run HEAD.

Regression target: the head's *last table* (the one that spilled onto the run) is the source
of the inherited columns. The earlier rule grabbed the head's last table WITH headers, which
silently borrowed an unrelated earlier table's grammar when the scan didn't capture the
spilling table's own headers (observed on cs-1887 page 60).

No LLM / PDFs / store — drives the pure functions with hand-built `PageScan`s. Runs under
pytest, or standalone: `python3 tests/test_continuation_headers.py`.
"""

from __future__ import annotations

from skunk.common import PageRef
from officeqa.page_index.data_model import (
    ContentBlock,
    continuation_chain,
    inherited_column_headers,
)
from officeqa.page_index.scan import PageScan

STEM = "doc"


def _table(title: str | None, cols: list[str]) -> ContentBlock:
    return ContentBlock(kind="table", title=title, column_headers=cols)


def _scan(is_cont: bool, blocks: list[ContentBlock]) -> PageScan:
    return PageScan(page_role="content", is_continuation=is_cont, blocks=blocks)


def _summary_fn(pages: dict[int, PageScan]):
    return lambda ref: pages.get(ref.page)


def test_inherits_last_table_of_head():
    # Head page 10's bottommost table ("Recapitulation", ['Amount']) is the one that spills.
    pages = {
        10: _scan(False, [_table("Disbursements", ["A", "B"]), _table("Recapitulation", ["Amount"])]),
        11: _scan(True, [_table("Recapitulation cont.", [])]),
    }
    got = inherited_column_headers(PageRef(stem=STEM, page=11), _summary_fn(pages))
    assert got == (10, "Recapitulation", ["Amount"])


def test_spilling_table_without_headers_returns_none_not_wrong_table():
    # The regression: head 10 has a labeled "Disbursements" table THEN the spilling
    # "Recapitulation" table whose own headers the scan missed (empty). We must inherit
    # nothing — never borrow "Disbursements"' columns.
    pages = {
        10: _scan(False, [_table("Disbursements", ["A", "B", "C"]), _table("Recapitulation", [])]),
        11: _scan(True, [_table("Recapitulation cont.", [])]),
    }
    assert inherited_column_headers(PageRef(stem=STEM, page=11), _summary_fn(pages)) is None


def test_page_restating_own_headers_is_skipped():
    # The flagged page re-states its own columns -> prefer those, inherit nothing.
    pages = {
        10: _scan(False, [_table("Recapitulation", ["Amount"])]),
        11: _scan(True, [_table("Recapitulation cont.", ["Amount"])]),
    }
    assert inherited_column_headers(PageRef(stem=STEM, page=11), _summary_fn(pages)) is None


def test_non_continuation_page_returns_none():
    pages = {10: _scan(False, [_table("Recapitulation", ["Amount"])])}
    assert inherited_column_headers(PageRef(stem=STEM, page=10), _summary_fn(pages)) is None


def test_head_with_no_table_returns_none():
    pages = {
        10: _scan(False, [ContentBlock(kind="prose", title="Intro")]),
        11: _scan(True, [_table("cont.", [])]),
    }
    assert inherited_column_headers(PageRef(stem=STEM, page=11), _summary_fn(pages)) is None


def test_deep_run_walks_back_to_head():
    # A 3-page run: 30 (head) -> 31 (cont) -> 32 (cont). Page 32 inherits 30's last table.
    pages = {
        30: _scan(False, [_table("Recapitulation", ["Amount"])]),
        31: _scan(True, [_table("cont.", [])]),
        32: _scan(True, [_table("cont.", [])]),
    }
    chain = continuation_chain(PageRef(stem=STEM, page=32), _summary_fn(pages))
    assert [r.page for r in chain] == [30, 31]  # ascending, excludes ref itself
    got = inherited_column_headers(PageRef(stem=STEM, page=32), _summary_fn(pages))
    assert got == (30, "Recapitulation", ["Amount"])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok: {name}")
    print("all passed")
