"""Unit tests for `PageStore.text()` serving the colleague-cleaned per-page corpus
(`cleaned/<stem>_<page>.txt`) instead of the raw parsed-JSON `pages/`, with a figure note
re-derived from the page's `PageScan` chart blocks.

No LLM / PDFs / Chroma — builds a tiny temp artifact root with `cleaned/` + `scans/`. Runs
under pytest if installed, or standalone: `python3 tests/test_page_store_cleaned.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

from skunk.common import PageRef
from skunk.errors import StepFailed
from officeqa.page_index.data_model import (
    CLEANED_SUBDIR,
    SCANS_SUBDIR,
    ContentBlock,
    figure_note,
)
from officeqa.page_index.scan import PageScan
from officeqa.page_index.store import PageStore

STEM = "combined_statement__historical__cs-1872"


def _build_root(tmp: Path) -> Path:
    """A minimal artifact root: one table page, one mixed prose+chart page, one figure-only
    page (empty cleaned body but a chart block in its scan), and a scan-less page."""
    cleaned = tmp / CLEANED_SUBDIR
    scans = tmp / SCANS_SUBDIR
    cleaned.mkdir()
    scans.mkdir()

    (cleaned / f"{STEM}_13.txt").write_text("DISBURSEMENTS.\n| a | b |\n| :--- | ---: |\n")
    (cleaned / f"{STEM}_20.txt").write_text("Receipts rose sharply this year.\n")
    (cleaned / f"{STEM}_1.txt").write_text("")  # figure-only page: empty cleaned body
    (cleaned / f"{STEM}_30.txt").write_text("Plain page, no scan entry.\n")

    def scan(role: str, blocks: list[ContentBlock]) -> dict:
        return PageScan(page_role=role, blocks=blocks).model_dump()

    payload = {
        "scans": {
            "13": scan("content", [ContentBlock(kind="table", title="Disbursements")]),
            "20": scan("content", [ContentBlock(kind="chart", title="Receipts by Source")]),
            "1": scan("content", [ContentBlock(kind="chart", title="Cover Figure")]),
        }
    }
    (scans / f"{STEM}.json").write_text(json.dumps(payload))
    return tmp


def _store(tmp: Path) -> PageStore:
    return PageStore(_build_root(tmp), pdf_dir="/nonexistent")


def test_table_page_serves_cleaned_markdown_no_note(tmp_path):
    store = _store(tmp_path)
    text = store.text(PageRef(stem=STEM, page=13))
    assert text == "DISBURSEMENTS.\n| a | b |\n| :--- | ---: |"  # outer whitespace stripped
    assert "| :---" in text
    assert "return [] so the" not in text  # no chart block -> no figure note


def test_chart_page_appends_scan_derived_note(tmp_path):
    store = _store(tmp_path)
    text = store.text(PageRef(stem=STEM, page=20))
    note = figure_note(1, ["Receipts by Source"])
    assert text == "Receipts rose sharply this year.\n\n" + note
    assert text.endswith("the vision tier can read it.]")


def test_figure_only_page_serves_note_alone(tmp_path):
    # Empty cleaned body but a chart block in the scan -> the note carries the page.
    store = _store(tmp_path)
    text = store.text(PageRef(stem=STEM, page=1))
    assert text == figure_note(1, ["Cover Figure"])


def test_missing_and_scanless_pages(tmp_path):
    store = _store(tmp_path)
    # No cleaned file and no scan -> None (falsy => extract defers to the vision tier).
    assert store.text(PageRef(stem=STEM, page=99999)) is None
    # Cleaned text present but no scan entry -> served as-is, no note.
    assert store.text(PageRef(stem=STEM, page=30)) == "Plain page, no scan entry."
    # Incomplete refs are None (a page-less ref is the only constructible incomplete one;
    # PageRef forbids page-without-stem at construction).
    assert store.text(PageRef(stem=STEM, page=None)) is None
    assert store.text(PageRef(stem=None, page=None)) is None


def test_text_is_memoized(tmp_path):
    store = _store(tmp_path)
    ref = PageRef(stem=STEM, page=13)
    first = store.text(ref)
    assert (STEM, 13) in store._text
    assert store.text(ref) == first


def test_missing_cleaned_dir_raises(tmp_path):
    # Scans present, cleaned/ absent -> a clear StepFailed, not a silent empty read.
    (tmp_path / SCANS_SUBDIR).mkdir()
    store = PageStore(tmp_path, pdf_dir="/nonexistent")
    try:
        store.text(PageRef(stem=STEM, page=13))
    except StepFailed as e:
        assert "cleaned pages not found" in str(e)
    else:
        raise AssertionError("expected StepFailed when cleaned/ is missing")


if __name__ == "__main__":
    import tempfile

    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            with tempfile.TemporaryDirectory() as d:
                fn(Path(d))
            print(f"ok: {name}")
    print("all passed")
