"""Tests for the JSON-backed per-page index."""

from __future__ import annotations

import pytest

from skunk.common.context import HarnessContext
from skunk.common.parsed_json import (
    _load_doc,
    _page_index,
    get_printed_page,
    get_text_for_pdf_page,
    parsed_json_dir,
)
from skunk.dsl import PageRef

JUNE_2025_JSON = parsed_json_dir() / "treasury_bulletin_2025_06.json"
JAN_1941_JSON = parsed_json_dir() / "treasury_bulletin_1941_01.json"


def _ctx() -> HarnessContext:
    return HarnessContext(question="")


@pytest.fixture(autouse=True)
def _clear_cache():
    # parsed_json caches via lru_cache on functions; clear so tests are deterministic.
    _load_doc.cache_clear()
    _page_index.cache_clear()


@pytest.mark.skipif(not JUNE_2025_JSON.exists(), reason="Parsed JSON corpus not present")
def test_get_text_for_pdf_page_2025_06_p76_has_esf1_table():
    text = get_text_for_pdf_page(PageRef(month="2025-06", page=76), _ctx())
    assert text is not None and len(text) > 1000
    assert "Japanese yen" in text
    assert "<table>" in text  # tables come through as HTML


@pytest.mark.skipif(not JUNE_2025_JSON.exists(), reason="Parsed JSON corpus not present")
def test_get_text_for_pdf_page_missing_returns_none():
    assert get_text_for_pdf_page(PageRef(month="2025-06", page=999), _ctx()) is None


@pytest.mark.skipif(not JUNE_2025_JSON.exists(), reason="Parsed JSON corpus not present")
def test_get_printed_page_reverse_lookup():
    # PDF page 76 of June 2025 is bulletin printed page "69"
    assert get_printed_page(PageRef(month="2025-06", page=76), _ctx()) == "69"


@pytest.mark.skipif(not JUNE_2025_JSON.exists(), reason="Parsed JSON corpus not present")
def test_get_printed_page_returns_none_when_no_footer():
    # Cover page typically has no printed-page footer marker
    assert get_printed_page(PageRef(month="2025-06", page=1), _ctx()) is None


@pytest.mark.skipif(not JAN_1941_JSON.exists(), reason="Parsed JSON corpus not present")
def test_old_bulletin_pdf_page_returns_rich_content():
    # Sanity: PDF page 15 of 1941-01 returns substantial structured content
    # (the budget expenditures table) via the JSON path.
    text = get_text_for_pdf_page(PageRef(month="1941-01", page=15), _ctx())
    assert text is not None and len(text) > 1000
    assert "<table>" in text
    assert "National defense" in text
    # Reverse-lookup: PDF page 15 of 1941-01 prints "5" in its footer (small front-matter drift).
    assert get_printed_page(PageRef(month="1941-01", page=15), _ctx()) == "5"


def test_missing_month_returns_none():
    # A month we don't have JSON for must yield None, not raise.
    assert get_text_for_pdf_page(PageRef(month="1800-01", page=1), _ctx()) is None
    assert get_printed_page(PageRef(month="1800-01", page=1), _ctx()) is None
