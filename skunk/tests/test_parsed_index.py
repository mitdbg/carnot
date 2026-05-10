"""Unit tests for the parsed-bulletin page slicer (common/parsed.py)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_PARSED_DIR = os.path.expanduser(
    "~/Desktop/officeqa/treasury_bulletins_parsed/transformed/treasury_bulletins_transformed"
)

_has_1990_09 = os.path.exists(os.path.join(_PARSED_DIR, "treasury_bulletin_1990_09.txt"))
_has_1939_01 = os.path.exists(os.path.join(_PARSED_DIR, "treasury_bulletin_1939_01.txt"))

skip_1990_09 = pytest.mark.skipif(not _has_1990_09, reason="Parsed corpus not present: 1990_09")
skip_1939_01 = pytest.mark.skipif(not _has_1939_01, reason="Parsed corpus not present: 1939_01")


@pytest.fixture()
def cache(tmp_path):
    return str(tmp_path)


@skip_1990_09
def test_index_coverage_1990_09(cache):
    from skunk.common.parsed import build_parsed_page_index

    index = build_parsed_page_index("1990-09", cache)
    assert index, "Index must not be empty"
    pages = [int(k) for k in index]
    max_page = max(pages)
    expected = set(range(3, max_page + 1))
    coverage = len(set(pages) & expected) / len(expected)
    assert coverage >= 0.8, f"Coverage {coverage:.1%} below 80% for pages 3–{max_page}"


@skip_1990_09
def test_index_monotonic_ranges_1990_09(cache):
    from skunk.common.parsed import build_parsed_page_index

    index = build_parsed_page_index("1990-09", cache)
    prev_end = -1
    for p in sorted(int(k) for k in index):
        start, end = index[str(p)]
        assert start > prev_end, f"Page {p} range [{start},{end}] overlaps previous end {prev_end}"
        assert start <= end, f"Page {p} has empty range [{start},{end}]"
        prev_end = end


@skip_1990_09
def test_slice_text_nonempty_1990_09(cache):
    from skunk.common.context import HarnessContext
    from skunk.common.parsed import build_parsed_page_index, get_parsed_text_for_page
    from skunk.dsl import PageRef

    index = build_parsed_page_index("1990-09", cache)
    pages = sorted(int(k) for k in index)
    sample = pages[len(pages) // 2]
    ctx = HarnessContext(question="test", cache_dir=cache)
    text = get_parsed_text_for_page(PageRef(month="1990-09", page=sample), ctx)
    assert text is not None, f"Text for page {sample} should not be None"
    assert len(text) > 10, f"Text for page {sample} too short: {text!r}"


@skip_1939_01
def test_index_coverage_1939_01(cache):
    from skunk.common.parsed import build_parsed_page_index

    index = build_parsed_page_index("1939-01", cache)
    assert index, "Index for 1939-01 must not be empty"


@skip_1990_09
def test_cache_roundtrip_1990_09(cache):
    from skunk.common.parsed import build_parsed_page_index

    index1 = build_parsed_page_index("1990-09", cache)
    cache_file = Path(cache) / "parsed_index" / "1990-09.json"
    assert cache_file.exists(), "Cache file should be written after first build"
    index2 = build_parsed_page_index("1990-09", cache)
    assert index1 == index2, "Cache read must match original build"


@skip_1990_09
def test_missing_month_returns_none(cache):
    from skunk.common.context import HarnessContext
    from skunk.common.parsed import get_parsed_text_for_page
    from skunk.dsl import PageRef

    ctx = HarnessContext(question="test", cache_dir=cache)
    text = get_parsed_text_for_page(PageRef(month="1990-09", page=9999), ctx)
    assert text is None, "Page 9999 should not be in any bulletin"
