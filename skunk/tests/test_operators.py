"""Integration tests for lookup_external, compute, read_visual, and extract operators.

Each test makes real LLM calls (no mocks). Tests assert on the *shape* of the
output (right type, plausible numeric magnitude) rather than exact values, since
LLM outputs are non-deterministic.

Run with: pytest tests/test_operators.py -v -s
"""

from __future__ import annotations

import math  # noqa: F401
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.expanduser("~/Desktop/officeqa"))

from skunk.dsl import DocHandle, FormattedString, NamedEntry, OpNode, PageRef, TypedValue


def _rel_err(got, expected):
    return abs(got - expected) / abs(expected) if expected != 0 else abs(got)


import skunk.subagents.compute as compute  # noqa: E402
import skunk.subagents.extract as extract  # noqa: E402
import skunk.subagents.lookup_external as lookup_external  # noqa: E402
import skunk.subagents.read_visual as read_visual  # noqa: E402
from skunk.common.context import HarnessContext  # noqa: E402

PDF_DIR = os.path.expanduser("~/Desktop/officeqa/treasury_bulletin_pdfs")
PARSED_JSON_DIR = os.path.expanduser(
    "~/Desktop/officeqa/treasury_bulletins_parsed/jsons"
)
SEPT_1990_PDF = os.path.join(PDF_DIR, "treasury_bulletin_1990_09.pdf")
JAN_1941_PDF = os.path.join(PDF_DIR, "treasury_bulletin_1941_01.pdf")
JAN_1941_JSON = os.path.join(PARSED_JSON_DIR, "treasury_bulletin_1941_01.json")


@pytest.fixture()
def ctx(tmp_path):
    return HarnessContext(question="test", cache_dir=str(tmp_path))


def _ctx_for(question: str, tmp_path) -> HarnessContext:
    return HarnessContext(question=question, cache_dir=str(tmp_path))


def _num(text: str) -> float:
    clean = text.replace(",", "").replace("$", "").replace("%", "").strip()
    m = re.search(r"-?\d+\.?\d*", clean)
    assert m, f"No number found in {text!r}"
    return float(m.group())


# ---------------------------------------------------------------------------
# lookup_external
# ---------------------------------------------------------------------------

class TestLookupExternal:

    def test_uid0055_wwii_end(self, ctx):
        op = OpNode(op="lookup_external", args={"nl": "year that WWII ended"})
        result = lookup_external.run(op, None, ctx)
        assert isinstance(result, TypedValue)
        assert result.value == 1945, f"Expected 1945, got {result.value}"

    def test_uid0055_korean_war_start(self, ctx):
        op = OpNode(op="lookup_external", args={"nl": "year the Korean War started"})
        result = lookup_external.run(op, None, ctx)
        assert isinstance(result, TypedValue)
        assert result.value == 1950, f"Expected 1950, got {result.value}"

    def test_uid0055_germany_invaded_poland(self, ctx):
        op = OpNode(op="lookup_external", args={"nl": "year Germany invaded Poland"})
        result = lookup_external.run(op, None, ctx)
        assert isinstance(result, TypedValue)
        assert result.value == 1939, f"Expected 1939, got {result.value}"

    def test_uid0010_usd_jpy_rate_2025(self, ctx):
        op = OpNode(op="lookup_external", args={"nl": "USD/JPY exchange rate on 2025-03-31"})
        result = lookup_external.run(op, None, ctx)
        assert isinstance(result, TypedValue)
        rate = float(result.value)
        assert 147 <= rate <= 153, f"USD/JPY on 2025-03-31 should be ≈149-151, got {rate}"


# ---------------------------------------------------------------------------
# compute (chain terminator: question-driven, returns FormattedString)
# ---------------------------------------------------------------------------

class TestCompute:

    def test_sum_of_list(self, tmp_path):
        ctx = _ctx_for("What is the sum of the values? Report as a plain integer with no commas.", tmp_path)
        prev = TypedValue(value=[10.0, 20.0, 30.0], dtype="list[scalar]", desc="monthly values")
        result = compute.run(OpNode(op="compute"), prev, ctx)
        assert isinstance(result, FormattedString)
        assert _num(result.text) == 60.0, f"Expected 60, got {result.text!r}"

    def test_named_values_sum_with_unit_context(self, tmp_path):
        ctx = _ctx_for(
            "What was the total US national defense expenditure in calendar year 1940? "
            "Report in millions of nominal dollars rounded to the nearest whole.",
            tmp_path,
        )
        monthly = [132, 129, 143, 159, 154, 153, 177, 200, 219, 287, 376, 473]
        # New contract: flat scalars with dims. One named scalar per month;
        # dims={'month': 'YYYY-MM'} disambiguates siblings.
        value: dict = {}
        meta: dict = {}
        for i, v in enumerate(monthly, start=1):
            name = f"national_defense_cy1940_m{i:02d}"
            value[name] = v
            meta[name] = NamedEntry(
                unit="usd_millions",
                quote=f"1940 month {i}",
                dims={"year": 1940, "month": f"1940-{i:02d}", "sub_category": "national_defense"},
            )
        prev = TypedValue(value=value, dtype="named", unit="", desc="", meta=meta)
        result = compute.run(OpNode(op="compute"), prev, ctx)
        assert isinstance(result, FormattedString)
        assert _rel_err(_num(result.text), 2602.0) < 0.001, f"Expected ≈2602, got {result.text!r}"

    def test_absolute_pct_change_parallel(self, tmp_path):
        ctx = _ctx_for(
            "What was the absolute percent change between the two values, as a percent value (e.g. 12.34%)?",
            tmp_path,
        )
        prev = [
            TypedValue(value=2602.0, dtype="scalar", unit="usd_millions", desc="CY1940 total"),
            TypedValue(value=44463.0, dtype="scalar", unit="usd_millions", desc="CY1953 total"),
        ]
        result = compute.run(OpNode(op="compute"), prev, ctx)
        assert isinstance(result, FormattedString)
        expected = abs((44463.0 - 2602.0) / 2602.0) * 100
        assert _rel_err(_num(result.text), expected) < 0.001, (
            f"Expected ≈{expected:.2f}%, got {result.text!r}"
        )

    def test_missing_data_failure_mode(self, tmp_path):
        from skunk.subagents.base import StepFailed
        ctx = _ctx_for(
            "What is the unemployment rate for January 1955? Report as a percent.",
            tmp_path,
        )
        # prev contains nothing about unemployment
        prev = TypedValue(
            value={"some_unrelated_value": 42},
            dtype="named", unit="",
            desc="",
            meta={"some_unrelated_value": NamedEntry(unit="count", quote="something else entirely")},
        )
        with pytest.raises(StepFailed) as excinfo:
            compute.run(OpNode(op="compute"), prev, ctx)
        assert "missing" in str(excinfo.value).lower() or "compute" in excinfo.value.op


# ---------------------------------------------------------------------------
# read_visual (no args; emits dict-of-named-values like extract)
# ---------------------------------------------------------------------------

class TestReadVisual:

    @pytest.mark.skipif(
        not os.path.exists(SEPT_1990_PDF),
        reason="Treasury Bulletin PDF corpus not present",
    )
    def test_uid0030_well_formed_output(self, tmp_path):
        ctx = _ctx_for(
            "Count the local maxima across all line plots on the page.",
            tmp_path,
        )
        op = OpNode(op="read_visual")
        ref = PageRef(month="1990-09", page=7, file_path=SEPT_1990_PDF)
        prev = DocHandle(refs=[ref], desc="Sept 1990 bulletin page")
        result = read_visual.run(op, prev, ctx)
        assert isinstance(result, TypedValue), f"Expected TypedValue, got {type(result)}"
        assert result.value is not None
        # New extract/read_visual contract: dtype='named' with a dict of values.
        # An empty dict (nothing relevant) raises StepFailed before we get here.
        assert result.dtype == "named"
        assert isinstance(result.value, dict)


# ---------------------------------------------------------------------------
# extract (no args; emits dict-of-named-values relevant to ctx.question)
# ---------------------------------------------------------------------------

class TestExtract:

    @pytest.mark.skipif(
        not os.path.exists(JAN_1941_JSON),
        reason="Parsed JSON corpus not present",
    )
    def test_uid0001_pdf_page_15_named_extraction(self, tmp_path):
        ctx = _ctx_for(
            "What were the total expenditures (in millions of nominal dollars) for U.S "
            "national defense in the calendar year of 1940?",
            tmp_path,
        )
        op = OpNode(op="extract")
        ref = PageRef(month="1941-01", page=15)
        prev = DocHandle(refs=[ref])
        result = extract.run(op, prev, ctx)
        assert isinstance(result, TypedValue)
        assert result.dtype == "named"
        assert isinstance(result.value, dict) and len(result.value) > 0
        # The page contains only FY data on national defense; the agent should still emit a
        # plausible national-defense-related entry (name varies but should mention 'defense').
        names = list(result.value.keys())
        assert any("defense" in n.lower() for n in names), (
            f"Expected at least one defense-related key, got {names!r}"
        )

    @pytest.mark.skipif(
        not os.path.exists(JAN_1941_PDF),
        reason="PDF corpus not present",
    )
    def test_tier2_cache_created(self, tmp_path):
        from pathlib import Path
        from skunk.common.pdf_text import get_ocr_text_for_pdf_page

        ctx = HarnessContext(question="x", cache_dir=str(tmp_path))
        ref = PageRef(month="1941-01", page=15, file_path=JAN_1941_PDF)
        text = get_ocr_text_for_pdf_page(ref, ctx)
        pages_dir = Path(ctx.cache_dir) / "pages" / "1941-01"
        txt_files = list(pages_dir.glob("p*.txt")) if pages_dir.exists() else []
        assert txt_files, "Cache .txt file should have been written"
        text2 = get_ocr_text_for_pdf_page(ref, ctx)
        assert text == text2
