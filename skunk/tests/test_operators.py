"""Integration tests for lookup_external, compute, and extract operators.

Each test makes real LLM calls (no mocks). Tests assert on the *shape* of the
output (right type, plausible numeric magnitude) rather than exact values, since
LLM outputs are non-deterministic.

Run with: pytest tests/test_operators.py -v -s
"""

from __future__ import annotations

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.expanduser("~/Desktop/officeqa"))

from skunk.dsl import AnnotatedValue, DocHandle, FormattedString, OpNode, PageRef


def _rel_err(got, expected):
    return abs(got - expected) / abs(expected) if expected != 0 else abs(got)


import skunk.subagents.compute as compute  # noqa: E402
import skunk.subagents.extract as extract  # noqa: E402
import skunk.subagents.lookup_external as lookup_external  # noqa: E402
from skunk.common import HarnessContext  # noqa: E402

PDF_DIR = os.path.expanduser("~/Desktop/officeqa/treasury_bulletin_pdfs")
PARSED_JSON_DIR = os.path.expanduser(
    "~/Desktop/officeqa/treasury_bulletins_parsed/jsons"
)
SEPT_1990_PDF = os.path.join(PDF_DIR, "treasury_bulletin_1990_09.pdf")
JAN_1941_PDF = os.path.join(PDF_DIR, "treasury_bulletin_1941_01.pdf")
JAN_1941_JSON = os.path.join(PARSED_JSON_DIR, "treasury_bulletin_1941_01.json")


@pytest.fixture()
def ctx():
    return HarnessContext(question="test")


def _ctx_for(question: str) -> HarnessContext:
    return HarnessContext(question=question)


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
        assert isinstance(result, list)
        assert result[0].value == 1945, f"Expected 1945, got {result.value['']}"

    def test_uid0055_korean_war_start(self, ctx):
        op = OpNode(op="lookup_external", args={"nl": "year the Korean War started"})
        result = lookup_external.run(op, None, ctx)
        assert isinstance(result, list)
        assert result[0].value == 1950, f"Expected 1950, got {result.value['']}"

    def test_uid0055_germany_invaded_poland(self, ctx):
        op = OpNode(op="lookup_external", args={"nl": "year Germany invaded Poland"})
        result = lookup_external.run(op, None, ctx)
        assert isinstance(result, list)
        assert result[0].value == 1939, f"Expected 1939, got {result.value['']}"

    def test_uid0010_usd_jpy_rate_2025(self, ctx):
        op = OpNode(op="lookup_external", args={"nl": "USD/JPY exchange rate on 2025-03-31"})
        result = lookup_external.run(op, None, ctx)
        assert isinstance(result, list)
        rate = float(result[0].value)
        assert 147 <= rate <= 153, f"USD/JPY on 2025-03-31 should be ≈149-151, got {rate}"


# ---------------------------------------------------------------------------
# compute (chain terminator: question-driven, returns FormattedString)
# ---------------------------------------------------------------------------

class TestCompute:

    def test_sum_of_list(self):
        ctx = _ctx_for("What is the sum of the values? Report as a plain integer with no commas.")
        prev = [AnnotatedValue(description="three numbers", value=[10.0, 20.0, 30.0])]
        result = compute.run(OpNode(op="compute"), prev, ctx)
        assert isinstance(result, FormattedString)
        assert _num(result.text) == 60.0, f"Expected 60, got {result.text!r}"

    def test_named_values_sum_with_unit_context(self):
        ctx = _ctx_for(
            "What was the total US national defense expenditure in calendar year 1940? "
            "Report in millions of nominal dollars rounded to the nearest whole."
        )
        monthly = [132, 129, 143, 159, 154, 153, 177, 200, 219, 287, 376, 473]
        entries = [
            AnnotatedValue(
                description=f"US national defense expenditures, 1940-{i:02d}, monthly",
                value=v,
                unit="usd_millions",
            )
            for i, v in enumerate(monthly, start=1)
        ]
        prev = entries
        result = compute.run(OpNode(op="compute"), prev, ctx)
        assert isinstance(result, FormattedString)
        assert _rel_err(_num(result.text), 2602.0) < 0.001, f"Expected ≈2602, got {result.text!r}"

    def test_absolute_pct_change_parallel(self):
        ctx = _ctx_for(
            "What was the absolute percent change between the two values, as a percent value (e.g. 12.34%)?"
        )
        prev = [
            AnnotatedValue(description="CY1940 total", value=2602.0, unit="usd_millions"),
            AnnotatedValue(description="CY1953 total", value=44463.0, unit="usd_millions"),
        ]
        result = compute.run(OpNode(op="compute"), prev, ctx)
        assert isinstance(result, FormattedString)
        expected = abs((44463.0 - 2602.0) / 2602.0) * 100
        assert _rel_err(_num(result.text), expected) < 0.001, (
            f"Expected ≈{expected:.2f}%, got {result.text!r}"
        )

    def test_missing_data_failure_mode(self):
        from skunk.subagents.base import StepFailed
        ctx = _ctx_for(
            "What is the unemployment rate for January 1955? Report as a percent."
        )
        prev = [AnnotatedValue(description="something else entirely", value=42, unit="count")]
        with pytest.raises(StepFailed) as excinfo:
            compute.run(OpNode(op="compute"), prev, ctx)
        assert "missing" in str(excinfo.value).lower() or "compute" in excinfo.value.op


# ---------------------------------------------------------------------------
# extract(visual_only=True) — vision-only path for charts/figures
# ---------------------------------------------------------------------------

class TestExtractVisualOnly:

    @pytest.mark.skipif(
        not os.path.exists(SEPT_1990_PDF),
        reason="Treasury Bulletin PDF corpus not present",
    )
    def test_uid0030_well_formed_output(self):
        ctx = _ctx_for(
            "Count the local maxima across all line plots on the page."
        )
        op = OpNode(op="extract", args={"visual_only": True})
        ref = PageRef(month="1990-09", page=7, file_path=SEPT_1990_PDF)
        prev = DocHandle(refs=[ref], desc="Sept 1990 bulletin page")
        result = extract.run(op, prev, ctx)
        assert isinstance(result, list), f"Expected list[AnnotatedValue], got {type(result)}"
        assert len(result) > 0


# ---------------------------------------------------------------------------
# extract (no args; emits dict-of-named-values relevant to ctx.question)
# ---------------------------------------------------------------------------

class TestExtract:

    @pytest.mark.skipif(
        not os.path.exists(JAN_1941_JSON),
        reason="Parsed JSON corpus not present",
    )
    def test_uid0001_pdf_page_15_named_extraction(self):
        ctx = _ctx_for(
            "What were the total expenditures (in millions of nominal dollars) for U.S "
            "national defense in the calendar year of 1940?"
        )
        op = OpNode(op="extract")
        ref = PageRef(month="1941-01", page=15)
        prev = DocHandle(refs=[ref])
        result = extract.run(op, prev, ctx)
        assert isinstance(result, list)
        assert len(result) > 0
        # The page contains only FY data on national defense; the agent should still emit a
        # plausible national-defense-related entry (description should mention 'defense').
        descriptions = [e.description for e in result]
        assert any("defense" in d.lower() for d in descriptions), (
            f"Expected at least one defense-related description, got {descriptions!r}"
        )

