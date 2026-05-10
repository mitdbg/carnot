"""Integration tests for format, lookup_external, compute, and read_visual operators.

Each test makes real LLM calls (no mocks).

Format tests: edge cases on Python input types — no officeQA ground truth.
All other tests: strict numeric / factual acceptance criteria.

Run with: pytest tests/test_operators.py -v -s
"""

from __future__ import annotations

import math
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.expanduser("~/Desktop/officeqa"))

from skunk.dsl import DocHandle, FormattedString, OpNode, PageRef, TypedValue
from skunk.subagents.base import HarnessContext
import skunk.subagents.format as fmt
import skunk.subagents.lookup_external as lookup_external
import skunk.subagents.compute as compute
import skunk.subagents.read_visual as read_visual

PDF_DIR = os.path.expanduser("~/Desktop/officeqa/treasury_bulletin_pdfs")
SEPT_1990_PDF = os.path.join(PDF_DIR, "treasury_bulletin_1990_09.pdf")


@pytest.fixture()
def ctx(tmp_path):
    return HarnessContext(question="test", cache_dir=str(tmp_path))


def _num(text: str) -> float:
    clean = text.replace(",", "").replace("$", "").replace("%", "").strip()
    m = re.search(r"-?\d+\.?\d*", clean)
    assert m, f"No number found in {text!r}"
    return float(m.group())


# ---------------------------------------------------------------------------
# format
# ---------------------------------------------------------------------------

class TestFormat:

    def _run(self, ctx, value, **args):
        op = OpNode(op="format", args=args)
        prev = TypedValue(value=value, dtype="scalar", desc="test")
        result = fmt.run(op, prev, ctx)
        assert isinstance(result, FormattedString), f"Expected FormattedString, got {type(result)}"
        assert result.text.strip(), "result.text must not be empty/whitespace"
        assert "\n" not in result.text, f"result.text must be single-line: {result.text!r}"
        assert len(result.text) <= 250, f"result.text too long: {len(result.text)}"
        return result.text

    def test_integer_precision_zero(self, ctx):
        text = self._run(ctx, 42, precision=0)
        assert _num(text) == 42.0

    def test_float_precision_two(self, ctx):
        text = self._run(ctx, 3.14159, precision=2)
        assert abs(_num(text) - 3.14) < 1e-9

    def test_percentage_unit(self, ctx):
        text = self._run(ctx, 15.5, precision=2, unit="pct")
        assert "%" in text, f"Expected '%' in: {text!r}"
        assert abs(_num(text) - 15.5) < 0.005

    def test_negative_number(self, ctx):
        text = self._run(ctx, -7.3, precision=1)
        assert _num(text) < 0, f"Expected negative in: {text!r}"
        assert abs(_num(text) - (-7.3)) < 0.005

    def test_zero_value(self, ctx):
        text = self._run(ctx, 0.0, precision=2)
        assert _num(text) == 0.0

    def test_large_integer(self, ctx):
        text = self._run(ctx, 1_000_000, precision=0)
        assert _num(text) == 1_000_000.0

    def test_high_precision(self, ctx):
        text = self._run(ctx, 0.88525, precision=5)
        assert abs(_num(text) - 0.88525) < 5e-6

    def test_list_of_floats(self, ctx):
        op = OpNode(op="format", args={"precision": 1})
        prev = TypedValue(value=[1.5, 2.5, 3.5], dtype="list[scalar]", desc="test list")
        result = fmt.run(op, prev, ctx)
        assert isinstance(result, FormattedString)
        nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", result.text)]
        for expected in [1.5, 2.5, 3.5]:
            assert any(abs(n - expected) < 0.05 for n in nums), (
                f"Expected {expected} in formatted list: {result.text!r}"
            )

    def test_formatted_string_passthrough(self, ctx):
        op = OpNode(op="format", args={"precision": 2})
        prev = FormattedString(text="already done", desc="pre-formatted")
        result = fmt.run(op, prev, ctx)
        assert result.text == "already done"

    def test_negative_percentage(self, ctx):
        text = self._run(ctx, -18.51, precision=2, unit="pct")
        assert "%" in text
        assert _num(text) < 0
        assert abs(_num(text) - (-18.51)) < 0.005


# ---------------------------------------------------------------------------
# lookup_external
# ---------------------------------------------------------------------------

class TestLookupExternal:

    def _rel_err(self, got, expected):
        return abs(got - expected) / abs(expected) if expected != 0 else abs(got)

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
# compute
# ---------------------------------------------------------------------------

class TestCompute:

    def _rel_err(self, got, expected):
        return abs(got - expected) / abs(expected) if expected != 0 else abs(got)

    def test_uid0003_sum_of_list(self, ctx):
        op = OpNode(op="compute", args={"nl": "sum all values in the list"})
        prev = TypedValue(value=[10.0, 20.0, 30.0], dtype="list[scalar]", desc="monthly values")
        result = compute.run(op, prev, ctx)
        assert isinstance(result, TypedValue)
        assert abs(float(result.value) - 60.0) < 1e-6, f"Expected 60.0, got {result.value}"

    def test_uid0004_absolute_pct_change(self, ctx):
        op = OpNode(op="compute", args={"nl": "absolute percent change between the two branch values"})
        prev = [
            TypedValue(value=2602.0, dtype="scalar:usd_millions", desc="CY1940"),
            TypedValue(value=44463.0, dtype="scalar:usd_millions", desc="CY1953"),
        ]
        result = compute.run(op, prev, ctx)
        expected = abs((44463.0 - 2602.0) / 2602.0) * 100
        assert isinstance(result, TypedValue)
        assert self._rel_err(float(result.value), expected) < 0.001, (
            f"Expected ≈{expected:.4f}, got {result.value}"
        )

    def test_uid0018_geometric_mean(self, ctx):
        op = OpNode(op="compute", args={"nl": "geometric mean of the list of values"})
        prev = TypedValue(value=[1.0, 4.0, 16.0], dtype="list[scalar]", desc="monthly outlays")
        result = compute.run(op, prev, ctx)
        expected = (1.0 * 4.0 * 16.0) ** (1.0 / 3.0)
        assert isinstance(result, TypedValue)
        assert self._rel_err(float(result.value), expected) < 0.001, (
            f"Expected ≈{expected:.6f}, got {result.value}"
        )

    def test_uid0013_linear_regression(self, ctx):
        op = OpNode(op="compute", args={
            "nl": (
                "Fit OLS linear regression of y_values vs x_values using numpy polyfit. "
                "Return [slope, intercept] each rounded to 3 decimal places."
            )
        })
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [5.0, 7.0, 9.0, 11.0, 13.0]  # y = 2x + 3
        prev = TypedValue(value={"x_values": xs, "y_values": ys}, dtype="df", desc="regression input")
        result = compute.run(op, prev, ctx)
        assert isinstance(result, TypedValue)
        vals = result.value
        assert isinstance(vals, list) and len(vals) == 2, f"Expected [slope, intercept], got {vals!r}"
        slope, intercept = float(vals[0]), float(vals[1])
        assert self._rel_err(slope, 2.0) < 0.001, f"Expected slope≈2.0, got {slope}"
        assert self._rel_err(intercept, 3.0) < 0.001, f"Expected intercept≈3.0, got {intercept}"


# ---------------------------------------------------------------------------
# read_visual
# ---------------------------------------------------------------------------

class TestReadVisual:

    @pytest.mark.skipif(
        not os.path.exists(SEPT_1990_PDF),
        reason="Treasury Bulletin PDF corpus not present",
    )
    def test_uid0030_well_formed_output(self, ctx):
        op = OpNode(op="read_visual", args={
            "concept": "count of local maxima across all line plots on the page",
        })
        ref = PageRef(month="1990-09", pdf_page=7, file_path=SEPT_1990_PDF)
        prev = DocHandle(refs=[ref], desc="Sept 1990 bulletin page")
        result = read_visual.run(op, prev, ctx)
        assert isinstance(result, TypedValue), f"Expected TypedValue, got {type(result)}"
        assert result.value is not None
        assert result.dtype not in ("", "unknown"), f"dtype should be set, got {result.dtype!r}"
        assert len(result.desc) > 0
