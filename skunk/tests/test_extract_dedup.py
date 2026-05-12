"""Unit tests for extract's dedup pick-only enforcement.

`_dedup_semantically` runs the LLM with `_DEDUP_SYSTEM` ("you are a picker, not
a calculator"), then runs `_output_entry_in_inputs` over each output entry as
a structural verifier. These tests poke the verifier directly with crafted
inputs/outputs to lock in the pick-only contract, and exercise the full
dedup pipeline with a mocked LLM to confirm computed outputs are rejected.
"""

from __future__ import annotations

from skunk.common import HarnessContext, LLMResponse
from skunk.dsl import AnnotatedValue
from skunk.subagents.extract import (
    _dedup_semantically,
    _output_entry_in_inputs,
)


# ---------------------------------------------------------------------------
# _output_entry_in_inputs — pure-function structural verifier
# ---------------------------------------------------------------------------

def test_verifier_accepts_scalar_picked_from_scalar_input():
    inputs = [AnnotatedValue(description="d", value=42, unit="count")]
    output = AnnotatedValue(description="d (rewording)", value=42, unit="count")
    assert _output_entry_in_inputs(output, inputs)


def test_verifier_accepts_scalar_picked_from_vector_cell():
    inputs = [AnnotatedValue(
        description="series", value={"1940": 10, "1953": 20}, kind="vector",
        unit="usd_millions", index_name="year",
    )]
    output = AnnotatedValue(description="picked", value=20, unit="usd_millions")
    assert _output_entry_in_inputs(output, inputs)


def test_verifier_rejects_computed_scalar_sum():
    # Inputs are vector cells; output is their sum — not present anywhere.
    inputs = [AnnotatedValue(
        description="series", value={"1940": 10, "1953": 20}, kind="vector",
        unit="usd_millions", index_name="year",
    )]
    output = AnnotatedValue(description="total", value=30, unit="usd_millions")
    assert not _output_entry_in_inputs(output, inputs)


def test_verifier_rejects_computed_scalar_average():
    inputs = [AnnotatedValue(
        description="series", value={"1940": 10, "1953": 20}, kind="vector",
        unit="usd_millions", index_name="year",
    )]
    output = AnnotatedValue(description="mean", value=15, unit="usd_millions")
    assert not _output_entry_in_inputs(output, inputs)


def test_verifier_accepts_vector_subset_picked_from_single_input():
    inputs = [AnnotatedValue(
        description="full series",
        value={"1940": 10, "1953": 20, "1961": 30}, kind="vector",
        unit="usd_millions", index_name="year",
    )]
    output = AnnotatedValue(
        description="subset", value={"1940": 10, "1953": 20}, kind="vector",
        unit="usd_millions", index_name="year",
    )
    assert _output_entry_in_inputs(output, inputs)


def test_verifier_rejects_vector_with_invented_cell():
    inputs = [AnnotatedValue(
        description="full series",
        value={"1940": 10, "1953": 20}, kind="vector",
        unit="usd_millions", index_name="year",
    )]
    output = AnnotatedValue(
        description="with extra year",
        value={"1940": 10, "1953": 20, "1962": 99}, kind="vector",
        unit="usd_millions", index_name="year",
    )
    assert not _output_entry_in_inputs(output, inputs)


def test_verifier_rejects_grafted_vector_across_inputs():
    # Two separate inputs; output grafts cells from both. Should reject.
    inputs = [
        AnnotatedValue(description="A", value={"1940": 10}, kind="vector",
                       unit="usd_millions", index_name="year"),
        AnnotatedValue(description="B", value={"1953": 20}, kind="vector",
                       unit="usd_millions", index_name="year"),
    ]
    output = AnnotatedValue(
        description="grafted",
        value={"1940": 10, "1953": 20}, kind="vector",
        unit="usd_millions", index_name="year",
    )
    assert not _output_entry_in_inputs(output, inputs)


def test_verifier_rejects_rescaled_vector():
    # Output multiplies every cell by 1000 (e.g. usd_millions → usd; same keys
    # but new values not present in any input).
    inputs = [AnnotatedValue(
        description="series", value={"1940": 10, "1953": 20}, kind="vector",
        unit="usd_millions", index_name="year",
    )]
    output = AnnotatedValue(
        description="rescaled", value={"1940": 10000, "1953": 20000}, kind="vector",
        unit="usd", index_name="year",
    )
    assert not _output_entry_in_inputs(output, inputs)


def test_verifier_accepts_table_subset_from_single_input():
    inputs = [AnnotatedValue(
        description="grid", kind="table",
        value={"1940": {"01": 1, "02": 2}, "1953": {"01": 3, "02": 4}},
        unit="usd_millions", row_name="year", col_name="month",
    )]
    output = AnnotatedValue(
        description="subset", kind="table",
        value={"1940": {"01": 1}},
        unit="usd_millions", row_name="year", col_name="month",
    )
    assert _output_entry_in_inputs(output, inputs)


def test_verifier_rejects_table_with_changed_cell():
    inputs = [AnnotatedValue(
        description="grid", kind="table",
        value={"1940": {"01": 1, "02": 2}},
        unit="usd_millions", row_name="year", col_name="month",
    )]
    output = AnnotatedValue(
        description="tweaked", kind="table",
        value={"1940": {"01": 999}},  # value changed
        unit="usd_millions", row_name="year", col_name="month",
    )
    assert not _output_entry_in_inputs(output, inputs)


# ---------------------------------------------------------------------------
# End-to-end: _dedup_semantically with a mocked LLM
# ---------------------------------------------------------------------------

class _MockLLM:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.calls = 0

    def call(self, system: str, user: str, **kwargs) -> LLMResponse:
        self.calls += 1
        return LLMResponse(text=self.response_text, latency_s=0.0,
                           input_tokens=None, output_tokens=None)


def _ctx_with(text: str) -> HarnessContext:
    return HarnessContext(question="t", llm_client=_MockLLM(text))


def test_dedup_accepts_grounded_pick():
    inputs = [
        AnnotatedValue(description="CY1940 defense, monthly", kind="vector",
                       value={"1940-01": 132, "1940-02": 129},
                       unit="usd_millions", index_name="month"),
        AnnotatedValue(description="National defense in 1940, by month", kind="vector",
                       value={"1940-01": 132, "1940-02": 129},
                       unit="usd_millions", index_name="month"),
    ]
    # LLM picks the first as the representative.
    llm_response = """[
      {"description": "CY1940 defense, monthly", "kind": "vector",
       "value": {"1940-01": 132, "1940-02": 129},
       "unit": "usd_millions", "index_name": "month"}
    ]"""
    ctx = _ctx_with(llm_response)
    deduped = _dedup_semantically(inputs, ctx)
    assert len(deduped) == 1
    assert deduped[0].value == {"1940-01": 132, "1940-02": 129}


def test_dedup_rejects_computed_output_and_returns_empty():
    # Two distinct inputs so we exceed the short-circuit threshold (>1).
    inputs = [
        AnnotatedValue(description="series", kind="vector",
                       value={"1940": 10, "1953": 20},
                       unit="usd_millions", index_name="year"),
        AnnotatedValue(description="another series", kind="vector",
                       value={"1940": 11},
                       unit="usd_millions", index_name="year"),
    ]
    # LLM tries to "consolidate" by emitting the sum as a new scalar.
    llm_response = """[
      {"description": "computed total", "kind": "scalar",
       "value": 30, "unit": "usd_millions"}
    ]"""
    ctx = _ctx_with(llm_response)
    deduped = _dedup_semantically(inputs, ctx)
    # Every output entry fails verification → return [] (no fallback).
    assert deduped == []


def test_dedup_rejects_grafted_output_and_returns_empty():
    inputs = [
        AnnotatedValue(description="A", kind="vector",
                       value={"1940": 10},
                       unit="usd_millions", index_name="year"),
        AnnotatedValue(description="B", kind="vector",
                       value={"1953": 20},
                       unit="usd_millions", index_name="year"),
    ]
    # LLM grafts both inputs' cells into one entry.
    llm_response = """[
      {"description": "grafted", "kind": "vector",
       "value": {"1940": 10, "1953": 20},
       "unit": "usd_millions", "index_name": "year"}
    ]"""
    ctx = _ctx_with(llm_response)
    deduped = _dedup_semantically(inputs, ctx)
    assert deduped == []


def test_dedup_unparseable_response_returns_empty():
    inputs = [
        AnnotatedValue(description="A", value=1, unit="count"),
        AnnotatedValue(description="B", value=2, unit="count"),
    ]
    ctx = _ctx_with("this is not JSON at all")
    deduped = _dedup_semantically(inputs, ctx)
    assert deduped == []


def test_dedup_short_circuits_when_input_size_le_1():
    only_one = [AnnotatedValue(description="x", value=1, unit="count")]
    # No LLM call should fire — short-circuit returns the single entry as-is.
    llm = _MockLLM("")
    ctx = HarnessContext(question="t", llm_client=llm)
    deduped = _dedup_semantically(only_one, ctx)
    assert deduped == only_one
    assert llm.calls == 0
