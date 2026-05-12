"""Planner Phase 2 tests — verify both flat and decomposed JSON shapes round
through the planner correctly, and that max_compute_depth=1 strips
decomposition guidance from the system prompt.

The LLM is mocked; no Gemini calls happen.
"""

from __future__ import annotations

import json

import pytest

import skunk.planner as planner
from skunk.common import HarnessContext, LLMResponse
from skunk.config import SkunkConfig
from skunk.dsl import ComputeNode, Plan, serialize, validate
from skunk.subagents.base import StepFailed


class _MockLLM:
    """Returns a queued response on each `call()`. Raises on unexpected call."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = 0

    def call(self, system: str, user: str, **kwargs) -> LLMResponse:
        self.calls += 1
        assert self.responses, f"unexpected planner call #{self.calls}; user={user[:200]}"
        return LLMResponse(
            text=self.responses.pop(0), latency_s=0.0,
            input_tokens=None, output_tokens=None,
        )


def _ctx(llm: _MockLLM, max_depth: int = 2) -> HarnessContext:
    config = SkunkConfig.from_env()
    config.max_compute_depth = max_depth
    return HarnessContext(question="test question", llm_client=llm, config=config)


def _wrap_json(d: dict) -> str:
    return f"```json\n{json.dumps(d, indent=2)}\n```"


# ---------------------------------------------------------------------------
# Round-trip from a flat planner response
# ---------------------------------------------------------------------------

def test_planner_accepts_flat_response_and_produces_single_compute_plan():
    flat = {
        "branches": [
            {"kind": "retrieve", "concept": "national_defense_expenditures", "period": "CY1940"},
        ],
    }
    llm = _MockLLM([_wrap_json(flat)])
    ctx = _ctx(llm)
    p = planner.plan("What were CY1940 defense expenditures?", ctx)
    assert isinstance(p, Plan)
    assert len(p.computes) == 1
    assert p.computes[0].final is True
    assert p.computes[0].task == ""
    assert len(p.computes[0].branches) == 1
    assert validate(p, max_compute_depth=ctx.config.max_compute_depth).ok
    # Serialize stays in the legacy text form for back-compat.
    text = serialize(p)
    assert text == "retrieve(concept='national_defense_expenditures', period='CY1940') --> extract() --> compute()"


def test_planner_accepts_flat_parallel_response():
    flat = {
        "branches": [
            {"kind": "retrieve", "concept": "national_defense_expenditures", "period": "CY1940"},
            {"kind": "retrieve", "concept": "national_defense_expenditures", "period": "CY1953"},
        ],
    }
    llm = _MockLLM([_wrap_json(flat)])
    ctx = _ctx(llm)
    p = planner.plan("absolute pct change between CY1940 and CY1953", ctx)
    assert len(p.computes) == 1
    assert len(p.computes[0].branches) == 2


# ---------------------------------------------------------------------------
# Round-trip from a decomposed planner response
# ---------------------------------------------------------------------------

def test_planner_accepts_decomposed_response_at_default_depth():
    decomposed = {
        "computes": [
            {
                "task": "Total dollar value of bids submitted for 2-year notes maturing end of July 1984.",
                "branches": [
                    {"kind": "retrieve", "concept": "treasury_note_auction_results", "period": "1984-07"},
                ],
            },
            {
                "task": "Percent of those bids that were noncash rollover tenders from non-domestic investors.",
                "branches": [
                    {"kind": "retrieve", "concept": "treasury_note_auction_results", "period": "1984-07"},
                ],
            },
        ],
    }
    llm = _MockLLM([_wrap_json(decomposed)])
    ctx = _ctx(llm, max_depth=2)
    p = planner.plan("multi-part: total bids and percent rollover", ctx)
    # Two intermediates + appended final aggregator.
    assert len(p.computes) == 3
    assert p.computes[0].task.startswith("Total dollar value")
    assert p.computes[0].final is False
    assert p.computes[1].task.startswith("Percent of those bids")
    assert p.computes[1].final is False
    assert p.computes[-1].final is True
    assert p.computes[-1].branches == []
    assert validate(p, max_compute_depth=2).ok


# ---------------------------------------------------------------------------
# Depth=1 strips decomposition guidance from the system prompt
# ---------------------------------------------------------------------------

def test_depth_1_system_prompt_strips_decomposition():
    ctx = _ctx(_MockLLM([]), max_depth=1)
    system = planner._build_system(ctx)
    # No mention of the decomposed shape, the literal `computes` key, or
    # the `task` argument that's specific to intermediate computes.
    assert "computes" not in system
    assert "decompose" not in system.lower()
    # The UID0017 positive few-shot's first sentence shouldn't be there.
    assert "Two distinct quantitative outputs" not in system
    # But the flat few-shots and base spec stay.
    assert "branches" in system
    assert "Plan shape (flat)" in system


def test_depth_2_system_prompt_includes_decomposition():
    ctx = _ctx(_MockLLM([]), max_depth=2)
    system = planner._build_system(ctx)
    assert "computes" in system
    assert "Plan shape (decomposed)" in system
    # The positive few-shot appears.
    assert "treasury_note_auction_results" in system


# ---------------------------------------------------------------------------
# Depth=1 rejects a decomposed planner response (validator backstop)
# ---------------------------------------------------------------------------

def test_depth_1_rejects_decomposed_response():
    decomposed = {
        "computes": [
            {"task": "X", "branches": [{"kind": "lookup_external", "nl": "a"}]},
            {"task": "Y", "branches": [{"kind": "lookup_external", "nl": "b"}]},
        ],
    }
    # Both attempts return the same decomposed plan; both should fail validation.
    llm = _MockLLM([_wrap_json(decomposed), _wrap_json(decomposed)])
    ctx = _ctx(llm, max_depth=1)
    with pytest.raises(StepFailed) as excinfo:
        planner.plan("multi-part question", ctx)
    assert "depth" in str(excinfo.value).lower()
    assert llm.calls == 2  # The planner retries once on validation failure.
