"""Unit tests for compute.run() — self-critique loop, mocked LLM client.

Verifies the control flow specified in the Fix-3 plan:
  attempt 1: codegen → exec → self-critique
    ACCEPT  → ship attempt-1 result
    REVISE  → attempt 2 codegen with critique as prior, ship unconditionally

Edge cases: MISSING propagation, exec/parse retries within attempt 1, fallback
when attempt 2 produces nothing, malformed critique reply.
"""

from __future__ import annotations

import pytest

import skunk.subagents.compute as compute
from skunk.common import HarnessContext, LLMResponse
from skunk.dsl import AnnotatedValue, FormattedString, OpNode
from skunk.subagents.base import MissingData, StepFailed


# ---------------------------------------------------------------------------
# Mock LLM client
# ---------------------------------------------------------------------------

class _MockLLM:
    """Routes calls by system-prompt identity into separate codegen / critique queues.

    Production code passes the module-level constants `_CODEGEN_SYSTEM` /
    `_CRITIQUE_SYSTEM` directly, so identity is the cleanest discriminator —
    no coupling to prompt prose. Each test sets up the queues explicitly; an
    empty queue when called is a test-author bug surfaced loudly.
    """

    def __init__(self, codegen: list[str] | None = None, critique: list[str] | None = None):
        self.codegen = list(codegen or [])
        self.critique = list(critique or [])
        self.codegen_calls = 0
        self.critique_calls = 0

    def call(self, system: str, user: str, **kwargs) -> LLMResponse:
        def _wrap(text: str) -> LLMResponse:
            return LLMResponse(text=text, latency_s=0.0, input_tokens=None, output_tokens=None)

        if system is compute._CRITIQUE_SYSTEM:
            self.critique_calls += 1
            if not self.critique:
                raise AssertionError(f"unexpected critique call #{self.critique_calls}; user={user[:200]}")
            return _wrap(self.critique.pop(0))
        if system is compute._CODEGEN_SYSTEM:
            self.codegen_calls += 1
            if not self.codegen:
                raise AssertionError(f"unexpected codegen call #{self.codegen_calls}; user={user[:200]}")
            return _wrap(self.codegen.pop(0))
        raise AssertionError(f"unknown system prompt: {system[:80]!r}")


def _ctx(llm: _MockLLM, question: str = "test question") -> HarnessContext:
    return HarnessContext(question=question, llm_client=llm)


def _prev_scalar(value: float = 35532.0, unit: str = "usd_millions") -> list[AnnotatedValue]:
    return [AnnotatedValue(description="x", value=value, unit=unit, kind="scalar")]


def _op() -> OpNode:
    return OpNode(op="compute", args={})


def _code(body: str) -> str:
    return f"CODE\n{body}"


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------

def test_accept_path_ships_attempt_1():
    llm = _MockLLM(
        codegen=[_code('result = "35,532"')],
        critique=["ACCEPT"],
    )
    out = compute.run(_op(), _prev_scalar(), _ctx(llm))
    assert isinstance(out, FormattedString)
    assert out.text == "35,532"
    assert llm.codegen_calls == 1
    assert llm.critique_calls == 1


def test_revise_then_attempt_2_ships_unconditionally():
    llm = _MockLLM(
        codegen=[
            _code('result = "35,532"'),
            _code('result = "35,532 millions of nominal dollars"'),
        ],
        critique=["REVISE: needs unit suffix"],
    )
    out = compute.run(_op(), _prev_scalar(), _ctx(llm))
    assert out.text == "35,532 millions of nominal dollars"
    assert llm.codegen_calls == 2
    # Critically: critique runs once, NOT on attempt 2's output.
    assert llm.critique_calls == 1


# ---------------------------------------------------------------------------
# Fallback paths (attempt 2 fails to produce; ship attempt 1)
# ---------------------------------------------------------------------------

def test_revise_attempt_2_missing_falls_back_to_attempt_1():
    llm = _MockLLM(
        codegen=[
            _code('result = "35,532"'),
            "MISSING: changed mind",
        ],
        critique=["REVISE: looks off"],
    )
    out = compute.run(_op(), _prev_scalar(), _ctx(llm))
    # Attempt 2 reported MISSING, but we don't raise — we fall back.
    assert out.text == "35,532"
    assert llm.codegen_calls == 2
    assert llm.critique_calls == 1


def test_revise_attempt_2_exec_failure_falls_back_to_attempt_1():
    llm = _MockLLM(
        codegen=[
            _code('result = "35,532"'),
            _code('1 / 0'),  # exec raises ZeroDivisionError
        ],
        critique=["REVISE: looks off"],
    )
    out = compute.run(_op(), _prev_scalar(), _ctx(llm))
    assert out.text == "35,532"
    assert llm.codegen_calls == 2
    assert llm.critique_calls == 1


def test_revise_attempt_2_malformed_falls_back_to_attempt_1():
    llm = _MockLLM(
        codegen=[
            _code('result = "35,532"'),
            "neither code nor missing",
        ],
        critique=["REVISE: hmm"],
    )
    out = compute.run(_op(), _prev_scalar(), _ctx(llm))
    assert out.text == "35,532"


# ---------------------------------------------------------------------------
# MISSING + transient retries on attempt 1
# ---------------------------------------------------------------------------

def test_missing_on_attempt_1_raises_no_critique():
    llm = _MockLLM(
        codegen=["MISSING: no FX rate"],
        critique=[],
    )
    with pytest.raises(MissingData):
        compute.run(_op(), _prev_scalar(), _ctx(llm))
    assert llm.critique_calls == 0


def test_attempt_1_exec_failure_then_recovery_then_critique():
    """Codegen #1 produces broken code; the within-attempt-1 retry budget recovers.
    Critique runs once on the recovered result.
    """
    llm = _MockLLM(
        codegen=[
            _code('1 / 0'),                    # exec fails
            _code('result = "ok"'),            # recovery within retry budget
        ],
        critique=["ACCEPT"],
    )
    out = compute.run(_op(), _prev_scalar(), _ctx(llm))
    assert out.text == "ok"
    assert llm.codegen_calls == 2
    assert llm.critique_calls == 1


def test_attempt_1_exhausts_budget_raises_step_failed():
    """All 3 tries within attempt 1 (default compute_max_attempts=3) fail to exec.
    No fallback exists, so we raise StepFailed.
    """
    llm = _MockLLM(
        codegen=[_code('1 / 0'), _code('1 / 0'), _code('1 / 0')],
        critique=[],
    )
    with pytest.raises(StepFailed):
        compute.run(_op(), _prev_scalar(), _ctx(llm))
    assert llm.critique_calls == 0


# ---------------------------------------------------------------------------
# Critique reply parsing
# ---------------------------------------------------------------------------

def test_critique_malformed_reply_treated_as_revise():
    llm = _MockLLM(
        codegen=[
            _code('result = "35,532"'),
            _code('result = "35,532 (revised)"'),
        ],
        critique=["i dunno"],  # neither ACCEPT nor REVISE
    )
    out = compute.run(_op(), _prev_scalar(), _ctx(llm))
    # Treated as REVISE → attempt 2 ships.
    assert out.text == "35,532 (revised)"
    assert llm.codegen_calls == 2
    assert llm.critique_calls == 1


def test_critique_revise_wins_when_both_tokens_present():
    llm = _MockLLM(
        codegen=[
            _code('result = "35,532"'),
            _code('result = "35,532 (revised)"'),
        ],
        critique=["could ACCEPT but REVISE: needs unit"],
    )
    out = compute.run(_op(), _prev_scalar(), _ctx(llm))
    assert out.text == "35,532 (revised)"
