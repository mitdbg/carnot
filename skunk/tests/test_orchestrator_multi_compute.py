"""Orchestrator end-to-end tests for the multi-compute path.

Exercises the chain walker with a mocked LLM that knows how to respond to
both _CODEGEN_SYSTEM (final compute) and _CODEGEN_SYSTEM_INTERMEDIATE
prompts, plus the critique prompt. Uses LookupBranch + a stubbed
lookup_external subagent so no Gemini calls and no PDF corpus are needed.
"""

from __future__ import annotations

import pytest

import skunk.subagents.compute as compute
import skunk.subagents.lookup_external as lookup_external
from skunk.common import HarnessContext, LLMResponse
from skunk.dsl import (
    AnnotatedValue,
    ComputeNode,
    LookupBranch,
    Plan,
    RetrieveBranch,
    parse,
    validate,
)
from skunk.orchestrator import execute


class _MockLLM:
    """Routes by system-prompt identity into codegen / intermediate / critique queues."""

    def __init__(
        self,
        codegen: list[str] | None = None,
        intermediate: list[str] | None = None,
        critique: list[str] | None = None,
        lookup_external: list[str] | None = None,
    ):
        self.codegen = list(codegen or [])
        self.intermediate = list(intermediate or [])
        self.critique = list(critique or [])
        self.lookup_external = list(lookup_external or [])

    def call(self, system: str, user: str, **kwargs) -> LLMResponse:
        def _wrap(text: str) -> LLMResponse:
            return LLMResponse(text=text, latency_s=0.0, input_tokens=None, output_tokens=None)

        if system is compute._CRITIQUE_SYSTEM:
            assert self.critique, f"unexpected critique call; user={user[:200]}"
            return _wrap(self.critique.pop(0))
        if system is compute._CODEGEN_SYSTEM:
            assert self.codegen, f"unexpected codegen call; user={user[:200]}"
            return _wrap(self.codegen.pop(0))
        if system is compute._CODEGEN_SYSTEM_INTERMEDIATE:
            assert self.intermediate, f"unexpected intermediate codegen call; user={user[:200]}"
            return _wrap(self.intermediate.pop(0))
        # lookup_external uses its own prompt — last queue.
        assert self.lookup_external, f"unexpected lookup call: system={system[:80]!r}"
        return _wrap(self.lookup_external.pop(0))


def _code(body: str) -> str:
    return f"CODE\n{body}"


def test_legacy_flat_plan_unchanged_behavior():
    """A single-compute Plan executes exactly the way the pre-change orchestrator did:
    one final compute call, FormattedString answer, no intermediate codegens."""
    llm = _MockLLM(
        lookup_external=['1945\nyear'],
        codegen=[_code('result = "1945"')],
        critique=["ACCEPT"],
    )
    plan = parse("lookup_external(nl='year WWII ended') --> compute()")
    assert validate(plan).ok
    ctx = HarnessContext(
        question="What year did WWII end?", llm_client=llm,
    )
    trace = execute(plan, ctx)
    assert not trace.failed, trace.failure_reason
    assert trace.answer == "1945"
    # Steps: lookup_external + final compute = 2.
    assert len(trace.steps) == 2
    assert trace.steps[-1].op == "compute"


def test_decomposed_two_intermediates_then_final():
    """Decomposed plan: two parallel intermediate computes feed a final aggregator.
    Each intermediate emits a scalar AnnotatedValue; the final concatenates and reports."""
    llm = _MockLLM(
        lookup_external=['1945\nyear', '1950\nyear'],
        intermediate=[
            # First intermediate: pull WWII year out of its prev[0]
            _code("result = prev[0].value\nresult_unit = 'year'\nresult_description = 'WWII end year'"),
            # Second intermediate: pull Korean War year
            _code("result = prev[0].value\nresult_unit = 'year'\nresult_description = 'Korean War start year'"),
        ],
        codegen=[
            # Final: subtract them and report the gap.
            _code(
                "a = next(e for e in prev if 'korean war' in e.description.lower()).value\n"
                "b = next(e for e in prev if 'wwii' in e.description.lower()).value\n"
                "result = str(a - b)"
            ),
        ],
        critique=["ACCEPT"],
    )
    plan = Plan(computes=[
        ComputeNode(
            branches=[LookupBranch(nl="year WWII ended")],
            task="Find the year WWII ended", final=False,
        ),
        ComputeNode(
            branches=[LookupBranch(nl="year Korean War started")],
            task="Find the year Korean War started", final=False,
        ),
        ComputeNode(branches=[], task="", final=True),
    ])
    assert validate(plan).ok

    ctx = HarnessContext(
        question="Years between WWII ending and Korean War starting.", llm_client=llm,
    )
    trace = execute(plan, ctx)
    assert not trace.failed, trace.failure_reason
    assert trace.answer == "5"

    # Step count: 2 lookup_external + 2 intermediate computes + 1 final compute = 5
    assert len(trace.steps) == 5
    ops = [s.op for s in trace.steps]
    assert ops.count("compute") == 3
    assert ops.count("lookup_external") == 2
    # Final compute step is the last one and has final=True in args
    assert trace.steps[-1].op == "compute"
    assert trace.steps[-1].args.get("final") is True


def test_intermediate_result_wrapped_with_kind_inference():
    """An intermediate that returns a dict (no explicit result_kind) is wrapped as kind=vector."""
    llm = _MockLLM(
        lookup_external=['1945\nyear', '1950\nyear'],
        intermediate=[
            _code("result = {'wwii_end': prev[0].value}"),
            _code("result = {'korean_start': prev[0].value}"),
        ],
        codegen=[
            _code(
                # Both intermediates produced kind=vector dicts; combine and report.
                "vals = []\n"
                "for e in prev:\n"
                "    if isinstance(e.value, dict):\n"
                "        vals.extend(e.value.values())\n"
                "result = str(max(vals) - min(vals))"
            ),
        ],
        critique=["ACCEPT"],
    )
    plan = Plan(computes=[
        ComputeNode(
            branches=[LookupBranch(nl="year WWII ended")],
            task="WWII end year as a dict", final=False,
        ),
        ComputeNode(
            branches=[LookupBranch(nl="year Korean War started")],
            task="Korean War start year as a dict", final=False,
        ),
        ComputeNode(branches=[], task="", final=True),
    ])
    ctx = HarnessContext(question="Gap between events.", llm_client=llm)
    trace = execute(plan, ctx)
    assert not trace.failed, trace.failure_reason
    assert trace.answer == "5"


def test_validate_rejects_decomposed_when_max_depth_one():
    plan = Plan(computes=[
        ComputeNode(
            branches=[LookupBranch(nl="x")],
            task="t1", final=False,
        ),
        ComputeNode(branches=[], task="", final=True),
    ])
    result = validate(plan, max_compute_depth=1)
    assert not result.ok
    assert any("depth" in e.lower() for e in result.errors)


def test_validate_accepts_legacy_when_max_depth_one():
    plan = parse("lookup_external(nl='x') --> compute()")
    assert validate(plan, max_compute_depth=1).ok
