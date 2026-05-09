"""Tests for LookupExternalSubagent — NL-based interface.

All tests make real Gemini calls (no mocks, no cache).
"""

from __future__ import annotations

import pytest

from skunk.dsl import OpNode, TypedValue
from skunk.subagents.base import HarnessContext, StepFailed
from skunk.subagents.lookup_external import LookupExternalSubagent


@pytest.fixture()
def agent():
    return LookupExternalSubagent()


@pytest.fixture()
def ctx(tmp_path):
    return HarnessContext(question="test", cache_dir=str(tmp_path))


def test_missing_nl_raises(agent, ctx):
    op = OpNode(op="lookup_external", args={})
    with pytest.raises(StepFailed):
        agent.run(op, None, ctx)


def test_wwii_end_year(agent, ctx):
    op = OpNode(op="lookup_external", args={"nl": "year that WWII ended"})
    result = agent.run(op, None, ctx)
    assert isinstance(result, TypedValue)
    assert int(result.value) == 1945


def test_korean_war_start_year(agent, ctx):
    op = OpNode(op="lookup_external", args={"nl": "year the Korean War started"})
    result = agent.run(op, None, ctx)
    assert isinstance(result, TypedValue)
    assert int(result.value) == 1950


def test_returns_typed_value(agent, ctx):
    """Any well-formed NL query returns a TypedValue with non-None value and set dtype."""
    op = OpNode(op="lookup_external", args={"nl": "year Germany invaded Poland"})
    result = agent.run(op, None, ctx)
    assert isinstance(result, TypedValue)
    assert result.value is not None
    assert result.dtype not in ("", "unknown")
