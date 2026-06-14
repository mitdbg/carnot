"""Unit tests for the agent step-budget bookkeeping (`MultiTurnAgent._run_loop`).

Drive the loop with a scripted `_llm_step` (no LLM, no tools): each scripted turn either
misfires (raises `ParseError`, like prose with no runnable block) or emits a final answer.
Asserts that misfires do NOT consume the `max_steps` budget, and that the `max_misfires`
cap still terminates a model that never progresses.

Runs under pytest if installed, or standalone: `python3 tests/test_multi_turn_steps.py`.
"""

from __future__ import annotations

import asyncio

from skunk.common import ExecutionContext
from skunk.config import SkunkConfig
from skunk.errors import ParseError, StepFailed
from skunk.multi_turn_agent import MultiTurnAgent, TextBlock, _parse_step, _StepOutput


class _ScriptedAgent(MultiTurnAgent):
    """Minimal agent whose `_llm_step` replays a script of behaviours. Each script item is
    either the string "misfire" (raise ParseError) or a dict (returned as a final answer)."""

    name = "scripted"
    briefing = "test"
    final_answer_doc = "a json object"

    def __init__(self, script, **kwargs):
        super().__init__([], **kwargs)
        self._script = list(script)
        self.llm_calls = 0  # total `_llm_step` invocations (loop + terminal turn)

    async def _llm_step(self, ctx, extra=None):
        self.llm_calls += 1
        item = self._script.pop(0) if self._script else "misfire"
        if item == "misfire":
            raise ParseError(raw="prose, no fence", detail="no fenced block")
        return _StepOutput(result=item, is_final=True, raw="```json\n{}\n```")

    async def _drive(self, ctx):
        # Bypass call()'s system-prompt assembly / executor build — neither the misfire nor
        # the final-answer path touches them — and exercise the shared loop directly.
        self.messages = [{"role": "user", "blocks": [TextBlock("q")]}]
        return await self._run_loop(ctx, self.max_steps)


def _ctx() -> ExecutionContext:
    return ExecutionContext(question="q", config=SkunkConfig(), llm_client=object())


def test_misfires_do_not_consume_step_budget():
    # 4 misfires (> max_steps) then a final answer. Old behaviour charged each misfire a
    # step and would never reach the answer; now misfires are free, so the answer commits.
    agent = _ScriptedAgent(
        ["misfire", "misfire", "misfire", "misfire", {"ok": True}],
        max_steps=2,
        max_misfires=10,
    )
    result = asyncio.run(agent._drive(_ctx()))
    assert result == {"ok": True}
    assert agent.llm_calls == 5  # 4 misfires + 1 final, all within budget


def test_misfire_cap_terminates_a_stuck_agent():
    # A model that ALWAYS misfires must still terminate: total attempts are capped at
    # max_steps + max_misfires, after which the (also-misfiring) terminal turn raises.
    agent = _ScriptedAgent(["misfire"] * 100, max_steps=2, max_misfires=3)
    try:
        asyncio.run(agent._drive(_ctx()))
    except StepFailed as e:
        assert "max steps" in e.reason
    else:
        raise AssertionError("expected StepFailed when the agent never progresses")
    # 2 + 3 = 5 loop attempts, then 1 forced terminal turn.
    assert agent.llm_calls == 6


def test_clean_run_uses_step_budget_normally():
    # No misfires: a single final-answer turn commits immediately.
    agent = _ScriptedAgent([{"v": 1}], max_steps=2, max_misfires=3)
    assert asyncio.run(agent._drive(_ctx())) == {"v": 1}
    assert agent.llm_calls == 1


def test_call_resume_appends_to_trajectory():
    # call(resume=True) continues the SAME trajectory (not reset) — the reviewer-feedback seam.
    # The resumed run misfires 3x (> max_steps=2) before answering, only possible because the
    # budget is fresh AND misfires are free.
    agent = _ScriptedAgent(
        [{"first": True}, "misfire", "misfire", "misfire", {"second": True}],
        max_steps=2,
        max_misfires=10,
    )
    ctx = _ctx()
    assert asyncio.run(agent.call(ctx, "q1")) == {"first": True}
    msgs_before = len(agent.messages)
    assert asyncio.run(agent.call(ctx, "more: keep searching", resume=True)) == {"second": True}
    # Trajectory preserved (grew, not reset) and the feedback landed as a user turn.
    assert len(agent.messages) > msgs_before
    assert any(
        b.text == "more: keep searching"
        for m in agent.messages
        for b in m.get("blocks", [])
        if isinstance(b, TextBlock)
    )


def test_call_without_resume_resets_trajectory():
    agent = _ScriptedAgent([{"a": 1}, {"b": 2}], max_steps=4)
    ctx = _ctx()
    asyncio.run(agent.call(ctx, "q1"))
    asyncio.run(agent.call(ctx, "q2"))  # no resume → fresh trajectory
    # Only the second question's turns remain (reset, not appended).
    assert any(
        b.text == "q2"
        for m in agent.messages
        for b in m.get("blocks", [])
        if isinstance(b, TextBlock)
    )
    assert not any(
        b.text == "q1"
        for m in agent.messages
        for b in m.get("blocks", [])
        if isinstance(b, TextBlock)
    )


# ---- single-tool-call parser (`_parse_step`) -------------------------------------


def test_parse_step_single_python_block():
    out = _parse_step("```python\nsearch_corpus('x')\n```", None)
    assert out.code == "search_corpus('x')"
    assert out.is_final is False
    assert out.notice is None


def test_parse_step_lone_json_is_final():
    out = _parse_step('```json\n{"page_keys": ["a"]}\n```', None)
    assert out.is_final is True
    assert out.result == {"page_keys": ["a"]}
    assert out.code is None


def test_parse_step_runs_first_block_and_notices_on_over_emission():
    # Two python blocks: run the FIRST, and flag the over-emission.
    text = "```python\nfirst()\n```\n```python\nsecond()\n```"
    out = _parse_step(text, None)
    assert out.code == "first()"
    assert out.notice is not None and "only the first" in out.notice
    # A python block alongside a json block also runs the python + notices.
    mixed = _parse_step("```python\nact()\n```\n```json\n{}\n```", None)
    assert mixed.code == "act()" and mixed.is_final is False and mixed.notice is not None


def test_parse_step_no_block_is_a_misfire():
    try:
        _parse_step("just prose, no fence", None)
    except ParseError:
        pass
    else:
        raise AssertionError("expected ParseError on a reply with no fenced block")


if __name__ == "__main__":
    test_misfires_do_not_consume_step_budget()
    test_misfire_cap_terminates_a_stuck_agent()
    test_clean_run_uses_step_budget_normally()
    test_call_resume_appends_to_trajectory()
    test_call_without_resume_resets_trajectory()
    test_parse_step_single_python_block()
    test_parse_step_lone_json_is_final()
    test_parse_step_runs_first_block_and_notices_on_over_emission()
    test_parse_step_no_block_is_a_misfire()
    print("ok")
