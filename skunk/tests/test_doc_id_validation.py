"""Unit tests for validating + correcting the doc_ids a SearchAgent returns.

Two surfaces:
  - `doc_ids_from_payload` — the lenient decoder (trim / drop-empty / dedup).
  - `SearchAgent.call` — the bounded correction loop wrapped around the agent run: it
    re-prompts the agent (resuming the same conversation, on a separate small budget) when a
    returned id names no real document, keeps the valid subset, and fails only when none are valid.

The correction loop is exercised with a fake whose *base-class* run replays a script (see
`_ScriptedRun`), so no LLM, chroma, or tools are constructed. Runs under pytest, or standalone:
`python3 tests/test_doc_id_validation.py`.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from skunk.common import ExecutionContext
from skunk.errors import StepFailed
from skunk.multi_turn_agent import MultiTurnAgent
from skunk.search_agent.search_agent import SearchAgent, doc_ids_from_payload


# ---- doc_ids_from_payload ----------------------------------------------------


def test_payload_trims_drops_empty_and_dedups():
    # Surrounding whitespace is stripped, blanks dropped, duplicates removed (first-seen order).
    assert doc_ids_from_payload({"doc_ids": ["  A::p1 ", "A::p1", "", "  ", "B"]}) == ["A::p1", "B"]


def test_payload_single_string_is_one_id():
    assert doc_ids_from_payload({"doc_ids": "X::p2"}) == ["X::p2"]


def test_payload_malformed_returns_empty():
    assert doc_ids_from_payload(None) == []
    assert doc_ids_from_payload({"nope": 1}) == []
    assert doc_ids_from_payload({"doc_ids": []}) == []
    assert doc_ids_from_payload("just a string") == []


# ---- the correction loop -----------------------------------------------------


class _ScriptedRun(MultiTurnAgent):
    """Replays a script of final-answer payloads in place of the real multi-turn loop, and
    records each call's kwargs so the tests can assert the correction turns resume the
    conversation on the tight budget.

    `SearchAgent.call()` — the method under test — reaches the loop via `super().call(...)`,
    which resolves to whatever follows `SearchAgent` in the MRO, so the script must be installed
    *below* SearchAgent rather than on the subclass. `_FakeSearchAgent` lists this mixin after
    SearchAgent to slot it in there."""

    async def call(self, ctx, user, *, resume=False, max_steps=None, **_):
        self.calls.append({"user": user, "resume": resume, "max_steps": max_steps})
        return self._script.pop(0) if self._script else {}


class _FakeSearchAgent(SearchAgent, _ScriptedRun):
    """The real `SearchAgent.call()` (validation + correction) over a scripted agent run,
    bypassing the heavy real __init__ (no chroma / tools)."""

    def __init__(self, script, document_map, correction_steps=3):
        self.document_map = document_map
        self.name = "search_agent"
        self.config = SimpleNamespace(doc_id_correction_steps=correction_steps)
        self._script = list(script)
        self.calls: list[dict] = []


def _ctx() -> ExecutionContext:
    # config is only read to build an LLMClient (skipped here — a mock client is passed), so a
    # bare stand-in suffices; the correction loop under test uses only ctx.emit / the fake agent.
    return ExecutionContext(question="q", config=SimpleNamespace(), document_map={}, llm_client=object())  # type: ignore[arg-type]


DOCMAP = {"A::p1": "text-a", "B::p2": "text-b"}


def test_all_valid_needs_no_correction():
    agent = _FakeSearchAgent([{"doc_ids": ["A::p1", "B::p2"]}], DOCMAP)
    payload, valid = asyncio.run(agent.call(_ctx(), "q"))
    assert valid == ["A::p1", "B::p2"]
    assert len(agent.calls) == 1  # main run only, no correction turn
    assert agent.calls[0]["resume"] is False


def test_bad_id_is_corrected_on_next_turn():
    # First the model emits a bare name; the correction turn fixes it.
    agent = _FakeSearchAgent([{"doc_ids": ["A"]}, {"doc_ids": ["A::p1"]}], DOCMAP)
    payload, valid = asyncio.run(agent.call(_ctx(), "q"))
    assert valid == ["A::p1"]
    assert len(agent.calls) == 2  # main + one correction
    # The correction turn resumes the conversation on the tight, separate budget.
    assert agent.calls[1]["resume"] is True
    assert agent.calls[1]["max_steps"] == 1


def test_persistent_bad_id_is_filtered_after_budget():
    # "B" is valid, "junk" never is; the model keeps re-emitting both. After the correction
    # budget is spent we keep the valid subset rather than failing.
    agent = _FakeSearchAgent(
        [{"doc_ids": ["B::p2", "junk"]}] * 6, DOCMAP, correction_steps=3
    )
    payload, valid = asyncio.run(agent.call(_ctx(), "q"))
    assert valid == ["B::p2"]
    assert len(agent.calls) == 4  # main + 3 correction attempts (all still had a bad id)


def test_all_bad_and_unfixable_raises():
    agent = _FakeSearchAgent([{"doc_ids": ["nope"]}] * 6, DOCMAP, correction_steps=3)
    try:
        asyncio.run(agent.call(_ctx(), "q"))
    except StepFailed as e:
        assert "well-formed" in e.reason
    else:
        raise AssertionError("expected StepFailed when no returned doc_id is valid")
    assert len(agent.calls) == 4  # main + 3 correction attempts, then give up


def test_correction_budget_is_configurable():
    agent = _FakeSearchAgent([{"doc_ids": ["nope"]}] * 10, DOCMAP, correction_steps=1)
    try:
        asyncio.run(agent.call(_ctx(), "q"))
    except StepFailed:
        pass
    assert len(agent.calls) == 2  # main + 1 correction only


if __name__ == "__main__":
    test_payload_trims_drops_empty_and_dedups()
    test_payload_single_string_is_one_id()
    test_payload_malformed_returns_empty()
    test_all_valid_needs_no_correction()
    test_bad_id_is_corrected_on_next_turn()
    test_persistent_bad_id_is_filtered_after_budget()
    test_all_bad_and_unfixable_raises()
    test_correction_budget_is_configurable()
    print("ok")
