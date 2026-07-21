"""Unit tests for the qatfd SearchAgent system wiring.

Focus: `SearchAgentSystem.retrieve()` must return only the doc_ids that name real documents,
because that list both scores recall and (in retrieve mode) selects the text handed to the
answerer. The real per-id validation loop lives in skunk and is tested there; here we confirm
the system runs the returned ids through it and threads answer-mode through unchanged.

The agent is faked (its `call()` replays a script), so no LLM / chroma / tools are built.
Runs under pytest, or standalone: `python3 tests/test_systems.py`.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from skunk.common import ExecutionContext
from skunk.config import PipelineConfig
from skunk.search_agent.search_agent import SearchAgent

from qatfd.systems.search_agent import SearchAgentSystem
from qatfd.types import Question

DOCMAP = {"A::p1": "text-a", "B::p2": "text-b"}


class _FakeAgent(SearchAgent):
    """SearchAgent with the real `run_with_validated_doc_ids` but a scripted `call()`."""

    def __init__(self, script):
        self.document_map = DOCMAP
        self.name = "search_agent"
        self.config = SimpleNamespace(doc_id_correction_steps=3)
        self._script = list(script)

    async def call(self, ctx, user, *, resume=False, max_steps=None, **_):
        return self._script.pop(0) if self._script else {}


class _StubSystem(SearchAgentSystem):
    def __init__(self, agent, agent_mode="retrieve"):
        self.config = SimpleNamespace(agent_mode=agent_mode)
        self._agent = agent

    def _build_agent(self, ctx, resources):
        return self._agent


def _ctx() -> ExecutionContext:
    return ExecutionContext(question="q", config=PipelineConfig(), llm_client=object())


def _q() -> Question:
    return Question(qid="q1", text="a question", gold="", gold_docs=[])


def test_retrieve_keeps_only_well_formed_ids():
    # The agent emits one real id and one bare name; the bare name triggers a correction turn
    # (any bad id does), on which the agent re-emits the clean list. retrieve() returns that.
    system = _StubSystem(_FakeAgent([
        {"doc_ids": ["A::p1", "NOT_A_DOC"]},
        {"doc_ids": ["A::p1"]},
    ]))
    r = asyncio.run(system.retrieve(_q(), resources=None, ctx=_ctx()))
    assert r.doc_ids == ["A::p1"]
    assert r.direct_answer is None


def test_retrieve_filters_when_correction_does_not_fix_everything():
    # The agent never drops the bad name; after the budget we keep the well-formed subset.
    system = _StubSystem(_FakeAgent([{"doc_ids": ["A::p1", "NOT_A_DOC"]}] * 6))
    r = asyncio.run(system.retrieve(_q(), resources=None, ctx=_ctx()))
    assert r.doc_ids == ["A::p1"]


def test_retrieve_answer_mode_threads_direct_answer():
    system = _StubSystem(
        _FakeAgent([
            {"answer": "42", "doc_ids": ["B::p2", "NOT_A_DOC"]},
            {"answer": "42", "doc_ids": ["B::p2"]},
        ]),
        agent_mode="answer",
    )
    r = asyncio.run(system.retrieve(_q(), resources=None, ctx=_ctx()))
    assert r.doc_ids == ["B::p2"]
    assert r.direct_answer == "42"


if __name__ == "__main__":
    test_retrieve_keeps_only_well_formed_ids()
    test_retrieve_filters_when_correction_does_not_fix_everything()
    test_retrieve_answer_mode_threads_direct_answer()
    print("ok")
