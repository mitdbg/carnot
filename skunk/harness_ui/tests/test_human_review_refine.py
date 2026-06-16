"""Natural-language refine path: the broker kicks off a background LLM revision of a
verify_extract review's extracted candidates from a reviewer's feedback, writes the result back
into the review's guidance, and flags the review while it runs. See human_work_broker.refine_review
/ _refine_and_record and task_registry.set_review_refining / set_review_candidates."""

from __future__ import annotations

import asyncio

import pytest

from skunk_server.domain import AnswerCandidate
from skunk_server.human_work_broker import HumanWorkBroker
from skunk_server.task_registry import TaskConflict, TaskRegistry

CLIENT_ID = "tester"


def _review_task(registry: TaskRegistry, candidates):
    """A READY task with one OPEN verify_extract review carrying `candidates` in its guidance."""
    task, _ = registry.create_task(1, "Q1", "prompt")
    attempt = registry.begin_attempt(task.task_id, "w1")
    registry.complete_attempt(
        task.task_id,
        attempt.attempt_id,
        AnswerCandidate(attempt_id=attempt.attempt_id, answer_text="LLM", reasoning=""),
    )
    review = registry.create_review(
        task.task_id,
        attempt.attempt_id,
        "verify_extract",
        "instr",
        None,
        ["Treasury Bulletin 1954-02 PDF page 17"],
        {"branch_id": 2, "candidates": candidates},
    )
    registry.acquire_review_lock(task.task_id, CLIENT_ID)
    return task, review


def test_refine_replaces_candidates_and_clears_flag() -> None:
    async def scenario():
        registry = TaskRegistry()
        registry.update_round(round_num=1, status="ACTIVE")
        refine_calls: list[tuple] = []

        async def refine_fn(candidates, feedback, source_docs):
            refine_calls.append((candidates, feedback, source_docs))
            # Shift every value back one year (the canonical off-by-one case); preserve _src.
            return [
                {**c, "value": c["value"] - 1, "_src": i}
                for i, c in enumerate(candidates)
            ]

        broker = HumanWorkBroker(registry, None, lambda: None, refine_fn=refine_fn)
        broker.start(asyncio.get_running_loop())
        task, review = _review_task(
            registry, [{"description": "1968", "value": 1969, "kind": "scalar"}]
        )

        edited = [{"description": "1968", "value": 1969, "kind": "scalar", "_src": 0}]
        broker.refine_review(review.review_id, "shift back one year", edited, CLIENT_ID)
        # Flag is set synchronously so the UI shows "Updating extraction…" immediately.
        assert registry.find_review(review.review_id)[1].refining is True

        await asyncio.sleep(0.05)  # let the scheduled background refine run

        _task, done = registry.find_review(review.review_id)
        assert done.refining is False  # cleared after completion
        assert done.guidance["candidates"] == [
            {"description": "1968", "value": 1968, "kind": "scalar", "_src": 0}
        ]
        # The reviewer's edited JSONs + feedback + source docs reach the refine fn.
        assert refine_calls[0][0] == edited
        assert refine_calls[0][1] == "shift back one year"
        assert refine_calls[0][2] == ["Treasury Bulletin 1954-02 PDF page 17"]

    asyncio.run(scenario())


def test_refine_stamps_src_when_model_omits_it() -> None:
    async def scenario():
        registry = TaskRegistry()

        async def refine_fn(candidates, feedback, source_docs):
            return [{"description": "x", "value": 1}, {"description": "y", "value": 2}]

        broker = HumanWorkBroker(registry, None, lambda: None, refine_fn=refine_fn)
        broker.start(asyncio.get_running_loop())
        task, review = _review_task(registry, [{"description": "x", "value": 9}])

        broker.refine_review(
            review.review_id, "fb", [{"description": "x", "value": 9, "_src": 0}], CLIENT_ID
        )
        await asyncio.sleep(0.05)

        cands = registry.find_review(review.review_id)[1].guidance["candidates"]
        assert [c["_src"] for c in cands] == [0, 1]  # positional fallback

    asyncio.run(scenario())


def test_refine_failure_keeps_candidates_and_clears_flag() -> None:
    async def scenario():
        registry = TaskRegistry()

        async def refine_fn(candidates, feedback, source_docs):
            raise RuntimeError("LLM blew up")

        broker = HumanWorkBroker(registry, None, lambda: None, refine_fn=refine_fn)
        broker.start(asyncio.get_running_loop())
        original = [{"description": "1968", "value": 1969, "kind": "scalar", "_src": 0}]
        task, review = _review_task(registry, list(original))

        broker.refine_review(review.review_id, "fb", original, CLIENT_ID)
        await asyncio.sleep(0.05)

        _task, done = registry.find_review(review.review_id)
        assert done.refining is False  # cleared even on failure
        assert done.guidance["candidates"] == original  # left intact

    asyncio.run(scenario())


def test_refine_on_resolved_review_raises() -> None:
    async def scenario():
        registry = TaskRegistry()

        async def refine_fn(candidates, feedback, source_docs):
            return candidates

        broker = HumanWorkBroker(registry, None, lambda: None, refine_fn=refine_fn)
        broker.start(asyncio.get_running_loop())
        task, review = _review_task(registry, [])
        registry.resolve_review(review.review_id, "", [], CLIENT_ID)  # now RESOLVED

        with pytest.raises(TaskConflict):
            broker.refine_review(review.review_id, "fb", [], CLIENT_ID)

    asyncio.run(scenario())


def test_concurrent_refine_rejected() -> None:
    async def scenario():
        registry = TaskRegistry()
        gate = asyncio.Event()

        async def refine_fn(candidates, feedback, source_docs):
            await gate.wait()  # hold the first refine open
            return candidates

        broker = HumanWorkBroker(registry, None, lambda: None, refine_fn=refine_fn)
        broker.start(asyncio.get_running_loop())
        task, review = _review_task(registry, [])

        broker.refine_review(review.review_id, "fb", [], CLIENT_ID)  # first refine in flight
        await asyncio.sleep(0)
        with pytest.raises(TaskConflict):
            broker.refine_review(review.review_id, "fb2", [], CLIENT_ID)  # rejected while refining
        gate.set()
        await asyncio.sleep(0.05)
        assert registry.find_review(review.review_id)[1].refining is False

    asyncio.run(scenario())


def test_refine_disabled_when_no_refine_fn() -> None:
    async def scenario():
        registry = TaskRegistry()
        broker = HumanWorkBroker(registry, None, lambda: None)  # refine_fn=None
        broker.start(asyncio.get_running_loop())
        task, review = _review_task(registry, [])
        with pytest.raises(TaskConflict):
            broker.refine_review(review.review_id, "fb", [], CLIENT_ID)

    asyncio.run(scenario())


def test_registry_refine_setters_open_only_and_bump_version() -> None:
    registry = TaskRegistry()
    task, review = _review_task(registry, [{"description": "x", "value": 1}])
    v0 = registry.get(task.task_id).version

    registry.set_review_refining(review.review_id, True)
    assert registry.find_review(review.review_id)[1].refining is True
    assert registry.get(task.task_id).version > v0

    registry.set_review_candidates(review.review_id, [{"description": "y", "value": 2, "_src": 0}])
    assert registry.find_review(review.review_id)[1].guidance["candidates"][0]["description"] == "y"

    # A resolved (non-OPEN) review ignores a late candidate write.
    registry.set_review_refining(review.review_id, False)
    registry.resolve_review(review.review_id, "", [], CLIENT_ID)
    registry.set_review_candidates(review.review_id, [{"description": "z", "value": 3}])
    assert registry.find_review(review.review_id)[1].guidance["candidates"][0]["description"] == "y"
