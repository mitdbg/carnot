"""Slim optimistic-review broker: registering an open review mid-run, resolving it into a
background recompute that supersedes the answer, best-effort resubmit, and round-close cancel."""

from __future__ import annotations

import asyncio
import json

from skunk_server.domain import AnswerCandidate, HumanReviewStatus, TaskStatus
from skunk_server.human_work_broker import HumanWorkBroker
from skunk_server.task_registry import TaskRegistry


def _ready_task(registry: TaskRegistry, recompute_state=None):
    task, _ = registry.create_task(1, "Q1", "prompt")
    # Simulate a completed optimistic attempt: an answer candidate + READY + a snapshot.
    attempt = registry.begin_attempt(task.task_id, "w1")
    registry.complete_attempt(
        task.task_id,
        attempt.attempt_id,
        AnswerCandidate(attempt_id=attempt.attempt_id, answer_text="LLM", reasoning=""),
    )
    if recompute_state is not None:
        registry.set_recompute_state(task.task_id, recompute_state)
    return task, attempt


def test_resolve_triggers_recompute_and_best_effort_submit() -> None:
    async def scenario():
        registry = TaskRegistry()
        registry.update_round(round_num=1, status="ACTIVE")
        submitted: list[str] = []
        recompute_calls: list[tuple] = []

        async def recompute_fn(state, overrides):
            recompute_calls.append((state, overrides))
            return "REVISED"

        async def submit_fn(task_id):
            submitted.append(task_id)

        broker = HumanWorkBroker(registry, recompute_fn, submit_fn, lambda: None)
        broker.start(asyncio.get_running_loop())

        task, attempt = _ready_task(
            registry,
            recompute_state={
                "question": "q",
                "order": [2],
                "entries_by_branch": {"2": []},
                "extra_entries": [],
                "explanations": [],
            },
        )
        rid = broker.register_review(
            task.task_id,
            attempt.attempt_id,
            "verify_extract",
            "instr",
            None,
            ["Treasury Bulletin 1954-02 PDF page 17"],
            {"branch_id": 2, "candidates": []},
        )
        assert rid is not None
        assert len(registry.get(task.task_id).open_reviews) == 1

        broker.resolve_review(
            rid, json.dumps([{"description": "d", "value": 43, "kind": "scalar"}]), []
        )
        # Let the scheduled background recompute run.
        await asyncio.sleep(0.05)

        revised = registry.get(task.task_id)
        assert revised.latest_candidate.answer_text == "REVISED"
        assert revised.latest_candidate.submission_type == "human-revised"
        assert revised.status == TaskStatus.READY
        assert submitted == [task.task_id]  # best-effort resubmit fired (round ACTIVE)
        assert recompute_calls[0][1] == {
            2: json.dumps([{"description": "d", "value": 43, "kind": "scalar"}])
        }
        assert revised.reviews[0].status == HumanReviewStatus.RESOLVED

    asyncio.run(scenario())


def test_accept_as_is_does_not_recompute() -> None:
    async def scenario():
        registry = TaskRegistry()
        registry.update_round(round_num=1, status="ACTIVE")
        recompute_calls: list = []

        async def recompute_fn(state, overrides):
            recompute_calls.append(overrides)
            return "REVISED"

        async def submit_fn(task_id):
            pass

        broker = HumanWorkBroker(registry, recompute_fn, submit_fn, lambda: None)
        broker.start(asyncio.get_running_loop())
        task, attempt = _ready_task(
            registry,
            recompute_state={
                "question": "q",
                "order": [],
                "entries_by_branch": {},
                "extra_entries": [],
                "explanations": [],
            },
        )
        rid = broker.register_review(
            task.task_id,
            attempt.attempt_id,
            "verify_extract",
            "i",
            None,
            [],
            {"branch_id": 0},
        )

        broker.resolve_review(rid, "", [])  # empty response = accept-as-is
        await asyncio.sleep(0.03)

        assert recompute_calls == []  # no recompute scheduled
        assert (
            registry.get(task.task_id).latest_candidate.answer_text == "LLM"
        )  # unchanged

    asyncio.run(scenario())


def test_cancel_all_closes_open_reviews() -> None:
    async def scenario():
        registry = TaskRegistry()
        broker = HumanWorkBroker(registry, None, None, lambda: None)  # type: ignore[arg-type]
        broker.start(asyncio.get_running_loop())
        task, attempt = _ready_task(registry)
        broker.register_review(
            task.task_id, attempt.attempt_id, "lookup", "i", None, [], {"branch_id": 0}
        )
        assert len(registry.get(task.task_id).open_reviews) == 1
        broker.cancel_all()
        assert registry.get(task.task_id).open_reviews == []

    asyncio.run(scenario())


def test_pool_injects_register_and_recompute_sink_hooks() -> None:
    """The worker pool detects a reasoner that accepts the optimistic-review kwargs and wires
    them to the broker/registry: a review opens mid-run and the snapshot is captured."""
    from cup_kit.agent_runtime import AgentAnswer

    from skunk_server.agent_worker_pool import AgentWorkerPool
    from skunk_server.task_queues import TaskQueues

    async def run():
        loop = asyncio.get_running_loop()
        registry = TaskRegistry()
        queues = TaskQueues()
        broker = HumanWorkBroker(registry, None, None, lambda: None)  # type: ignore[arg-type]
        broker.start(loop)

        def reasoner(
            _prompt,
            *,
            human_review_register=None,
            recompute_sink=None,
            trace_event_handler=None,
        ):
            assert human_review_register is not None and recompute_sink is not None
            human_review_register(
                "verify_extract",
                "instr",
                "q",
                ["doc"],
                {"branch_id": 7, "candidates": []},
            )
            recompute_sink(
                {
                    "question": "q",
                    "order": [7],
                    "entries_by_branch": {"7": []},
                    "extra_entries": [],
                    "explanations": [],
                }
            )
            return AgentAnswer("42", "r", [])

        pool = AgentWorkerPool(registry, queues, reasoner, 1, human_broker=broker)
        pool.start(loop, lambda *_: None, lambda *_: None)
        task, _ = registry.create_task(1, "Q1", "prompt")
        queues.enqueue_agent(task.task_id)
        try:
            for _ in range(100):
                if registry.get(task.task_id).status == TaskStatus.READY:
                    break
                await asyncio.sleep(0.02)
            t = registry.get(task.task_id)
            assert t.status == TaskStatus.READY
            assert len(t.reviews) == 1 and t.reviews[0].guidance["branch_id"] == 7
            assert t.recompute_state is not None and t.recompute_state["order"] == [7]
        finally:
            pool.stop()

    asyncio.run(run())
