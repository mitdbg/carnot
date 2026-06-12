from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import pytest  # type: ignore[import-not-found]

from cup_kit.agent_runtime import AgentAnswer
from skunk_server.agent_worker_pool import AgentWorkerPool
from skunk_server.domain import AnswerCandidate, TaskStatus
from skunk_server.submission_coordinator import SubmissionCoordinator
from skunk_server.task_queues import TaskQueues
from skunk_server.task_registry import TaskRegistry
from skunk_reasoner import (
    SKUNK_ROOT,
    _set_default_env,
    _source_docs_from_events,
    _structured_reasoning_payload,
)


def _ready_task(registry: TaskRegistry, question_id: str = "q1"):
    task, _ = registry.create_task(1, question_id, "What is the answer?")
    attempt = registry.begin_attempt(task.task_id, "agent")
    assert attempt is not None
    candidate = AnswerCandidate(
        attempt_id=attempt.attempt_id,
        answer_text="42",
        reasoning="The test reasoner returned the expected value.",
    )
    assert registry.complete_attempt(task.task_id, attempt.attempt_id, candidate)
    return task


def test_reasoner_defaults_page_index_to_repo_cache(monkeypatch) -> None:
    monkeypatch.delenv("SKUNK_PAGE_INDEX_DIR", raising=False)

    _set_default_env()

    assert os.environ["SKUNK_PAGE_INDEX_DIR"] == str(SKUNK_ROOT / "cache/build_v3")


def test_agent_worker_pool_success_and_failure() -> None:
    async def run() -> None:
        loop = asyncio.get_running_loop()

        success_registry = TaskRegistry()
        success_queues = TaskQueues()
        success_task, _ = success_registry.create_task(1, "success", "Question")
        success_events: list[tuple[str, str]] = []
        success_pool = AgentWorkerPool(
            success_registry,
            success_queues,
            lambda _prompt: AgentAnswer("42", "Reasoning from worker thread.", []),
            1,
        )
        success_pool.start(
            loop,
            lambda task_id, outcome: success_events.append((task_id, outcome)),
            lambda *_: None,
        )
        success_queues.enqueue_agent(success_task.task_id)
        try:
            for _ in range(100):
                if success_task.status == TaskStatus.READY:
                    break
                await asyncio.sleep(0.02)
            assert success_task.status == TaskStatus.READY
            assert success_task.latest_candidate is not None
            assert success_events == [
                (success_task.task_id, "processing"),
                (success_task.task_id, "ready"),
            ]
        finally:
            success_pool.stop()

        failure_registry = TaskRegistry()
        failure_queues = TaskQueues()
        failure_task, _ = failure_registry.create_task(1, "failure", "Question")

        def fail(_prompt: str):
            raise RuntimeError("simulated failure")

        failure_pool = AgentWorkerPool(failure_registry, failure_queues, fail, 1)
        failure_pool.start(loop, lambda _task_id, _outcome: None, lambda *_: None)
        failure_queues.enqueue_agent(failure_task.task_id)
        try:
            for _ in range(100):
                if failure_task.status == TaskStatus.FAILED:
                    break
                await asyncio.sleep(0.02)
            assert failure_task.status == TaskStatus.FAILED
            assert failure_task.failures[-1].error_message == "simulated failure"
        finally:
            failure_pool.stop()

    asyncio.run(run())


def test_round_close_frees_worker_for_next_round() -> None:
    # A round closing must cancel its still-running reasoners so the (single) worker is
    # freed for the next round, instead of staying blocked on un-cancellable in-flight work.
    async def run() -> None:
        loop = asyncio.get_running_loop()
        registry = TaskRegistry()
        queues = TaskQueues()

        async def reasoner(prompt: str):
            if "block" in prompt:
                await asyncio.sleep(3600)  # round-1 question: never returns on its own
            return AgentAnswer("42", "Reasoning from the next round's worker.", [])

        pool = AgentWorkerPool(registry, queues, reasoner, 1)  # one worker on purpose
        pool.start(loop, lambda _task_id, _outcome: None, lambda *_: None)
        r1, _ = registry.create_task(1, "blocker", "block this round-1 question")
        r2, _ = registry.create_task(2, "fast", "answer this round-2 question")
        try:
            queues.enqueue_agent(r1.task_id)
            for _ in range(200):
                if r1.status == TaskStatus.PROCESSING:
                    break
                await asyncio.sleep(0.02)
            assert r1.status == TaskStatus.PROCESSING

            queues.enqueue_agent(r2.task_id)
            await asyncio.sleep(0.1)
            assert r2.status == TaskStatus.QUEUED  # worker still blocked on round 1

            registry.close_round(1, "CLOSED")  # cancels round 1's reasoner, frees the worker

            for _ in range(200):
                if r2.status == TaskStatus.READY:
                    break
                await asyncio.sleep(0.02)
            assert r1.status == TaskStatus.CANCELLED
            assert r2.status == TaskStatus.READY  # the freed worker ran round 2
            assert r2.latest_candidate is not None
        finally:
            pool.stop()

    asyncio.run(run())


def test_agent_worker_pool_streams_trace_events() -> None:
    # Trace events are appended to the attempt (stamped with a per-task seq) and published to
    # the task's event-stream subscribers — not routed through the completion callback.
    async def run() -> None:
        loop = asyncio.get_running_loop()
        registry = TaskRegistry()
        queues = TaskQueues()
        task, _ = registry.create_task(1, "trace", "Question")
        published: list[tuple[str, str, list[dict]]] = []  # publish_event delivers batches

        def reasoner(_prompt: str, *, trace_event_handler=None):
            assert trace_event_handler is not None
            trace_event_handler({"message": "planner started", "kind": "plan", "op": "planner", "t": 0.1})
            trace_event_handler({"message": "step done", "kind": "step", "op": "planner", "t": 1.0})
            return AgentAnswer("42", "Reasoning from worker thread.", [])

        pool = AgentWorkerPool(registry, queues, reasoner, 1)
        pool.start(
            loop,
            lambda _task_id, _outcome: None,
            lambda task_id, attempt_id, events: published.append((task_id, attempt_id, events)),
        )
        queues.enqueue_agent(task.task_id)

        def flat() -> list[dict]:
            return [event for (_t, _a, events) in published for event in events]

        try:
            for _ in range(100):
                if task.status == TaskStatus.READY and len(flat()) == 2:
                    break
                await asyncio.sleep(0.02)
            assert task.status == TaskStatus.READY
            assert [event["message"] for event in flat()] == ["planner started", "step done"]
            assert [event["seq"] for event in flat()] == [0, 1]
            # The same events back-fill from the registry with matching seq.
            backfill, next_seq = registry.snapshot_task_events(task.task_id)
            assert next_seq == 2
            assert [(seq, event["message"]) for (seq, _a, event) in backfill] == [
                (0, "planner started"),
                (1, "step done"),
            ]
        finally:
            pool.stop()

    asyncio.run(run())


def test_cancelled_run_dumps_partial_trace(tmp_path, monkeypatch) -> None:
    # A run cancelled at round close should still write its (partial) trace, marked
    # "cancelled", and must re-raise the cancellation rather than swallow it.
    import json

    import skunk
    import skunk_reasoner as sr

    class _FakeCtx:
        def __init__(self) -> None:
            self.events = [{"message": "plan label=initial branches=1", "kind": "plan"}]

        def close(self) -> None:
            pass

    class _FakeOrch:
        def __init__(self, *_args, **_kwargs) -> None:
            self.ctx = _FakeCtx()

        async def execute(self):
            raise asyncio.CancelledError

    monkeypatch.setattr(skunk, "Orchestrator", _FakeOrch)
    monkeypatch.setenv("SKUNK_CONSOLE_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("SKUNK_PROMPT_OVERRIDES", str(tmp_path / "nonexistent.yaml"))

    async def run() -> None:
        with pytest.raises(asyncio.CancelledError):
            await sr.SkunkReasoner().execute("a cancelled question about defense outlays")

    asyncio.run(run())

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["status"] == "cancelled"
    assert payload["events"]  # partial trace was captured


def test_auto_submit_records_immediate_score_feedback() -> None:
    registry = TaskRegistry()
    queues = TaskQueues()
    task = _ready_task(registry, "auto")

    class FakeAdapter:
        async def submit(self, _submission, _candidate):
            return SimpleNamespace(
                accepted=True,
                submission_id="cup-submission",
                tokens_remaining=2,
                score=SimpleNamespace(correct=False, points_awarded=0.0),
            )

    async def run() -> None:
        coordinator = SubmissionCoordinator(
            registry,
            queues,
            FakeAdapter(),  # type: ignore[arg-type]
            lambda: None,
            auto_submit=True,
        )
        coordinator.handle_agent_completion(task.task_id, "ready")
        for _ in range(100):
            if task.status == TaskStatus.SUBMITTED:
                break
            await asyncio.sleep(0.01)

    asyncio.run(run())

    assert task.status == TaskStatus.SUBMITTED
    assert task.submissions[-1].status == "ACCEPTED"
    assert task.cup_feedback[-1] == "Cup score: correct=False, points_awarded=0.0"


def test_reasoning_payload_includes_branch_cards_and_code() -> None:
    events = [
        {
            "kind": "plan",
            "data": {
                "label": "initial",
                "branches": [
                    {"branch_id": 0, "kind": "retrieve", "key": "inflation", "period": "1954-02", "as_of": "1954-02"},
                    {"branch_id": 1, "kind": "lookup_external", "target": "cpi", "src": "fred"},
                ],
            },
        },
        {
            "kind": "step",
            "op": "extract",
            "data": {"branch_id": 0, "summary": {"type": "values", "values": [{"description": "inflation"}]}},
        },
        {
            "kind": "step",
            "op": "lookup_external",
            "data": {"branch_id": 1, "summary": {"type": "scalar", "value": 3.1}},
        },
        {
            "kind": "step",
            "op": "compute",
            "data": {"attempt": 1, "code": "result = '42'"},
        },
        {
            "kind": "observation",
            "message": "human_directed_retrieval",
            "data": {"directives": [{"branch_id": 0, "bulletins": ["1954-02"]}]},
        },
    ]

    payload = _structured_reasoning_payload(events, ["Treasury Bulletin 1954-02 PDF page 4"])

    assert payload["summary"]["branch_count"] == 2
    assert payload["summary"]["human_directed_retrieval"] is True
    assert payload["branches"][0]["searched"]["key"] == "inflation"
    assert payload["branches"][1]["searched"]["target"] == "cpi"
    assert payload["python_code"] == "result = '42'"


def test_source_docs_are_read_from_structured_step_provenance() -> None:
    events = [
        {
            "kind": "step",
            "op": "extract",
            "data": {
                "summary": {
                    "type": "values",
                    "values": [
                        {
                            "description": "reported value",
                            "bulletin": "1954-02",
                            "pages": [4, 5],
                        }
                    ],
                }
            },
        }
    ]

    assert _source_docs_from_events(events) == [
        "Treasury Bulletin 1954-02 PDF page 4",
        "Treasury Bulletin 1954-02 PDF page 5",
    ]
