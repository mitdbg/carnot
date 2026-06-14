from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest  # type: ignore[import-not-found]

from cup_kit.agent_runtime import AgentAnswer
from skunk_server.agent_worker_pool import AgentWorkerPool
from skunk_server.domain import AnswerCandidate, TaskStatus
from skunk_server.submission_coordinator import DEADLINE_SUBMIT_LEAD_S, SubmissionCoordinator
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


def test_agent_worker_pool_writes_trace_events() -> None:
    # Each trace event is written straight to the task's file via the EventWriter, inline on the
    # worker thread (no batcher, no main-loop hop). The pool hands the writer the RAW event; seq
    # stamping + compaction are the FileSink/web's job, not the pool's.
    async def run() -> None:
        loop = asyncio.get_running_loop()
        registry = TaskRegistry()
        queues = TaskQueues()
        task, _ = registry.create_task(1, "trace", "Question")
        written: list[tuple[str, str, dict]] = []  # write_event delivers ONE event at a time

        def reasoner(_prompt: str, *, trace_event_handler=None):
            assert trace_event_handler is not None
            trace_event_handler({"message": "planner started", "kind": "plan", "op": "planner", "t": 0.1})
            trace_event_handler({"message": "step done", "kind": "step", "op": "planner", "t": 1.0})
            return AgentAnswer("42", "Reasoning from worker thread.", [])

        pool = AgentWorkerPool(registry, queues, reasoner, 1)
        pool.start(
            loop,
            lambda _task_id, _outcome: None,
            lambda task_id, attempt_id, event: written.append((task_id, attempt_id, event)),
        )
        queues.enqueue_agent(task.task_id)

        try:
            for _ in range(100):
                if task.status == TaskStatus.READY and len(written) == 2:
                    break
                await asyncio.sleep(0.02)
            assert task.status == TaskStatus.READY
            assert [t for (t, _a, _e) in written] == [task.task_id, task.task_id]
            assert [e["message"] for (_t, _a, e) in written] == ["planner started", "step done"]
            # the pool forwards the event RAW — no seq injected by the pool (that is FileSink's job).
            assert all("seq" not in e for (_t, _a, e) in written)
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


class _AcceptingAdapter:
    async def submit(self, _submission, _candidate):
        return SimpleNamespace(
            accepted=True,
            submission_id="cup-submission",
            tokens_remaining=2,
            score=SimpleNamespace(correct=False, points_awarded=0.0),
        )


async def _wait_for_submitted(task, ticks: int = 200) -> None:
    for _ in range(ticks):
        if task.status == TaskStatus.SUBMITTED:
            break
        await asyncio.sleep(0.01)


def test_completion_in_final_window_submits_immediately() -> None:
    registry = TaskRegistry()
    queues = TaskQueues()
    # Deadline already inside the final auto-submit window: a freshly-READY answer must submit
    # immediately rather than waiting for the (already-fired) sweep.
    registry.update_round(
        round_num=1,
        status="ACTIVE",
        ends_at=datetime.now(timezone.utc) + timedelta(seconds=DEADLINE_SUBMIT_LEAD_S - 2),
    )
    task = _ready_task(registry, "final-window")

    async def run() -> None:
        coordinator = SubmissionCoordinator(registry, queues, _AcceptingAdapter(), lambda: None)  # type: ignore[arg-type]
        coordinator.handle_agent_completion(task.task_id, "ready")
        await _wait_for_submitted(task)

    asyncio.run(run())

    assert task.status == TaskStatus.SUBMITTED
    assert task.submissions[-1].status == "ACCEPTED"
    assert task.cup_feedback[-1] == "Cup score: correct=False, points_awarded=0.0"


def test_completion_outside_final_window_defers_to_sweep() -> None:
    registry = TaskRegistry()
    queues = TaskQueues()
    # Deadline far in the future: a READY answer is NOT auto-submitted on completion; it waits
    # for the manual button or the pre-deadline sweep.
    registry.update_round(
        round_num=1,
        status="ACTIVE",
        ends_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    task = _ready_task(registry, "deferred")

    async def run() -> None:
        coordinator = SubmissionCoordinator(registry, queues, _AcceptingAdapter(), lambda: None)  # type: ignore[arg-type]
        coordinator.handle_agent_completion(task.task_id, "ready")
        await asyncio.sleep(0.1)

    asyncio.run(run())

    assert task.status == TaskStatus.READY
    assert not task.submissions


def test_deadline_sweep_submits_un_submitted_ready_task() -> None:
    registry = TaskRegistry()
    queues = TaskQueues()
    # Sweep fires DEADLINE_SUBMIT_LEAD_S before the deadline; set it just past the lead so the
    # scheduled sleep is tiny.
    registry.update_round(
        round_num=1,
        status="ACTIVE",
        ends_at=datetime.now(timezone.utc) + timedelta(seconds=DEADLINE_SUBMIT_LEAD_S + 0.1),
    )
    task = _ready_task(registry, "sweep")

    async def run() -> None:
        coordinator = SubmissionCoordinator(registry, queues, _AcceptingAdapter(), lambda: None)  # type: ignore[arg-type]
        coordinator.on_round_active(1)
        await _wait_for_submitted(task)

    asyncio.run(run())

    assert task.status == TaskStatus.SUBMITTED
    assert task.submissions[-1].status == "ACCEPTED"


def test_reasoning_payload_includes_branch_cards_and_code() -> None:
    events = [
        {
            "kind": "plan",
            "data": {
                "label": "initial",
                "branches": [
                    {"branch_id": 0, "kind": "retrieve", "key": "inflation", "period": "1954-02"},
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
