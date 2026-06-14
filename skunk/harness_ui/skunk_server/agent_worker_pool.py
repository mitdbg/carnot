"""Server-owned worker threads that execute reasoning attempts."""

from __future__ import annotations

import asyncio
import inspect
import logging
import queue
import threading
import traceback
from collections.abc import Callable
from typing import Any

from cup_kit.agent_runtime import AgentAnswer
from cup_kit.protocol import MAX_SOURCE_DOC_REF_CHARS

from skunk_server.domain import AnswerCandidate, Attempt, FailureRecord
from skunk_server.task_queues import TaskQueues
from skunk_server.task_registry import TaskRegistry

logger = logging.getLogger(__name__)

Reasoner = Callable[..., AgentAnswer | dict[str, Any] | Any]
CompletionCallback = Callable[[str, str], None]
# Writes ONE raw trace event straight to the task's file (FileSink.write_event). Called inline
# on the reasoner worker thread — no batcher, no periodic flusher, no main-loop hop. The agent's
# entire per-event job is: stamp a cheap seq + append one line. The web process interprets.
EventWriter = Callable[[str, str, dict], None]


class AgentWorkerPool:
    def __init__(
        self,
        registry: TaskRegistry,
        queues: TaskQueues,
        reasoner: Reasoner,
        max_workers: int,
        human_broker: Any | None = None,
        human_blocking: bool = False,
    ) -> None:
        self._registry = registry
        self._queues = queues
        self._reasoner = reasoner
        self._max_workers = max(1, max_workers)
        self._human_broker = human_broker
        # Blocking transport: wire `human_intervention_handler` (branch suspends on the human via
        # the web UI) instead of the optimistic `human_review_register`. See HumanWorkBroker.
        self._human_blocking = human_blocking
        try:
            parameters = inspect.signature(reasoner).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        self._reasoner_accepts_trace = (
            "trace_event_handler" in parameters or accepts_kwargs
        )
        # Optimistic human-review hooks: register opens reviews mid-run; the sink captures the
        # recompute snapshot after a successful compute. Injected only if the reasoner accepts
        # them (so a bare reasoner still runs).
        self._reasoner_accepts_review = (
            "human_review_register" in parameters or accepts_kwargs
        )
        self._reasoner_accepts_intervention = (
            "human_intervention_handler" in parameters or accepts_kwargs
        )
        self._reasoner_accepts_recompute_sink = (
            "recompute_sink" in parameters or accepts_kwargs
        )
        self._reasoner_accepts_review_discard = (
            "human_reviews_discard" in parameters or accepts_kwargs
        )
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._on_completion: CompletionCallback | None = None
        self._write_event: EventWriter | None = None
        # In-flight reasoner runs by task_id → (round_num, worker_loop, future), so a
        # closing round can cancel its still-running reasoners and free the workers. Each
        # worker runs its reasoner on its OWN event loop (see `_worker_loop`), so the
        # future and the loop that owns it are both recorded for cross-thread cancellation.
        self._inflight: dict[
            str, tuple[int, asyncio.AbstractEventLoop, asyncio.Future]
        ] = {}
        self._inflight_lock = threading.Lock()

    def start(
        self,
        loop: asyncio.AbstractEventLoop,
        on_completion: CompletionCallback,
        write_event: EventWriter,
    ) -> None:
        self._loop = loop
        self._on_completion = on_completion
        self._write_event = write_event
        self._registry.set_round_close_callback(self.cancel_round)
        for index in range(self._max_workers):
            thread = threading.Thread(
                target=self._worker_loop,
                name=f"skunk-agent-{index + 1}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def cancel_round(self, round_num: int) -> int:
        """Cancel every still-running reasoner for `round_num`. The cancel is scheduled on
        each reasoner's OWN worker loop, which makes that worker's blocking
        `run_until_complete` raise `CancelledError`, freeing the worker for the next round
        even though the underlying agent could not be interrupted in place. Safe to call
        repeatedly (a closing round emits CLOSED then RESULTS)."""
        with self._inflight_lock:
            targets = [
                (loop, future)
                for (rn, loop, future) in self._inflight.values()
                if rn == round_num
            ]
        cancelled = 0
        for loop, future in targets:
            # `future.cancel()` must run on the loop that owns the future.
            loop.call_soon_threadsafe(future.cancel)
            cancelled += 1
        if cancelled:
            logger.info(
                "round %s closed: cancelled %d in-flight agent run(s) to free workers",
                round_num,
                cancelled,
            )
        return cancelled

    def stop(self) -> None:
        self._stop.set()
        for _ in self._threads:
            try:
                self._queues.agent.put_nowait("")
            except queue.Full:
                break
        for thread in self._threads:
            thread.join(timeout=5)
        self._threads.clear()

    def _worker_loop(self) -> None:
        worker_id = threading.current_thread().name
        # Each worker runs its reasoner on its OWN event loop (mirroring how
        # `eval/eval_e2e.py` does `asyncio.run` per worker thread). This gives every
        # in-flight question an independent loop AND its own default ThreadPoolExecutor for
        # `asyncio.to_thread` tool offloads, instead of all workers contending on the
        # single uvicorn loop. The main loop (`self._loop`) is still used, via
        # `call_soon_threadsafe`, for the completion/trace callbacks that drive the UI.
        worker_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(worker_loop)
        try:
            self._run_worker(worker_id, worker_loop)
        finally:
            worker_loop.close()

    def _run_worker(
        self, worker_id: str, worker_loop: asyncio.AbstractEventLoop
    ) -> None:
        while not self._stop.is_set():
            try:
                task_id = self._queues.agent.get(timeout=0.5)
            except queue.Empty:
                continue
            if not task_id:
                self._queues.agent.task_done()
                continue
            attempt = self._registry.begin_attempt(task_id, worker_id)
            if attempt is None:
                self._queues.agent.task_done()
                continue
            if self._loop is not None and self._on_completion is not None:
                self._loop.call_soon_threadsafe(
                    self._on_completion,
                    task_id,
                    "processing",
                )
            outcome = "failed"
            try:
                task = self._registry.get(task_id)
                if task is None:
                    continue
                prompt = self._reasoner_prompt(task.prompt, attempt)
                reasoner_kwargs: dict[str, Any] = {}
                if self._reasoner_accepts_trace and self._write_event is not None:
                    # Write each orchestrator event straight to the task's file, inline on this
                    # worker thread — the agent stamps a cheap seq and appends one line; the web
                    # process scans + compacts. No batcher, no main-loop hop.
                    write = self._write_event
                    reasoner_kwargs["trace_event_handler"] = (
                        lambda ev, _t=task_id, _a=attempt.attempt_id: write(_t, _a, ev)
                    )
                if self._human_broker is not None:
                    broker: Any = self._human_broker
                    aid = attempt.attempt_id
                    # The replan barrier ALWAYS blocks on the human via the intervention handler:
                    # a replan means compute produced no answer to submit, so optimism is never
                    # valid there. Wire it regardless of `human_blocking` (in blocking mode it ALSO
                    # serves the inline extract-verify suspend). Leaving `human_review_register`
                    # unset under blocking is what makes the orchestrator pick the blocking
                    # transport for the verify/lookup seams too.
                    if self._reasoner_accepts_intervention:
                        reasoner_kwargs["human_intervention_handler"] = (
                            lambda task, instr, q, docs, guid, _t=task_id, _a=aid: (
                                broker.await_intervention(_t, _a, task, instr, q, docs, guid)
                            )
                        )
                    if not self._human_blocking and self._reasoner_accepts_review:
                        # OPTIMISTIC (default) extract/lookup verify: open a review mid-run
                        # (fire-and-forget; keep going). The replan path stays blocking via the
                        # handler above; on a replan the orchestrator discards these via the hook
                        # below, since the replan supersedes the branches they belong to.
                        reasoner_kwargs["human_review_register"] = (
                            lambda kind, instr, ctx_q, docs, guid, _t=task_id, _a=aid: (
                                broker.register_review(_t, _a, kind, instr, ctx_q, docs, guid)
                            )
                        )
                        if self._reasoner_accepts_review_discard:
                            reasoner_kwargs["human_reviews_discard"] = (
                                lambda _t=task_id: broker.discard_reviews(_t)
                            )
                if self._reasoner_accepts_recompute_sink:
                    # Capture the recompute snapshot so a resolved review can revise the answer.
                    reasoner_kwargs["recompute_sink"] = lambda state, _t=task_id: (
                        self._registry.set_recompute_state(_t, state)
                    )
                raw = self._reasoner(prompt, **reasoner_kwargs)
                if inspect.isawaitable(raw):
                    future = asyncio.ensure_future(raw, loop=worker_loop)
                    with self._inflight_lock:
                        self._inflight[task_id] = (task.round_num, worker_loop, future)
                    try:
                        raw = worker_loop.run_until_complete(future)
                    finally:
                        with self._inflight_lock:
                            self._inflight.pop(task_id, None)
                answer, reasoning, source_docs = self._normalize_answer(raw)
                candidate = AnswerCandidate(
                    attempt_id=attempt.attempt_id,
                    answer_text=answer,
                    reasoning=reasoning,
                    source_docs=source_docs,
                )
                if self._registry.complete_attempt(
                    task_id, attempt.attempt_id, candidate
                ):
                    self._queues.enqueue_ready(task_id)
                    outcome = "ready"
            except asyncio.CancelledError:
                # The round closed and `cancel_round` cancelled this reasoner to free the
                # worker. The task is already CANCELLED in the registry, so don't record a
                # failure — just fall through and pick up the next round's work.
                outcome = "cancelled"
            except Exception as error:
                failure = FailureRecord(
                    attempt_id=attempt.attempt_id,
                    error_type=type(error).__name__,
                    error_message=str(error),
                    traceback="".join(traceback.format_exception(error))[-20_000:],
                )
                if self._registry.fail_attempt(task_id, attempt.attempt_id, failure):
                    self._queues.enqueue_failed(task_id)
            finally:
                # Events were written inline as they were emitted — nothing to flush here.
                self._queues.agent.task_done()
                if self._loop is not None and self._on_completion is not None:
                    self._loop.call_soon_threadsafe(
                        self._on_completion, task_id, outcome
                    )

    @staticmethod
    def _reasoner_prompt(prompt: str, attempt: Attempt) -> str:
        if not attempt.context_feedback:
            return prompt
        feedback = "Competition feedback:\n" + "\n".join(attempt.context_feedback)
        return prompt + "\n\n" + feedback

    @staticmethod
    def _normalize_answer(raw: Any) -> tuple[str, str, list[str]]:
        if isinstance(raw, dict):
            answer = raw.get("answer", "")
            reasoning = raw.get("reasoning", "")
            source_docs = raw.get("source_docs", [])
        else:
            answer = getattr(raw, "answer", "")
            reasoning = getattr(raw, "reasoning", "")
            source_docs = getattr(raw, "source_docs", [])
        cleaned_docs = [
            str(item).strip()[:MAX_SOURCE_DOC_REF_CHARS]
            for item in source_docs
            if str(item).strip()
        ][:64]
        return str(answer), str(reasoning), cleaned_docs
