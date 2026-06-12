"""Server-owned worker threads that execute reasoning attempts."""

from __future__ import annotations

import asyncio
import inspect
import logging
import queue
import threading
import time
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
EventPublisher = Callable[[str, str, list[dict]], None]


class _EventBatcher:
    """Coalesces a task's trace events into batches, so the global registry lock and the
    main-loop publish wakeup stay OFF the per-event hot path — one lock acquire + one
    `call_soon_threadsafe` per ~32 events / 150ms instead of per event. This frees the GIL
    for the GIL-bound corpus tools (search_corpus/grep_corpus) the reasoner runs; per-event
    telemetry was starving them and ~doubling per-question latency vs the bare eval harness.
    `add` is called synchronously from the reasoner (worker loop or a tool thread); the
    registry append + publish happen at flush time."""

    _FLUSH_N = 32
    _FLUSH_INTERVAL_S = 0.15

    def __init__(
        self,
        registry: TaskRegistry,
        loop: asyncio.AbstractEventLoop | None,
        publish: EventPublisher | None,
        task_id: str,
        attempt_id: str,
    ) -> None:
        self._registry = registry
        self._loop = loop
        self._publish = publish
        self._task_id = task_id
        self._attempt_id = attempt_id
        self._buf: list[dict] = []
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()

    def add(self, event: dict) -> None:
        with self._lock:
            self._buf.append(event)
            now = time.monotonic()
            if len(self._buf) < self._FLUSH_N and (now - self._last_flush) < self._FLUSH_INTERVAL_S:
                return
            batch, self._buf = self._buf, []
            self._last_flush = now
        self._flush(batch)

    def flush(self) -> None:
        with self._lock:
            if not self._buf:
                return
            batch, self._buf = self._buf, []
            self._last_flush = time.monotonic()
        self._flush(batch)

    def _flush(self, batch: list[dict]) -> None:
        stored = self._registry.append_attempt_events(self._task_id, self._attempt_id, batch)
        if stored and self._loop is not None and self._publish is not None:
            self._loop.call_soon_threadsafe(self._publish, self._task_id, self._attempt_id, stored)


class AgentWorkerPool:
    def __init__(
        self,
        registry: TaskRegistry,
        queues: TaskQueues,
        reasoner: Reasoner,
        max_workers: int,
    ) -> None:
        self._registry = registry
        self._queues = queues
        self._reasoner = reasoner
        self._max_workers = max(1, max_workers)
        try:
            parameters = inspect.signature(reasoner).parameters
        except (TypeError, ValueError):
            parameters = {}
        self._reasoner_accepts_trace = (
            "trace_event_handler" in parameters
            or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
        )
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._on_completion: CompletionCallback | None = None
        self._publish_event: EventPublisher | None = None
        # In-flight reasoner runs by task_id → (round_num, worker_loop, future), so a
        # closing round can cancel its still-running reasoners and free the workers. Each
        # worker runs its reasoner on its OWN event loop (see `_worker_loop`), so the
        # future and the loop that owns it are both recorded for cross-thread cancellation.
        self._inflight: dict[str, tuple[int, asyncio.AbstractEventLoop, asyncio.Future]] = {}
        self._inflight_lock = threading.Lock()

    def start(
        self,
        loop: asyncio.AbstractEventLoop,
        on_completion: CompletionCallback,
        publish_event: EventPublisher,
    ) -> None:
        self._loop = loop
        self._on_completion = on_completion
        self._publish_event = publish_event
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
            # Coalesce trace events into batches (registry append + main-loop publish happen
            # per-batch, not per-event) to keep the GIL free for the reasoner's corpus tools.
            batcher = _EventBatcher(
                self._registry, self._loop, self._publish_event, task_id, attempt.attempt_id
            )
            outcome = "failed"
            try:
                task = self._registry.get(task_id)
                if task is None:
                    continue
                prompt = self._reasoner_prompt(task.prompt, attempt)
                reasoner_kwargs: dict[str, Any] = {}
                if self._reasoner_accepts_trace:
                    # Stream every orchestrator event to the task's SSE subscribers; the
                    # batcher buffers them and flushes on the worker's own loop / tool thread.
                    reasoner_kwargs["trace_event_handler"] = batcher.add
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
                if self._registry.complete_attempt(task_id, attempt.attempt_id, candidate):
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
                batcher.flush()  # deliver any events buffered since the last flush
                self._queues.agent.task_done()
                if self._loop is not None and self._on_completion is not None:
                    self._loop.call_soon_threadsafe(self._on_completion, task_id, outcome)

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
