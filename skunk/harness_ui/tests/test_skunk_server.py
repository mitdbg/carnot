from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import fitz
import httpx
import pytest  # type: ignore[import-not-found]
from fastapi import FastAPI

from cup_kit.agent_runtime import AgentAnswer
from skunk_client.client import create_app as create_client_app
from skunk_server.agent_worker_pool import AgentWorkerPool
from skunk_server.api import install_routes, snapshot
from skunk_server.domain import (
    AnswerCandidate,
    AssignmentStatus,
    FailureRecord,
    TaskStatus,
)
from skunk_server.events import EventHub
from skunk_server.human_work_broker import HumanWorkBroker
from skunk_server.human_worker_registry import HumanWorkerRegistry
from skunk_server.task_queues import TaskQueues
from skunk_server.task_registry import TaskConflict, TaskRegistry
from skunk_server.submission_coordinator import SubmissionCoordinator
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


def test_first_human_action_wins_and_supersedes_sibling() -> None:
    registry = TaskRegistry()
    task = _ready_task(registry)
    first = registry.create_assignment(task.task_id, "worker-1")
    second = registry.create_assignment(task.task_id, "worker-2")

    registry.retry_from_assignment(
        first.assignment_id,
        first.worker_id,
        first.task_version,
        "Check the cited source.",
    )

    assert task.status == TaskStatus.RETRY_QUEUED
    assert first.status == AssignmentStatus.COMPLETED
    assert second.status == AssignmentStatus.SUPERSEDED
    with pytest.raises(TaskConflict):
        registry.retry_from_assignment(
            second.assignment_id,
            second.worker_id,
            second.task_version,
            "This action is stale.",
        )


def test_retry_feedback_and_cup_feedback_reach_next_attempt() -> None:
    registry = TaskRegistry()
    task = _ready_task(registry)
    assignment = registry.create_assignment(task.task_id, "worker")
    task.cup_feedback.append("Cup score: correct=False, points_awarded=0")
    registry.retry_from_assignment(
        assignment.assignment_id,
        assignment.worker_id,
        assignment.task_version,
        "Try a different interpretation.",
    )

    attempt = registry.begin_attempt(task.task_id, "agent-2")

    assert attempt is not None
    assert attempt.feedback == "Try a different interpretation."
    assert attempt.context_feedback == ["Cup score: correct=False, points_awarded=0"]
    assert task.pending_retry_feedback is None


def test_failed_task_accepts_direct_human_answer() -> None:
    registry = TaskRegistry()
    task, _ = registry.create_task(1, "failed", "Question")
    attempt = registry.begin_attempt(task.task_id, "agent")
    assert attempt is not None
    registry.fail_attempt(
        task.task_id,
        attempt.attempt_id,
        FailureRecord(attempt.attempt_id, "RuntimeError", "model failed"),
    )
    assignment = registry.create_assignment(task.task_id, "human")

    task, candidate = registry.human_answer_from_assignment(
        assignment.assignment_id,
        "human",
        assignment.task_version,
        "Final human answer",
        "Reviewed manually.",
        [],
    )

    assert task.status == TaskStatus.SUBMITTING
    assert candidate.submission_type == "human"
    assert assignment.status == AssignmentStatus.COMPLETED


def test_worker_registry_names_are_mnemonic() -> None:
    workers = HumanWorkerRegistry()
    first = workers.create_worker()
    second = workers.create_worker()

    workers.save_display_name(first.worker_id, "Treasury Tables")

    assert first.worker_id != second.worker_id
    assert first.mnemonic == f"Treasury Tables ({first.worker_id[:4]})"


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
        success_pool.start(loop, lambda task_id, outcome: success_events.append((task_id, outcome)))
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
        failure_pool.start(loop, lambda _task_id, _outcome: None)
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
        pool.start(loop, lambda _task_id, _outcome: None)
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


class _FakeCoordinator:
    def __init__(self) -> None:
        self.submissions: list[dict[str, object]] = []

    async def submit_ready(self, task_id: str, **kwargs) -> None:
        self.submissions.append({"task_id": task_id, **kwargs})
        return None

    async def submit_human_candidate(self, *_args, **_kwargs) -> None:
        return None


def test_api_save_name_claim_and_retry() -> None:
    registry = TaskRegistry()
    queues = TaskQueues()
    workers = HumanWorkerRegistry()
    broker = HumanWorkBroker(registry, queues, workers)
    task = _ready_task(registry, "api")
    worker = workers.create_worker()
    events = EventHub(lambda worker_id=None: snapshot(registry, workers, worker_id))
    app = FastAPI()
    install_routes(app, registry, workers, broker, _FakeCoordinator(), events)  # type: ignore[arg-type]

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),  # type: ignore[arg-type]
            base_url="http://test",
        ) as client:
            response = await client.put(
                f"/api/workers/{worker.worker_id}",
                json={"display_name": "Treasury Tables"},
            )
            assert response.status_code == 200
            response = await client.post(
                "/api/assignments",
                json={"worker_id": worker.worker_id, "task_id": task.task_id},
            )
            assert response.status_code == 200
            assignment = response.json()
            response = await client.post(
                "/api/retries",
                json={
                    "worker_id": worker.worker_id,
                    "assignment_id": assignment["assignment_id"],
                    "task_version": assignment["task_version"],
                    "feedback": "Try again.",
                },
            )
            assert response.status_code == 200
            assert response.json()["status"] == TaskStatus.RETRY_QUEUED

    asyncio.run(run())


def test_api_allows_submission_without_claim() -> None:
    registry = TaskRegistry()
    queues = TaskQueues()
    workers = HumanWorkerRegistry()
    broker = HumanWorkBroker(registry, queues, workers)
    task = _ready_task(registry, "direct-submit")
    events = EventHub(lambda worker_id=None: snapshot(registry, workers, worker_id))
    fake = _FakeCoordinator()
    app = FastAPI()
    install_routes(app, registry, workers, broker, fake, events)  # type: ignore[arg-type]

    async def run() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),  # type: ignore[arg-type]
            base_url="http://test",
        ) as client:
            return await client.post(
                "/api/submissions",
                json={"task_id": task.task_id},
            )

    response = asyncio.run(run())
    assert response.status_code == 200
    assert fake.submissions == [{"task_id": task.task_id, "assignment_id": None, "worker_id": None, "task_version": None}]


def test_client_page_contains_server_and_controls() -> None:
    app = create_client_app("http://example.test:8787")
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),  # type: ignore[arg-type]
            base_url="http://test",
        ) as client:
            return await client.get("/")

    response = asyncio.run(run())
    assert response.status_code == 200
    assert "http://example.test:8787" in response.text
    assert "client.css" in response.text
    assert "client.js" in response.text


def test_client_lists_corpus_documents_newest_first(tmp_path) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    for filename in (
        "treasury_bulletin_1985_12.pdf",
        "treasury_bulletin_1986_06.pdf",
        "treasury_bulletin_1987_13.pdf",
        "other_document_1985.pdf",
    ):
        (pdf_dir / filename).write_bytes(b"%PDF-1.4\n")
    app = create_client_app("http://example.test:8787", pdf_dir=pdf_dir)

    async def run() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),  # type: ignore[arg-type]
            base_url="http://test",
        ) as client:
            return await client.get("/api/corpus-documents")

    response = asyncio.run(run())

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": "1986-06",
            "title": "Treasury Bulletin, June 1986",
            "filename": "treasury_bulletin_1986_06.pdf",
            "reference": "Treasury Bulletin 1986-06 PDF",
        },
        {
            "id": "1985-12",
            "title": "Treasury Bulletin, December 1985",
            "filename": "treasury_bulletin_1985_12.pdf",
            "reference": "Treasury Bulletin 1985-12 PDF",
        },
    ]


def test_client_lists_no_documents_for_missing_or_empty_corpus(tmp_path) -> None:
    async def fetch(pdf_dir) -> httpx.Response:
        app = create_client_app("http://example.test:8787", pdf_dir=pdf_dir)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),  # type: ignore[arg-type]
            base_url="http://test",
        ) as client:
            return await client.get("/api/corpus-documents")

    missing_response = asyncio.run(fetch(tmp_path / "missing"))
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    empty_response = asyncio.run(fetch(empty_dir))

    assert missing_response.status_code == 200
    assert missing_response.json() == []
    assert empty_response.status_code == 200
    assert empty_response.json() == []


def test_client_source_route_serves_pdf_pages(tmp_path) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    pdf_path = pdf_dir / "treasury_bulletin_2026_06.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%test\n")
    app = create_client_app("http://example.test:8787", pdf_dir=pdf_dir)

    async def run() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),  # type: ignore[arg-type]
            base_url="http://test",
        ) as client:
            return await client.get("/api/source/2026-06")

    response = asyncio.run(run())
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert response.headers["content-disposition"].startswith("inline")
    assert response.content.startswith(b"%PDF-1.4")


def test_client_source_page_route_renders_and_caches_png(tmp_path) -> None:
    pdf_dir = tmp_path / "pdfs"
    cache_root = tmp_path / "page-cache"
    pdf_dir.mkdir()
    pdf_path = pdf_dir / "treasury_bulletin_2026_06.pdf"
    with fitz.open() as document:
        page = document.new_page(width=200, height=300)
        page.insert_text((30, 50), "Requested page")
        document.save(pdf_path)
    app = create_client_app(
        "http://example.test:8787",
        pdf_dir=pdf_dir,
        page_cache_root=cache_root,
    )

    async def run() -> tuple[httpx.Response, httpx.Response]:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),  # type: ignore[arg-type]
            base_url="http://test",
        ) as client:
            first = await client.get("/api/source/2026-06/page/1.png")
            second = await client.get("/api/source/2026-06/page/1.png")
            return first, second

    first, second = asyncio.run(run())
    cached_page = cache_root / "renders/2026-06/1.png"
    assert first.status_code == 200
    assert first.headers["content-type"].startswith("image/png")
    assert first.content.startswith(b"\x89PNG\r\n\x1a\n")
    assert second.content == first.content
    assert cached_page.read_bytes() == first.content


def test_client_source_page_route_rejects_invalid_pages(tmp_path) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    app = create_client_app(
        "http://example.test:8787",
        pdf_dir=pdf_dir,
        page_cache_root=tmp_path / "page-cache",
    )

    async def run() -> tuple[httpx.Response, httpx.Response]:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),  # type: ignore[arg-type]
            base_url="http://test",
        ) as client:
            invalid_page = await client.get("/api/source/2026-06/page/0.png")
            missing_page = await client.get("/api/source/2026-06/page/1.png")
            return invalid_page, missing_page

    invalid_page, missing_page = asyncio.run(run())
    assert invalid_page.status_code == 400
    assert missing_page.status_code == 404


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
    ]

    payload = _structured_reasoning_payload(events, ["Treasury Bulletin 1954-02 PDF page 4"])

    assert payload["summary"]["branch_count"] == 2
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
            lambda: asyncio.sleep(0),
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
