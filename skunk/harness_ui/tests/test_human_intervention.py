from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import FastAPI

from cup_kit.agent_runtime import AgentAnswer
from skunk_server.agent_worker_pool import AgentWorkerPool
from skunk_server.api import install_routes, snapshot
from skunk_server.domain import HumanInterventionStatus, TaskStatus
from skunk_server.events import EventHub
from skunk_server.human_work_broker import HumanWorkBroker
from skunk_server.human_worker_registry import HumanWorkerRegistry
from skunk_server.task_queues import TaskQueues
from skunk_server.task_registry import TaskConflict, TaskRegistry


def _processing_task(registry: TaskRegistry, question_id: str = "q1"):
    task, _ = registry.create_task(1, question_id, "What is the answer?")
    attempt = registry.begin_attempt(task.task_id, "agent")
    assert attempt is not None
    return task, attempt


class _FakeCoordinator:
    async def submit_ready(self, *_args, **_kwargs) -> None:
        return None

    async def submit_human_candidate(self, *_args, **_kwargs) -> None:
        return None


def test_multiple_interventions_resume_only_after_all_resolve() -> None:
    registry = TaskRegistry()
    task, attempt = _processing_task(registry)
    first = registry.create_intervention(
        task.task_id,
        attempt.attempt_id,
        "pdf_extraction",
        "Read the final row.",
        "Page 17",
        ["Treasury Bulletin 1954-02 PDF page 17"],
    )
    second = registry.create_intervention(
        task.task_id,
        attempt.attempt_id,
        "external_lookup",
        "Find the published rate.",
        None,
        [],
    )

    registry.claim_intervention(first.intervention_id, "worker-1")
    registry.claim_intervention(second.intervention_id, "worker-2")
    registry.resolve_intervention(first.intervention_id, "worker-1", "42", [])

    assert task.status == TaskStatus.AWAIT_HUMAN
    assert first.status == HumanInterventionStatus.RESOLVED

    registry.resolve_intervention(second.intervention_id, "worker-2", "3.1%", [])

    assert task.status == TaskStatus.PROCESSING
    assert second.status == HumanInterventionStatus.RESOLVED


def test_intervention_claim_release_and_owner_validation() -> None:
    registry = TaskRegistry()
    task, attempt = _processing_task(registry)
    intervention = registry.create_intervention(
        task.task_id,
        attempt.attempt_id,
        "external_lookup",
        "Find the value.",
        None,
        [],
    )

    registry.claim_intervention(intervention.intervention_id, "worker-1")
    with pytest.raises(TaskConflict):
        registry.claim_intervention(intervention.intervention_id, "worker-2")
    with pytest.raises(TaskConflict):
        registry.release_intervention(intervention.intervention_id, "worker-2")

    registry.release_intervention(intervention.intervention_id, "worker-1")
    registry.claim_intervention(intervention.intervention_id, "worker-2")
    with pytest.raises(TaskConflict):
        registry.resolve_intervention(
            intervention.intervention_id,
            "worker-1",
            "not the owner",
            [],
        )


def test_broker_waiter_returns_structured_human_response() -> None:
    registry = TaskRegistry()
    workers = HumanWorkerRegistry()
    broker = HumanWorkBroker(registry, TaskQueues(), workers)
    task, attempt = _processing_task(registry)
    worker = workers.create_worker()

    async def run() -> dict:
        waiter = asyncio.create_task(
            broker.request_intervention(
                task.task_id,
                attempt.attempt_id,
                "external_lookup",
                "Find the value.",
                None,
                [],
            )
        )
        await asyncio.sleep(0)
        intervention = task.human_interventions[0]
        broker.claim_intervention(worker.worker_id, intervention.intervention_id)
        broker.resolve_intervention(
            worker.worker_id,
            intervention.intervention_id,
            "",
            ["Treasury Bulletin 1986-06 PDF"],
            [
                {
                    "branch_id": 3,
                    "documents": ["Treasury Bulletin 1986-06 PDF"],
                }
            ],
        )
        return await waiter

    assert asyncio.run(run()) == {
        "response": None,
        "source_docs": ["Treasury Bulletin 1986-06 PDF"],
        "retrieval_directives": [
            {
                "branch_id": 3,
                "documents": ["Treasury Bulletin 1986-06 PDF"],
            }
        ],
    }


def test_round_close_cancels_pending_broker_waiter() -> None:
    registry = TaskRegistry()
    broker = HumanWorkBroker(registry, TaskQueues(), HumanWorkerRegistry())
    task, attempt = _processing_task(registry)

    async def run() -> None:
        waiter = asyncio.create_task(
            broker.request_intervention(
                task.task_id,
                attempt.attempt_id,
                "pdf_extraction",
                "Read the chart.",
                None,
                [],
            )
        )
        await asyncio.sleep(0)
        registry.close_round(1, "CLOSED")
        with pytest.raises(RuntimeError, match="cancelled"):
            await waiter

    asyncio.run(run())
    assert task.status == TaskStatus.CANCELLED
    assert task.human_interventions[0].status == HumanInterventionStatus.CANCELLED


def test_api_claim_resolve_and_serialize_intervention() -> None:
    registry = TaskRegistry()
    queues = TaskQueues()
    workers = HumanWorkerRegistry()
    broker = HumanWorkBroker(registry, queues, workers)
    task, attempt = _processing_task(registry, "api")
    worker = workers.create_worker()
    events = EventHub(lambda worker_id=None: snapshot(registry, workers, worker_id))
    app = FastAPI()
    install_routes(
        app,
        registry,
        workers,
        broker,
        _FakeCoordinator(),  # type: ignore[arg-type]
        events,
    )

    async def run() -> None:
        waiter = asyncio.create_task(
            broker.request_intervention(
                task.task_id,
                attempt.attempt_id,
                "pdf_extraction",
                "Read the total.",
                "Page 4",
                [],
                {
                    "recovery_round": 1,
                    "partial_plan": [
                        {"branch_id": 0, "kind": "retrieve", "key": "total"}
                    ],
                    "likely_pages": [{"bulletin": "1954-02", "page": 4}],
                },
            )
        )
        await asyncio.sleep(0)
        intervention = task.human_interventions[0]
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),  # type: ignore[arg-type]
            base_url="http://test",
        ) as client:
            response = await client.post(
                f"/api/interventions/{intervention.intervention_id}/claim",
                json={"worker_id": worker.worker_id},
            )
            assert response.status_code == 200
            assert response.json()["status"] == HumanInterventionStatus.CLAIMED

            response = await client.post(
                f"/api/interventions/{intervention.intervention_id}/resolve",
                json={
                    "worker_id": worker.worker_id,
                    "response": "100",
                    "source_docs": ["Treasury Bulletin 1954-02 PDF page 4"],
                },
            )
            assert response.status_code == 200
            assert response.json()["task"]["status"] == TaskStatus.PROCESSING
            assert (
                response.json()["intervention"]["status"]
                == HumanInterventionStatus.RESOLVED
            )

            response = await client.get(f"/api/tasks/{task.task_id}")
            assert response.status_code == 200
            assert (
                response.json()["human_interventions"][0]["intervention_id"]
                == intervention.intervention_id
            )
            assert (
                response.json()["human_interventions"][0]["guidance"][
                    "recovery_round"
                ]
                == 1
            )
        assert (await waiter)["response"] == "100"

    asyncio.run(run())


def test_agent_worker_waits_for_mandatory_human_response() -> None:
    registry = TaskRegistry()
    queues = TaskQueues()
    workers = HumanWorkerRegistry()
    broker = HumanWorkBroker(registry, queues, workers)
    task, _ = registry.create_task(1, "worker-hitl", "Question")
    human = workers.create_worker()

    async def reasoner(_prompt, *, human_intervention_handler):
        result = await human_intervention_handler(
            "missing_data",
            "Provide the missing value.",
            "Question context",
            [],
        )
        return AgentAnswer(result["response"], "Used human response.", result["source_docs"])

    async def run() -> None:
        loop = asyncio.get_running_loop()
        broker.start(loop, lambda: asyncio.sleep(0))
        pool = AgentWorkerPool(
            registry,
            queues,
            reasoner,
            1,
            human_requester=broker.request_intervention,
        )
        pool.start(loop, lambda _task_id, _outcome: None)
        queues.enqueue_agent(task.task_id)
        try:
            for _ in range(100):
                if task.status == TaskStatus.AWAIT_HUMAN:
                    break
                await asyncio.sleep(0.01)
            assert task.status == TaskStatus.AWAIT_HUMAN
            intervention = task.human_interventions[0]
            broker.claim_intervention(human.worker_id, intervention.intervention_id)
            broker.resolve_intervention(
                human.worker_id,
                intervention.intervention_id,
                "42",
                ["source"],
            )
            for _ in range(100):
                if task.status == TaskStatus.READY:
                    break
                await asyncio.sleep(0.01)
            assert task.status == TaskStatus.READY
            assert task.latest_candidate is not None
            assert task.latest_candidate.answer_text == "42"
        finally:
            broker.cancel_all()
            pool.stop()

    asyncio.run(run())
