"""FastAPI routes for human workers and server state."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from skunk_server.domain import to_jsonable
from skunk_server.events import EventHub
from skunk_server.human_work_broker import HumanWorkBroker
from skunk_server.human_worker_registry import HumanWorkerRegistry
from skunk_server.submission_coordinator import SubmissionCoordinator
from skunk_server.task_registry import TaskConflict, TaskRegistry


class WorkerUpdate(BaseModel):
    display_name: str = Field(max_length=80)


class AssignmentRequest(BaseModel):
    worker_id: str
    task_id: str


class AssignmentRelease(BaseModel):
    worker_id: str


class SubmissionRequest(BaseModel):
    worker_id: str | None = None
    assignment_id: str | None = None
    task_id: str
    task_version: int | None = None


class RetryRequest(BaseModel):
    worker_id: str
    assignment_id: str
    task_version: int
    feedback: str


class HumanAnswerRequest(BaseModel):
    worker_id: str
    assignment_id: str
    task_version: int
    answer_text: str
    reasoning: str = ""
    source_docs: list[str] = Field(default_factory=list)


class InterventionClaimRequest(BaseModel):
    worker_id: str


class InterventionResolveRequest(BaseModel):
    worker_id: str
    response: str = ""
    source_docs: list[str] = Field(default_factory=list)
    retrieval_directives: list[dict[str, Any]] = Field(default_factory=list)


def install_routes(
    app: FastAPI,
    registry: TaskRegistry,
    workers: HumanWorkerRegistry,
    broker: HumanWorkBroker,
    coordinator: SubmissionCoordinator,
    events: EventHub,
) -> None:
    @app.get("/api/state")
    async def state() -> dict[str, Any]:
        return snapshot(registry, workers)

    @app.get("/api/tasks")
    async def tasks() -> list[dict[str, Any]]:
        return [to_jsonable(task) for task in registry.list_tasks()]

    @app.get("/api/tasks/{task_id:path}")
    async def task_detail(task_id: str) -> dict[str, Any]:
        task = registry.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="task not found")
        return to_jsonable(task)

    @app.put("/api/workers/{worker_id}")
    async def save_worker(worker_id: str, body: WorkerUpdate) -> dict[str, Any]:
        try:
            worker = workers.save_display_name(worker_id, body.display_name)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="worker not found") from error
        await events.broadcast()
        return to_jsonable(worker)

    @app.post("/api/assignments")
    async def create_assignment(body: AssignmentRequest) -> dict[str, Any]:
        try:
            assignment = broker.claim(body.worker_id, body.task_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="worker or task not found") from error
        except TaskConflict as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        await events.broadcast()
        return to_jsonable(assignment)

    @app.post("/api/assignments/{assignment_id}/release")
    async def release_assignment(
        assignment_id: str,
        body: AssignmentRelease,
    ) -> dict[str, Any]:
        try:
            assignment = broker.release(body.worker_id, assignment_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="assignment not found") from error
        except TaskConflict as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        await events.broadcast()
        return to_jsonable(assignment)

    @app.post("/api/submissions")
    async def submit_candidate(body: SubmissionRequest) -> dict[str, str]:
        try:
            await coordinator.submit_ready(
                body.task_id,
                assignment_id=body.assignment_id,
                worker_id=body.worker_id,
                task_version=body.task_version,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail="task or assignment not found") from error
        except (TaskConflict, ValueError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"status": "submitted"}

    @app.post("/api/retries")
    async def retry(body: RetryRequest) -> dict[str, Any]:
        try:
            task = broker.retry(
                body.worker_id,
                body.assignment_id,
                body.task_version,
                body.feedback,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail="task or assignment not found") from error
        except (TaskConflict, ValueError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        await events.broadcast()
        return to_jsonable(task)

    @app.post("/api/human-answers")
    async def human_answer(body: HumanAnswerRequest) -> dict[str, str]:
        try:
            task, candidate = broker.direct_answer(
                body.worker_id,
                body.assignment_id,
                body.task_version,
                body.answer_text,
                body.reasoning,
                body.source_docs,
            )
            await coordinator.submit_human_candidate(task.task_id, candidate)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="task or assignment not found") from error
        except (TaskConflict, ValueError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"status": "submitted"}

    @app.post("/api/interventions/{intervention_id}/claim")
    async def claim_intervention(
        intervention_id: str,
        body: InterventionClaimRequest,
    ) -> dict[str, Any]:
        try:
            intervention = broker.claim_intervention(
                body.worker_id,
                intervention_id,
            )
        except KeyError as error:
            raise HTTPException(
                status_code=404,
                detail="worker or human intervention not found",
            ) from error
        except TaskConflict as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        await events.broadcast()
        return to_jsonable(intervention)

    @app.post("/api/interventions/{intervention_id}/release")
    async def release_intervention(
        intervention_id: str,
        body: InterventionClaimRequest,
    ) -> dict[str, Any]:
        try:
            intervention = broker.release_intervention(
                body.worker_id,
                intervention_id,
            )
        except KeyError as error:
            raise HTTPException(
                status_code=404,
                detail="human intervention not found",
            ) from error
        except TaskConflict as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        await events.broadcast()
        return to_jsonable(intervention)

    @app.post("/api/interventions/{intervention_id}/resolve")
    async def resolve_intervention(
        intervention_id: str,
        body: InterventionResolveRequest,
    ) -> dict[str, Any]:
        try:
            task, intervention = broker.resolve_intervention(
                body.worker_id,
                intervention_id,
                body.response,
                body.source_docs,
                body.retrieval_directives,
            )
        except KeyError as error:
            raise HTTPException(
                status_code=404,
                detail="human intervention not found",
            ) from error
        except (TaskConflict, ValueError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        await events.broadcast()
        return {
            "task": to_jsonable(task),
            "intervention": to_jsonable(intervention),
        }

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        worker = workers.create_worker()
        await events.connect(websocket, worker.worker_id)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            workers.disconnect(worker.worker_id)
            await events.disconnect(websocket)
            await events.broadcast()


def snapshot(
    registry: TaskRegistry,
    workers: HumanWorkerRegistry,
    worker_id: str | None = None,
) -> dict[str, Any]:
    return {
        "worker": to_jsonable(workers.get(worker_id)) if worker_id else None,
        "round": to_jsonable(registry.round_state()),
        "tasks": [to_jsonable(task) for task in registry.list_tasks()],
        "workers": [to_jsonable(worker) for worker in workers.list_workers()],
    }
