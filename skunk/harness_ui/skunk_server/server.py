"""Skunk server composition root and command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI

from skunk_server.agent_worker_pool import AgentWorkerPool, Reasoner
from skunk_server.api import install_routes
from skunk_server.competition_adapter import CompetitionAdapter
from skunk_server.domain import to_jsonable
from skunk_server.hub import StreamHub
from skunk_server.submission_coordinator import SubmissionCoordinator
from skunk_server.task_queues import TaskQueues
from skunk_server.task_registry import TaskRegistry

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    cup_base_url: str
    cup_team_token: str
    reasoner_ref: str = "skunk_reasoner:solve"
    concurrency: int = 3
    queue_size: int = 200
    auto_submit: bool = False
    reconnect_backoff_s: float = 1.0


def create_app(config: ServerConfig, reasoner: Reasoner | None = None) -> FastAPI:
    registry = TaskRegistry()
    queues = TaskQueues(config.queue_size)
    loaded_reasoner = reasoner or load_reasoner(config.reasoner_ref)

    def status_snapshot() -> dict[str, Any]:
        # Compact, events-free status: small enough to re-send whole on every change.
        def summary(task) -> dict[str, Any]:
            candidate = task.latest_candidate
            submission = task.submissions[-1] if task.submissions else None
            return {
                "task_id": task.task_id,
                "round_num": task.round_num,
                "question_id": task.question_id,
                "prompt": task.prompt,
                "status": task.status.value,
                "answer": candidate.answer_text if candidate else None,
                "points": submission.points_awarded if submission else None,
            }

        return {
            "round": to_jsonable(registry.round_state()),
            "tasks": [summary(task) for task in registry.list_tasks()],
        }

    hub = StreamHub(status_snapshot)
    adapter = CompetitionAdapter(
        config.cup_base_url,
        config.cup_team_token,
        registry,
        queues,
        hub.publish_status,
        config.reconnect_backoff_s,
    )
    coordinator = SubmissionCoordinator(
        registry,
        queues,
        adapter,
        hub.publish_status,
        config.auto_submit,
    )
    pool = AgentWorkerPool(registry, queues, loaded_reasoner, config.concurrency)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        loop = asyncio.get_running_loop()
        pool.start(loop, coordinator.handle_agent_completion, hub.publish_event)
        listener = asyncio.create_task(adapter.listen())
        try:
            yield
        finally:
            listener.cancel()
            try:
                await listener
            except asyncio.CancelledError:
                pass
            pool.stop()

    app = FastAPI(title="Skunk Server", lifespan=lifespan)
    install_routes(app, registry, hub)
    app.state.registry = registry
    app.state.queues = queues
    app.state.hub = hub
    app.state.coordinator = coordinator
    return app


def load_reasoner(ref: str) -> Reasoner:
    if ":" not in ref:
        raise ValueError("reasoner must use module:function form")
    module_name, function_name = ref.split(":", 1)
    harness_dir = Path(__file__).resolve().parents[1]
    for path in (str(harness_dir), os.getcwd()):
        if path not in sys.path:
            sys.path.insert(0, path)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    if not callable(function):
        raise TypeError(f"{ref} is not callable")
    return function


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Skunk competition coordination server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--cup-base-url", default=os.environ.get("CUP_BASE_URL", ""))
    parser.add_argument("--team-token", default=os.environ.get("CUP_TEAM_TOKEN", ""))
    parser.add_argument("--reasoner", default=os.environ.get("SKUNK_REASONER", "skunk_reasoner:solve"))
    parser.add_argument("--concurrency", type=int, default=int(os.environ.get("SKUNK_CONCURRENCY", "3")))
    parser.add_argument("--queue-size", type=int, default=int(os.environ.get("SKUNK_QUEUE_SIZE", "200")))
    parser.add_argument(
        "--auto-submit",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("SKUNK_AUTO_SUBMIT", "").lower() in {"1", "true", "yes"},
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    if not args.cup_base_url:
        raise SystemExit("CUP_BASE_URL or --cup-base-url is required")
    if not args.team_token:
        raise SystemExit("CUP_TEAM_TOKEN or --team-token is required")
    config = ServerConfig(
        cup_base_url=args.cup_base_url,
        cup_team_token=args.team_token,
        reasoner_ref=args.reasoner,
        concurrency=max(1, args.concurrency),
        queue_size=max(1, args.queue_size),
        auto_submit=args.auto_submit,
    )
    # Cap graceful shutdown: the monitoring UI holds open SSE responses
    # (status_frames / event_frames loop forever and the browser EventSource never
    # disconnects), so without a deadline uvicorn waits indefinitely for them to close
    # on Ctrl-C and never reaches the lifespan teardown that stops the worker pool.
    uvicorn.run(create_app(config), host=args.host, port=args.port, timeout_graceful_shutdown=5)


if __name__ == "__main__":
    main()
