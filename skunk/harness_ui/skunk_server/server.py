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
from skunk_server.api import install_command_routes
from skunk_server.competition_adapter import CompetitionAdapter
from skunk_server.domain import to_jsonable, utc_now
from skunk_server.file_stream import FileSink
from skunk_server.human_work_broker import HumanWorkBroker
from skunk_server.submission_coordinator import SubmissionCoordinator
from skunk_server.task_queues import TaskQueues
from skunk_server.task_registry import TaskRegistry

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    cup_base_url: str
    cup_team_token: str
    stream_dir: str
    reasoner_ref: str = "skunk_reasoner:solve"
    concurrency: int = 3
    queue_size: int = 200
    reconnect_backoff_s: float = 1.0
    # Blocking human transport: a gated branch suspends on the human (web UI) before compute,
    # instead of the default optimistic register-and-keep-going. See HumanWorkBroker.
    human_blocking: bool = False


def create_app(config: ServerConfig, reasoner: Reasoner | None = None) -> FastAPI:
    registry = TaskRegistry()
    queues = TaskQueues(config.queue_size)
    loaded_reasoner = reasoner or load_reasoner(config.reasoner_ref)

    def status_snapshot() -> dict[str, Any]:
        # Compact, events-free status: small enough to re-send whole on every change.
        now = utc_now()  # shared instant so every task's lock TTL is judged consistently

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
                "correct": submission.correct if submission else None,
                "revising": task.revising,
                # client_id annotating this task (None = free); drives the greyed-out buttons +
                # lock indicator for every other client. Expired leases read as free.
                "locked_by": task.active_lock_holder(now),
                # Open human reviews drive the sidebar bump + the review overlay. Small lists,
                # so the whole payload (instruction + candidates + page refs) rides the snapshot.
                "reviews": [
                    {
                        "review_id": review.review_id,
                        "kind": review.kind,
                        "instructions": review.instructions,
                        "source_docs": review.source_docs,
                        "guidance": review.guidance,
                    }
                    for review in task.open_reviews
                ],
            }

        return {
            "round": to_jsonable(registry.round_state()),
            "tasks": [summary(task) for task in registry.list_tasks()],
        }

    # Split backend: the browser-facing fan-out is written to the filesystem (FileSink), which
    # the web process tails back into its own StreamHub. The backend serves no browsers — it
    # installs only command routes (below) and binds to localhost.
    sink = FileSink(status_snapshot, config.stream_dir)
    adapter = CompetitionAdapter(
        config.cup_base_url,
        config.cup_team_token,
        registry,
        queues,
        sink.publish_status,
        sink.roll_round,
        config.reconnect_backoff_s,
    )
    coordinator = SubmissionCoordinator(
        registry,
        queues,
        adapter,
        sink.publish_status,
    )
    adapter.set_round_active_callback(coordinator.on_round_active)
    # Optimistic human-review broker: registers open reviews mid-run and, on resolve, recomputes
    # the answer (re-running only compute) via the reasoner module's `recompute` entry point.
    recompute_fn = _load_recompute(config.reasoner_ref) if reasoner is None else None
    broker = HumanWorkBroker(registry, recompute_fn, sink.publish_status)
    pool = AgentWorkerPool(
        registry,
        queues,
        loaded_reasoner,
        config.concurrency,
        human_broker=broker,
        human_blocking=config.human_blocking,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        loop = asyncio.get_running_loop()
        # Write status.json once up front so the web process has a complete document to read
        # before the first round arrives.
        sink.publish_status()
        broker.start(loop)
        pool.start(loop, coordinator.handle_agent_completion, sink.write_event)
        listener = asyncio.create_task(adapter.listen())
        # Background sweeper that frees review locks whose holder stopped heart-beating.
        lock_sweeper = asyncio.create_task(broker.run_lock_sweeper())
        try:
            yield
        finally:
            listener.cancel()
            lock_sweeper.cancel()
            for task in (listener, lock_sweeper):
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            broker.cancel_all()
            pool.stop()
            sink.close()  # close cached per-task append handles

    app = FastAPI(title="Skunk Server", lifespan=lifespan)
    install_command_routes(app)
    app.state.registry = registry
    app.state.queues = queues
    app.state.coordinator = coordinator
    app.state.broker = broker
    return app


def _load_recompute(reasoner_ref: str):
    """Resolve the reasoner module's `recompute` entry point (used to revise an answer after a
    human review). Returns None if the module doesn't define one — reviews still open and
    resolve, they just don't trigger a recompute."""
    module_name = reasoner_ref.split(":", 1)[0]
    try:
        return load_reasoner(f"{module_name}:recompute")
    except Exception:
        logger.warning(
            "reasoner module %s has no `recompute`; human-review revisions disabled",
            module_name,
        )
        return None


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
    parser = argparse.ArgumentParser(
        description="Skunk competition coordination server"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--cup-base-url", default=os.environ.get("CUP_BASE_URL", ""))
    parser.add_argument("--team-token", default=os.environ.get("CUP_TEAM_TOKEN", ""))
    parser.add_argument(
        "--reasoner", default=os.environ.get("SKUNK_REASONER", "skunk_reasoner:solve")
    )
    parser.add_argument(
        "--concurrency", type=int, default=int(os.environ.get("SKUNK_CONCURRENCY", "3"))
    )
    parser.add_argument(
        "--queue-size", type=int, default=int(os.environ.get("SKUNK_QUEUE_SIZE", "200"))
    )
    parser.add_argument(
        "--stream-dir", default=os.environ.get("SKUNK_STREAM_DIR", "")
    )
    parser.add_argument(
        "--human-blocking",
        action="store_true",
        default=os.environ.get("SKUNK_HUMAN_BLOCKING", "0").lower()
        not in ("", "0", "false", "no", "off"),
        help="suspend a gated branch on the human (web UI) before compute, instead of the "
        "default optimistic register-and-keep-going (env: SKUNK_HUMAN_BLOCKING)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()
    if not args.cup_base_url:
        raise SystemExit("CUP_BASE_URL or --cup-base-url is required")
    if not args.team_token:
        raise SystemExit("CUP_TEAM_TOKEN or --team-token is required")
    if not args.stream_dir:
        raise SystemExit("SKUNK_STREAM_DIR or --stream-dir is required")
    config = ServerConfig(
        cup_base_url=args.cup_base_url,
        cup_team_token=args.team_token,
        stream_dir=args.stream_dir,
        reasoner_ref=args.reasoner,
        concurrency=max(1, args.concurrency),
        queue_size=max(1, args.queue_size),
        human_blocking=args.human_blocking,
    )
    # Cap graceful shutdown so Ctrl-C always reaches the lifespan teardown that stops the
    # worker pool. The browser-facing SSE now lives in the web process, but the backend's
    # localhost command routes can still hold a request open (a resolve → awaited recompute),
    # so keep the deadline defensively.
    uvicorn.run(
        create_app(config), host=args.host, port=args.port, timeout_graceful_shutdown=5
    )


if __name__ == "__main__":
    main()
