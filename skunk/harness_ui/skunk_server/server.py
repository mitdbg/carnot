"""Skunk server composition root and command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import os
import re
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


_YEAR_RE = re.compile(r"(17\d{2}|18\d{2}|19\d{2}|20\d{2})")


def _normal_confidence(value: Any) -> float | None:
    if not isinstance(value, int | float):
        return None
    if value < 0:
        return None
    return float(value) / 100.0 if value > 1 else float(value)


def _confidence_stats(values: list[float]) -> dict[str, float] | None:
    clean = sorted(v for v in values if 0 <= v <= 1)
    if not clean:
        return None
    p10_index = min(len(clean) - 1, max(0, int(len(clean) * 0.1)))
    return {
        "min": round(clean[0], 4),
        "median": round(clean[len(clean) // 2], 4),
        "p10": round(clean[p10_index], 4),
    }


def _collect_confidences(value: Any, out: list[float]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"confidence", "ocr_confidence", "parser_confidence"}:
                confidence = _normal_confidence(item)
                if confidence is not None:
                    out.append(confidence)
            elif isinstance(item, dict | list):
                _collect_confidences(item, out)
    elif isinstance(value, list):
        for item in value:
            _collect_confidences(item, out)


def _doc_meta_from_ref(ref: Any) -> dict[str, Any] | None:
    if isinstance(ref, dict):
        text = " ".join(
            str(ref.get(k, ""))
            for k in (
                "id",
                "source",
                "source_doc",
                "path",
                "label",
                "family",
                "era",
                "fiscal_year",
                "year",
            )
        )
    else:
        text = str(ref or "")
    if not text:
        return None

    low = text.lower()
    year_match = _YEAR_RE.search(low)
    year = None
    if isinstance(ref, dict):
        raw_year = ref.get("fiscal_year", ref.get("year"))
        if isinstance(raw_year, int):
            year = raw_year
        elif isinstance(raw_year, str) and raw_year.isdigit():
            year = int(raw_year)
    if year is None:
        year = int(year_match.group(1)) if year_match else None

    family = None
    label = ref.get("label") if isinstance(ref, dict) else None
    explicit_family = str(ref.get("family", "")).lower() if isinstance(ref, dict) else ""
    if (
        "govinfo_receipts" in low
        or "gov_into_receipts" in low
        or "govinfo receipts" in low
        or explicit_family in {"govinfo_receipts", "gov_into_receipts"}
    ):
        family = "govinfo_receipts"
        label = label or "GovInfo Receipts"
    elif (
        "combined_statement" in low
        or "combined statement" in low
        or explicit_family == "combined_statement"
    ):
        family = "combined_statement"
        label = label or "Combined Statement"

    era = None
    explicit_era = str(ref.get("era", "")).lower() if isinstance(ref, dict) else ""
    if explicit_era in {"modern", "transition", "historical"}:
        era = explicit_era
    elif "__modern__" in low or " modern " in f" {low} ":
        era = "modern"
    elif "__transition__" in low or " transition " in f" {low} ":
        era = "transition"
    elif "__historical__" in low or " historical " in f" {low} ":
        era = "historical"
    elif family == "combined_statement" and year is not None:
        if year >= 2001:
            era = "modern"
        elif 1995 <= year <= 2000:
            era = "transition"
        elif 1872 <= year <= 1994:
            era = "historical"

    if family is None and era is None:
        return None

    if family == "combined_statement" and era in {"modern", "transition"}:
        return {
            "family": family,
            "label": label,
            "era": era,
            "year": year,
            "structure": "chapter_split",
            "ocr_risk": "none",
            "ocr_review_allowed": False,
            "source": text,
        }
    if family == "combined_statement":
        return {
            "family": family,
            "label": label,
            "era": era or "historical",
            "year": year,
            "structure": "scanned_ocr",
            "ocr_risk": "medium",
            "ocr_review_allowed": True,
            "source": text,
        }
    return {
        "family": family,
        "label": label,
        "era": "messy",
        "year": year,
        "structure": "ad_hoc",
        "ocr_risk": "high",
        "ocr_review_allowed": True,
        "source": text,
    }


def _review_corpus_summary(review) -> dict[str, Any] | None:
    guidance = review.guidance or {}
    refs: list[Any] = list(review.source_docs)
    refs.extend(guidance.get("documents") or [])
    for candidate in guidance.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        for key in ("document_id", "source_doc", "source", "path", "bulletin"):
            if candidate.get(key):
                refs.append(candidate[key])

    docs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ref in refs:
        meta = _doc_meta_from_ref(ref)
        if meta is None:
            continue
        key = f"{meta.get('family')}:{meta.get('era')}:{meta.get('year')}:{meta.get('source')}"
        if key in seen:
            continue
        seen.add(key)
        docs.append(meta)

    confidences: list[float] = []
    _collect_confidences(guidance.get("documents") or [], confidences)
    _collect_confidences(guidance.get("candidates") or [], confidences)
    confidence = _confidence_stats(confidences)

    if not docs and confidence is None:
        return None

    risk_rank = {"none": 0, "low": 1, "medium": 2, "high": 3}
    highest_risk = max((d.get("ocr_risk", "none") for d in docs), key=lambda r: risk_rank.get(r, 0), default=None)
    return {
        "documents": docs[:8],
        "families": sorted({d["family"] for d in docs if d.get("family")}),
        "eras": sorted({d["era"] for d in docs if d.get("era")}),
        "confidence": confidence,
        "ocr_review_allowed": any(d.get("ocr_review_allowed") for d in docs) if docs else None,
        "ocr_review_blocked": any(d.get("ocr_review_allowed") is False for d in docs),
        "ocr_risk": highest_risk,
    }


def _review_reason(
    kind: str, corpus: dict[str, Any] | None, guidance: dict[str, Any] | None = None
) -> dict[str, str] | None:
    # A pool review whose values were read by the vision tier (`guidance.visual`) is a visual
    # validation, not an OCR/value check — show that reason regardless of corpus OCR signals.
    if kind == "verify_extract" and guidance and guidance.get("visual"):
        return {"kind": "visual_validation", "policy": "confirm the figure read against source"}
    if corpus is None:
        return None
    if kind == "verify_extract":
        if corpus.get("ocr_review_blocked") and not corpus.get("ocr_review_allowed"):
            return {
                "kind": "value_validation",
                "policy": "structured source; OCR review disabled",
            }
        # ocr_quality is intentionally never surfaced: a vision read shows the visual_validation
        # reason above; otherwise the extract review carries no OCR-quality reason chip.
    if kind == "figure":
        return {"kind": "visual_validation", "policy": "confirm the figure read against source"}
    if kind == "lookup":
        return {"kind": "external_lookup", "policy": "verify value against cited external source"}
    if kind == "replan_approval":
        return {"kind": "replan_approval", "policy": "approve or steer the proposed next plan"}
    return None


def _task_corpus_summary(task) -> dict[str, Any] | None:
    documents: list[dict[str, Any]] = []
    confidences: list[float] = []
    blocked = False
    allowed = False
    for review in task.open_reviews:
        corpus = _review_corpus_summary(review)
        if corpus is None:
            continue
        documents.extend(corpus.get("documents") or [])
        if corpus.get("confidence"):
            confidences.extend(v for v in corpus["confidence"].values() if isinstance(v, float | int))
        blocked = blocked or bool(corpus.get("ocr_review_blocked"))
        allowed = allowed or bool(corpus.get("ocr_review_allowed"))
    candidate = task.latest_candidate
    if candidate is not None:
        for ref in candidate.source_docs:
            meta = _doc_meta_from_ref(ref)
            if meta is not None:
                documents.append(meta)

    if not documents and not confidences:
        return None

    seen: set[str] = set()
    unique_docs: list[dict[str, Any]] = []
    for doc in documents:
        key = f"{doc.get('family')}:{doc.get('era')}:{doc.get('year')}:{doc.get('source')}"
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    risk_rank = {"none": 0, "low": 1, "medium": 2, "high": 3}
    highest_risk = max((d.get("ocr_risk", "none") for d in unique_docs), key=lambda r: risk_rank.get(r, 0), default=None)
    return {
        "documents": unique_docs[:8],
        "families": sorted({d["family"] for d in unique_docs if d.get("family")}),
        "eras": sorted({d["era"] for d in unique_docs if d.get("era")}),
        "confidence": _confidence_stats(confidences),
        "ocr_review_allowed": allowed if unique_docs else None,
        "ocr_review_blocked": blocked or any(d.get("ocr_review_allowed") is False for d in unique_docs),
        "ocr_risk": highest_risk,
    }


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
            open_reviews = task.open_reviews
            review_payloads = []
            for review in open_reviews:
                corpus = _review_corpus_summary(review)
                review_payloads.append(
                    {
                        "review_id": review.review_id,
                        "kind": review.kind,
                        "instructions": review.instructions,
                        "source_docs": review.source_docs,
                        "guidance": review.guidance,
                        "refining": review.refining,
                        "corpus_summary": corpus,
                        "review_reason": _review_reason(review.kind, corpus, review.guidance),
                    }
                )
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
                "corpus_summary": _task_corpus_summary(task),
                # Open human reviews drive the sidebar bump + the review overlay. Small lists,
                # so the whole payload (instruction + candidates + page refs) rides the snapshot.
                "reviews": review_payloads,
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
    refine_fn = _load_refine(config.reasoner_ref) if reasoner is None else None
    broker = HumanWorkBroker(
        registry, recompute_fn, sink.publish_status, refine_fn=refine_fn
    )
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
    app.state.publish_status = sink.publish_status
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


def _load_refine(reasoner_ref: str):
    """Resolve the reasoner module's `refine` entry point (used to revise extracted candidates
    from a reviewer's natural-language feedback). Returns None if the module doesn't define one —
    reviews still open and resolve, the NL-feedback button is just inert."""
    module_name = reasoner_ref.split(":", 1)[0]
    try:
        return load_reasoner(f"{module_name}:refine")
    except Exception:
        logger.warning(
            "reasoner module %s has no `refine`; review NL-feedback disabled",
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
