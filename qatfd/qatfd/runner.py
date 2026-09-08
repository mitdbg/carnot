"""Shared parallel runner: pairs a Benchmark with a System, runs the selected
questions concurrently, scores them, and writes report.csv + per-question traces.

Configuration is composed by Hydra from `configs/` — four groups (experiments /
benchmarks / systems / skunk) selected by the root `configs/config.yaml`. `main()` is
the Hydra entry point; it builds a SkunkConfig (env defaults + the `skunk` overrides),
constructs the benchmark + system via `qatfd.registry`, then calls `run()`.

Mirrors skunk eval/eval_e2e.py's patterns: ThreadPoolExecutor over questions with
one asyncio.run per worker, a timestamped run directory, per-question JSONL event
traces, and a pre-allocated results list that preserves input order.
"""

from __future__ import annotations

import asyncio
import csv
import datetime
import json
import logging
import os
import random
import requests
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

# --- Python 3.14 / Hydra 1.3.x compatibility shim ---------------------------------------
# Py3.14's argparse eagerly validates each argument's `help` at add_argument() time (new
# ArgumentParser._check_help -> HelpFormatter._expand_help does `'%' not in help`). Hydra
# passes a lazy, non-str help object (LazyCompletionHelp, defines only __repr__) for its
# `--shell-completion` arg, so that membership test raises TypeError and the parser build
# dies before our @hydra.main entry point ever runs. hydra-core 1.3.3 is the latest release
# and predates 3.14, so there is no fixed upstream version to upgrade to. Coerce any
# non-str help to str before the eager check. No-op on Pythons without _check_help (<3.14),
# and a genuine bad string help still raises normally.
import argparse as _argparse

if hasattr(_argparse.ArgumentParser, "_check_help"):
    _qatfd_orig_check_help = _argparse.ArgumentParser._check_help # type: ignore

    def _qatfd_check_help_str_safe(self, action):
        if action.help is not None and not isinstance(action.help, str):
            action.help = str(action.help)
        return _qatfd_orig_check_help(self, action)

    _argparse.ArgumentParser._check_help = _qatfd_check_help_str_safe # type: ignore

from enum import Enum
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from skunk.common import ExecutionContext
from skunk.config import InferenceConfig, LookupAgentConfig, SkunkConfig, SearchAgentConfig
from skunk.trace import Tracer

from qatfd.benchmarks.base import Benchmark, BenchmarkResources
from qatfd.config import ExperimentConfig, benchmark_config_factory, system_config_factory
from qatfd.registry import build_benchmark, build_system
from qatfd.systems.base import RetrieveComputeSystem, System
from qatfd.types import Question, Result, report_columns, result_to_row

# analytics API constants;
# - metrics are returned for a session; token metrics arrive as strings (or ints)
# - analytics ingestion lags the request by up to ~1 min, so we poll until the row is stable
OPENROUTER_ANALYTICS_API = "https://openrouter.ai/api/v1/analytics/query"
_OPENROUTER_SESSION_METRICS = ["request_count", "total_usage", "tokens_prompt", "tokens_completion", "reasoning_tokens", "cached_tokens"]
OPENROUTER_ANALYTICS_FIRST_WAIT_S = 20.0
OPENROUTER_ANALYTICS_POLL_S = 10.0
OPENROUTER_ANALYTICS_MAX_WAIT_S = 120.0

# keys to ignore when resuming an experiment from a previous run; these fields may naturally change
_RESUME_IGNORED_KEYS = {"experiments.resume_dir", "experiments.run_name", "results_root", "dry_run"}

# modes for running benchmark questions
class RunMode(Enum):
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    ALL = "all"

@dataclass
class _RunCtx:
    benchmark: Benchmark
    system: System
    resources: BenchmarkResources
    trace_dir: Path
    model: str
    verbose: bool


# ---------------------------------------------------------------------------
# Question selection (dev/test split + --qids / --sample)
# ---------------------------------------------------------------------------

def _select_questions(benchmark: Benchmark, config: ExperimentConfig) -> list[Question]:
    all_q = benchmark.load_questions()
    by_qid = {q.qid: q for q in all_q}
    test_qids = benchmark.test_qids()
    dev_qids = benchmark.dev_qids()  # explicit dev list, or None => "everything not in test"

    if config.qids is not None:
        requested = [s.strip() for s in config.qids if s.strip()]
        missing = [r for r in requested if r not in by_qid]
        if missing:
            raise Exception(f"[qatfd] ABORT: unknown qid(s): {', '.join(missing)}")

        if config.split == "dev" and test_qids:
            collision = sorted(set(requested) & test_qids)
            if collision:
                raise Exception(
                    f"[qatfd] ABORT: --qids names {len(collision)} held-out test qid(s): "
                    f"{', '.join(collision)}. Use --split test for a deliberate final run."
                )

        print(f"[qatfd] split '{config.split}': running {len(requested)} question(s).")
        return [by_qid[r] for r in requested]

    if config.split == "test":
        pool = [q for q in all_q if q.qid in test_qids]
    elif dev_qids is not None:
        # explicit dev split from the split file (dev is NOT necessarily test's complement — e.g.
        # BrowseComp-Plus dev is a 50-question sample disjoint from KARL's test set).
        pool = [q for q in all_q if q.qid in dev_qids]
    else:
        pool = [q for q in all_q if q.qid not in test_qids]  # legacy: no split file => dev = all - test
    if config.sample:
        # sort first so the draw depends only on the seed, not load order; a local RNG
        # (seeded when config.seed is set) keeps the subset reproducible across systems.
        pool = sorted(pool, key=lambda q: q.qid)
        rng = random.Random(config.seed) if config.seed is not None else random
        pool = rng.sample(pool, min(config.sample, len(pool)))
    print(f"[qatfd] split '{config.split}': running {len(pool)} question(s).")

    return pool


def _resolve_run_dir(results_root: str, exp_config: ExperimentConfig, benchmark_name: str, system_name: str):
    """
    Resolve the run directory from the given configuration.
    """
    # run directory: resume into an existing one (skip finished questions) or create a fresh
    # timestamped dir. A relative results_root resolves against the cwd; absolute passes through.
    if exp_config.resume_dir:
        run_dir = Path(exp_config.resume_dir).expanduser().resolve()
        if not run_dir.is_dir():
            raise Exception(f"[qatfd] ABORT: resume_dir {run_dir} does not exist.")
        print(f"[qatfd] resuming run at {run_dir}")
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_label = f"{exp_config.run_name}_{ts}" if exp_config.run_name else ts
        run_dir = Path(results_root).expanduser().resolve() / benchmark_name / system_name / run_label

    return run_dir


def _print_summary(rows: list[Result], report_path: Path) -> None:
    assert len(rows) > 0, "no results to summarize"
    n = len(rows)
    n_failed = sum(1 for r in rows if r.failed)
    mean_score = sum(r.score for r in rows) / n
    n_perfect = sum(1 for r in rows if r.score >= 1.0)
    cost = sum(r.cost for r in rows)
    print(f"\n[qatfd] Wrote {report_path} ({n} rows)")
    print(f"[qatfd] {n - n_failed}/{n} produced an answer")
    print(f"[qatfd] Mean score: {mean_score:.3f} ({n_perfect}/{n} fully correct)")
    metric_keys: list[str] = []
    for r in rows:
        for k in r.recall_metrics:
            if k not in metric_keys:
                metric_keys.append(k)
    for k in metric_keys:
        vals = [r.recall_metrics.get(k, 0.0) for r in rows]
        print(f"[qatfd] Mean {k}: {sum(vals) / len(vals):.3f} (over {len(vals)} questions)")
    print(f"[qatfd] Total cost: ${cost:.4f}")


def _persist_run_config(run_dir: Path, cfg: DictConfig, overrides: list[str] | None) -> None:
    """Snapshot the EXACT config this run used into its own results dir, so a historical
    run is self-describing without depending on Hydra's separate `.hydra/` output tree.
    Writes `config.yaml` (the fully-composed, interpolation-resolved config printed at
    startup) and `overrides.txt` (the CLI overrides that produced it)."""
    config_file = run_dir / "config.yaml"
    config_file.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    if overrides:
        overrides_file = run_dir / "overrides.txt"
        overrides_file.write_text("\n".join(overrides) + "\n", encoding="utf-8")


def _flatten_config(node: object, prefix: str = "") -> dict[str, object]:
    """Flatten a nested config container into {dotted.key: leaf} (lists compare as leaves)."""
    if isinstance(node, dict):
        flat: dict[str, object] = {}
        for k, v in node.items():
            flat.update(_flatten_config(v, f"{prefix}.{k}" if prefix else str(k)))
        return flat
    return {prefix: node}


def _check_resume_config(run_dir: Path, cfg: DictConfig) -> None:
    """A resume must be the SAME experiment as the run that created the dir: compare the
    current composed config against the persisted `config.yaml` key by key, and abort on
    any difference outside `_RESUME_IGNORED_KEYS` — otherwise the resumed questions would
    be measured under a different configuration than the rows already in results.jsonl."""
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise Exception(
            f"[qatfd] ABORT: {config_path} is missing, so the resume cannot be verified against "
            f"the original run's config. Restore it or start a fresh run."
        )
    persisted = _flatten_config(OmegaConf.to_container(OmegaConf.load(config_path), resolve=True))
    current = _flatten_config(OmegaConf.to_container(cfg, resolve=True))
    diffs = [
        f"  {key}: persisted={persisted.get(key, '<missing>')!r} current={current.get(key, '<missing>')!r}"
        for key in sorted(set(persisted) | set(current))
        if key not in _RESUME_IGNORED_KEYS and persisted.get(key) != current.get(key)
    ]
    if diffs:
        raise Exception(
            f"[qatfd] ABORT: resume config does not match {config_path}:\n"
            + "\n".join(diffs)
            + f"\nResume with the original run's config (only {', '.join(sorted(_RESUME_IGNORED_KEYS))} may differ)."
        )


def _load_finished_qids(results_path: Path) -> dict[str, Result]:
    """Completed results from a prior (possibly interrupted) run, keyed by qid. Each line of
    `results.jsonl` is one `asdict(Result)`; a truncated/corrupt trailing line (process killed
    mid-write) is skipped rather than fatal."""
    qid_to_result: dict[str, Result] = {}
    if not results_path.exists():
        return qid_to_result
    with results_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = Result(**json.loads(line))
            except Exception:  # noqa: BLE001 — skip a partial/corrupt line (e.g. killed mid-flush)
                continue
            qid_to_result[r.qid] = r
    return qid_to_result


# ---------------------------------------------------------------------------
# Per-question execution
# ---------------------------------------------------------------------------

def _openrouter_session_usage(
    session_id: str, started_at: datetime.datetime, finished_at: datetime.datetime
) -> dict[str, int | float] | None:
    """Per-question spend for one OpenRouter `session_id`. Codex forwards its thread id as the
    session id on every request via the analytics API. Requires a management key to be set as
    OPENROUTER_MGMT_API_KEY. The session filter filters `time_range` is mandatory but is only
    honored at day granularity when no `granularity` is given, so it is widened to whole UTC
    days around the question. Ingestion lags the request by up to about a minute, so the query
    is polled until two consecutive reads agree (or a cap is hit) — the last read is returned
    either way.

    Returns:
      A dictionary with:
      {
        "requests", "input_tokens", "output_tokens",
        "reasoning_tokens", "cached_tokens", "cost"
      }
      or None if there's a missing key / API error / no rows.
    """
    # check to see management key is set
    mgmt_key = os.environ.get("OPENROUTER_MGMT_API_KEY")
    if not mgmt_key:
        print("[qatfd] OPENROUTER_MGMT_API_KEY not set; codex spend not metered", file=sys.stderr)
        return None

    # prepare query for session data
    day = datetime.timedelta(days=1)
    payload = {
        "metrics": _OPENROUTER_SESSION_METRICS,
        "time_range": {
            "start": (started_at - day).strftime("%Y-%m-%dT00:00:00Z"),
            "end": (finished_at + day).strftime("%Y-%m-%dT00:00:00Z"),
        },
        "filters": [{"field": "session_id", "operator": "eq", "value": session_id}],
    }
    headers = {"Authorization": f"Bearer {mgmt_key}", "Content-Type": "application/json"}

    # helper function to fire the query and return the response (or None on bad code / data)
    def _query() -> dict[str, int | float] | None:
        resp = requests.post(OPENROUTER_ANALYTICS_API, json=payload, headers=headers, timeout=30)
        if resp.status_code != 200:
            print(f"[qatfd] openrouter analytics {resp.status_code}: {resp.text[:200]}", file=sys.stderr)
            return None
        rows = resp.json().get("data", {}).get("data", [])
        if not rows:
            return None
        row = rows[0]
        return {
            "requests": int(row.get("request_count") or 0),
            "input_tokens": int(row.get("tokens_prompt") or 0),
            "output_tokens": int(row.get("tokens_completion") or 0),
            "reasoning_tokens": int(row.get("reasoning_tokens") or 0),
            "cached_tokens": int(row.get("cached_tokens") or 0),
            "cost": float(row.get("total_usage") or 0.0),
        }

    # return once we have stable response on usage from OpenRouter
    deadline = time.monotonic() + OPENROUTER_ANALYTICS_MAX_WAIT_S
    time.sleep(OPENROUTER_ANALYTICS_FIRST_WAIT_S)
    last: dict[str, int | float] | None = None
    while True:
        try:
            cur = _query()
        except (requests.RequestException, ValueError) as e:
            print(f"[qatfd] openrouter analytics request failed: {e}", file=sys.stderr)
            cur = None
        if cur is not None and cur == last:
            return cur
        last = cur
        if time.monotonic() >= deadline:
            if last is None:
                print(f"[qatfd] openrouter analytics: no rows for session {session_id} after {OPENROUTER_ANALYTICS_MAX_WAIT_S:.0f}s", file=sys.stderr)
            return last
        time.sleep(OPENROUTER_ANALYTICS_POLL_S)


async def _run_one(q: Question, rc: _RunCtx) -> Result:
    search_config = (
        rc.system.retrieve_config
        if isinstance(rc.system, RetrieveComputeSystem)
        else SearchAgentConfig(name="search", agent_id="search")
    )
    assert isinstance(search_config, SearchAgentConfig)
    ctx = ExecutionContext.build(
        config=SkunkConfig(
            inference=rc.system.inference_cfg,
            storage=rc.benchmark.config.storage,
            search=search_config,
            lookup=LookupAgentConfig(name="lookup", agent_id="lookup"),
        ),
        document_map=rc.resources.document_map,
        chroma_collection=rc.resources.chroma_collection,
        tracer=Tracer(
            log_path=str(rc.trace_dir / f"{q.qid}.jsonl"),
            verbose=rc.verbose,
        ),
    )

    # run the question and parse the output
    predicted, failed, reason, retrieved = "", False, "", None
    terminate_state: str | None = None
    retrieve_wall_s = compute_wall_s = 0.0
    session_id: str | None = None
    t0 = time.monotonic()
    started_at = datetime.datetime.now(datetime.timezone.utc)
    try:
        out = await rc.system.answer(q, rc.resources, ctx)
        predicted = out.answer
        retrieved = out.retrieved_doc_ids
        terminate_state = out.terminate_state
        retrieve_wall_s, compute_wall_s = out.retrieve_wall_s, out.compute_wall_s
        session_id = out.session_id
    except Exception as e:  # noqa: BLE001 — record any failure as a row, never crash the run
        failed, reason = True, f"{type(e).__name__}: {e}"
    wall_s = time.monotonic() - t0
    finished_at = datetime.datetime.now(datetime.timezone.utc)

    # determine whether agent terminated early for an expected reason
    out_of_steps = "out_of_steps" in (terminate_state or "")
    over_cost_budget = "over_cost_budget" in (terminate_state or "")
    over_latency_budget = "over_latency_budget" in (terminate_state or "")

    # Snapshot system usage BEFORE scoring so the judge's tokens are not counted.
    # `cost` is the all-in total across every caller; the system's own spend is split by phase via
    # its stable per-phase keys ({agent_id}_retrieve / _compute), so we can attribute the retrieval
    # method's cost apart from the shared compute answerer. Without a configured agent_id the
    # agents mint their own keys and we can't isolate them, so report 0.
    usage = {
        "total_input_tokens": 0, "total_output_tokens": 0, "total_cache_input_tokens": 0,
        "cost": 0.0, "retrieve_cost": 0.0, "compute_cost": 0.0,
        "embed_tokens": 0.0, "embed_calls": 0.0, "embed_cost": 0.0,
    }
    if isinstance(rc.system, RetrieveComputeSystem):
        tracker = ctx.llm_client.usage
        retrieve_key, compute_key = rc.system.retrieve_usage_key, rc.system.compute_usage_key
        usage = {
            "total_input_tokens": tracker.total_input_tokens,
            "total_output_tokens": tracker.total_output_tokens,
            "total_cache_input_tokens": tracker.total_cached_tokens,
            "cost": tracker.cost(),
            "retrieve_cost": tracker.cost(key=retrieve_key) if retrieve_key else 0.0,
            "compute_cost": tracker.cost(key=compute_key) if compute_key else 0.0,
            "embed_tokens": tracker.total_embed_tokens,
            "embed_calls": tracker.total_embed_calls,
            "embed_cost": tracker.embed_cost(),
        }
    else:
        # CodexSystem: the model calls are made by the codex subprocess (and any subagents it spawns)
        # against OpenRouter directly, so nothing lands in our tracker. Codex forwards its thread id
        # as OpenRouter's `session_id`, and subagent traffic is billed under the parent thread's id,
        # so one analytics query scoped to that session is this question's exact spend.
        if session_id is not None:
            external_usage = _openrouter_session_usage(session_id, started_at, finished_at)
            if external_usage is not None:
                usage = {
                    "total_input_tokens": external_usage["input_tokens"],
                    "total_output_tokens": external_usage["output_tokens"],
                    "total_cache_input_tokens": external_usage["cached_tokens"],
                    "cost": external_usage["cost"],
                    "retrieve_cost": 0.0,
                    "compute_cost": 0.0,
                    "embed_tokens": 0.0,
                    "embed_calls": 0.0,
                    "embed_cost": 0.0,
                }

    score, scorer, judge_rationale = 0.0, "", ""
    if not failed:
        try:
            sc = await rc.benchmark.score(q, predicted, ctx)
            score = float(sc.get("score", 0.0))
            scorer = sc.get("scorer", "")
            judge_rationale = sc.get("judge_rationale", "")
        except Exception as e:  # noqa: BLE001
            reason = f"scoring_failed: {type(e).__name__}: {e}"

    recall_metrics = rc.benchmark.recall_metrics(retrieved, q)
    ctx.tracer.close()

    return Result(
        benchmark=rc.benchmark.name, system=rc.system.name, qid=q.qid, question=q.text,
        predicted=predicted, gold=q.gold, score=score, scorer=scorer,
        recall_metrics=recall_metrics, retrieved_docs=json.dumps(retrieved or []),
        gold_docs=json.dumps(q.gold_docs), failed=failed, reason=reason,
        judge_rationale=judge_rationale, wall_s=round(wall_s, 3),
        retrieve_wall_s=round(retrieve_wall_s, 3), compute_wall_s=round(compute_wall_s, 3),
        out_of_steps=out_of_steps, over_cost_budget=over_cost_budget,
        over_latency_budget=over_latency_budget, **usage,
    )


# ---------------------------------------------------------------------------
# Question Execution Strategies (parallel | sequential | all)
# ---------------------------------------------------------------------------
def _execute_parallel(rc: _RunCtx, qid_to_result: dict[str, Result], todo: list[Question], results_path: Path, exp_config: ExperimentConfig) -> None:
    """Run the benchmark questions in parallel; speeds up experiments when questions are executed in isolation."""
    assert exp_config.workers >= 1, "workers must be >= 1"
    with results_path.open("a", encoding="utf-8") as rf:
        with ThreadPoolExecutor(max_workers=exp_config.workers) as pool:
            futures = [pool.submit(lambda q: asyncio.run(_run_one(q, rc)), q) for q in todo]
            for fut in as_completed(futures):
                r = fut.result()
                rf.write(json.dumps(asdict(r)) + "\n")
                rf.flush()
                qid_to_result[r.qid] = r

def _execute_sequential(rc: _RunCtx, qid_to_result: dict[str, Result], todo: list[Question], results_path: Path, exp_config: ExperimentConfig) -> None:
    """Run the benchmark questions in sequence; good for experiments where state is shared / reused across questions."""
    with results_path.open("a", encoding="utf-8") as rf:
        for q in todo:
            r = asyncio.run(_run_one(q, rc))
            rf.write(json.dumps(asdict(r)) + "\n")
            rf.flush()
            qid_to_result[r.qid] = r

# TODO
# def _execute_all(rc: _RunCtx, qid_to_result: dict[str, Result], todo: list[Question], results_path: Path, exp_config: ExperimentConfig) -> None:
#     """Run the benchmark questions in sequence; good for experiments where state is shared / reused across questions."""
#     results = asyncio.run(_run_all(todo, rc))
#     with results_path.open("a", encoding="utf-8") as rf:
#         for r in results:
#             rf.write(json.dumps(asdict(r)) + "\n")
#             rf.flush()
#             qid_to_result[r.qid] = r

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(
    benchmark: Benchmark,
    system: System,
    exp_config: ExperimentConfig,
    cfg: DictConfig,
    overrides: list[str] | None = None,
) -> None:
    # force stdout to flush after every newline
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore

    # resolve the directory for storing logs and results for this run
    run_dir = _resolve_run_dir(cfg.results_root, exp_config, benchmark.name, system.name)
    trace_dir = run_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    # snapshot the config for a fresh run; a resume must match the original run's snapshot
    if not exp_config.resume_dir:
        _persist_run_config(run_dir, cfg, overrides)
    else:
        _check_resume_config(run_dir, cfg)

    # retrieve questions to run based on configuration
    questions = _select_questions(benchmark, exp_config)
    if not questions:
        raise Exception("[qatfd] ABORT: no questions selected for the run. Check your split, qids, and sample settings.")

    # load any already-completed results so we can skip these qids; also drops any partial question traces
    results_path = run_dir / "results.jsonl"
    qid_to_result = _load_finished_qids(results_path)
    todo = [q for q in questions if q.qid not in qid_to_result]
    for q in todo:
        for ext in (".jsonl", ".txt"):
            (trace_dir / f"{q.qid}{ext}").unlink(missing_ok=True)

    print(f"[qatfd] Trace dir: {trace_dir}")
    print(
        f"[qatfd] benchmark={benchmark.name} system={system.name} questions={len(questions)} "
        f"done={len(qid_to_result)} todo={len(todo)} workers={exp_config.workers}"
    )

    # build the shared retrieval substrate once (chroma + document_map).
    resources = benchmark.get_resources()
    model = system.inference_cfg.llm_model
    rc = _RunCtx(
        benchmark=benchmark, system=system, resources=resources,
        trace_dir=trace_dir, model=model, verbose=exp_config.console,
    )

    # execute each question and immediately persist its result to results.jsonl
    if RunMode(exp_config.run_mode) == RunMode.PARALLEL:
        _execute_parallel(rc, qid_to_result, todo, results_path, exp_config)
    elif RunMode(exp_config.run_mode) == RunMode.SEQUENTIAL:
        _execute_sequential(rc, qid_to_result, todo, results_path, exp_config)
    # TODO
    # elif RunMode(exp_config.run_mode) == RunMode.ALL:
    #     _execute_all(rc, qid_to_result, todo, results_path, exp_config)
    else:
        raise Exception(f"Unsupported run_mode: {exp_config.run_mode}")

    # write report.csv from the full completed set, in the original selection order
    rows = [qid_to_result[q.qid] for q in questions if q.qid in qid_to_result]
    report_path = run_dir / "report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=report_columns(rows))
        writer.writeheader()
        for r in rows:
            writer.writerow(result_to_row(r))

    # print final summary to console
    _print_summary(rows, report_path)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # httpx logs EVERY request (successes included) at INFO — one line per LLM call and per
    # Chroma-server read, which floods multirun stdout under Hydra's INFO job logging. Bumping
    # it to WARNING loses no error visibility: httpx has no per-status levels (its 429 lines
    # are INFO too), and transient LLM faults (429/5xx) are already logged at WARNING with
    # status + provider detail by skunk's llm_client retry path.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    print(f"[qatfd] composed config:\n{OmegaConf.to_yaml(cfg)}")

    # the CLI overrides that produced this composed config (snapshotted alongside it).
    try:
        overrides = list(HydraConfig.get().overrides.task)
    except Exception:  # noqa: BLE001 — HydraConfig is unset outside a @hydra.main job
        overrides = []

    exp_cfg = ExperimentConfig(**cast(dict, OmegaConf.to_container(cfg.experiments, resolve=True)))
    inference_cfg = InferenceConfig(**cast(dict, OmegaConf.to_container(cfg.inference, resolve=True)))
    bench_cfg = benchmark_config_factory(cfg)
    retrieve_system_cfg, compute_system_cfg, codex_system_cfg = system_config_factory(cfg)

    # build benchmark and system
    benchmark = build_benchmark(bench_cfg)
    system = build_system(
        inference_cfg,
        retrieve_config=retrieve_system_cfg,
        compute_config=compute_system_cfg,
        codex_config=codex_system_cfg,
    )

    if not cfg.dry_run:
        run(benchmark, system, exp_cfg, cfg, overrides=overrides)


if __name__ == "__main__":
    main()
