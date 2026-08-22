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

# ruff: noqa: E402 — qatfd.env.load_env() MUST run before any `import skunk` (it
# populates the API keys / config skunk reads at import), so the skunk + qatfd
# imports below deliberately follow the load_env() call rather than sitting at the top.
from __future__ import annotations

import asyncio
import csv
import datetime
import json
import logging
import random
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
from skunk.config import InferenceConfig, LookupAgentConfig, OrchestratorConfig, SearchAgentConfig
from skunk.errors import MissingData, StepFailed
from skunk.llm_client import LLMClient

from qatfd.benchmarks.base import Benchmark, BenchmarkResources
from qatfd.config import ExperimentConfig, benchmark_config_factory, system_config_factory
from qatfd.registry import build_benchmark, build_system
from qatfd.systems.base import System
from qatfd.trace_util import dump_trace
from qatfd.types import Question, Result, report_columns, result_to_row

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
    trace_dir: Path | None
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

async def _run_one(q: Question, rc: _RunCtx) -> Result:
    llm_client = LLMClient(rc.system.inference_cfg)
    tracker = llm_client.usage
    log_path = str(rc.trace_dir / f"{q.qid}.jsonl") if rc.trace_dir else None
    search_config = rc.system.config if isinstance(rc.system.config, SearchAgentConfig) else SearchAgentConfig(name="search")
    ctx = ExecutionContext(
        question=q.text, uid=q.qid,
        config=OrchestratorConfig(
            inference=rc.system.inference_cfg,
            storage=rc.benchmark.config.storage,
            search=search_config,
            lookup=LookupAgentConfig(name="lookup"),   # unused
        ),
        llm_client=llm_client,
        document_map=rc.resources.document_map,
        log_path=log_path, verbose=rc.verbose,
        prompt_overrides=rc.resources.prompt_overrides,
    )

    predicted, failed, reason, retrieved = "", False, "", None
    # Defaults for the failure path: if answer() raises we get no AnswerOutput, so the phase
    # splits stay 0 and the terminate flags stay False (the StepFailed text lands in `reason`).
    terminate_state: str | None = None
    retrieve_wall_s = compute_wall_s = 0.0
    t0 = time.monotonic()
    try:
        out = await rc.system.answer(q, rc.resources, ctx)
        predicted = out.answer
        retrieved = out.retrieved_doc_ids
        terminate_state = out.terminate_state
        retrieve_wall_s, compute_wall_s = out.retrieve_wall_s, out.compute_wall_s
    except (StepFailed, MissingData) as e:
        failed, reason = True, f"{type(e).__name__}: {e}"
    except Exception as e:  # noqa: BLE001 — record any failure as a row, never crash the run
        failed, reason = True, f"{type(e).__name__}: {e}"
    wall_s = time.monotonic() - t0

    # A budgeted agent stamps terminate_state as a `|`-joined subset of these tokens (else
    # "finished"/None); split it into one boolean column per reason for easy filtering.
    ts = terminate_state or ""
    out_of_steps = "out_of_steps" in ts
    over_cost_budget = "over_cost_budget" in ts
    over_latency_budget = "over_latency_budget" in ts

    # Snapshot system usage BEFORE scoring so the judge's tokens are not counted.
    # `cost` is the all-in total (generation + embeddings); embed_* break out the
    # query-embedding portion (tokens are exact on OpenRouter, char/4-estimated local).
    # `cost` is the all-in total across every caller; the system's own spend is split by phase via
    # its stable per-phase keys ({agent_id}_retrieve / _compute), so we can attribute the retrieval
    # method's cost apart from the shared compute answerer. `system_cost` is their sum. Without a
    # configured agent_id the agents mint their own keys and we can't isolate them, so report 0.
    retrieve_key, compute_key = rc.system.retrieve_usage_key, rc.system.compute_usage_key
    retrieve_cost = tracker.cost(key=retrieve_key) if retrieve_key else 0.0
    compute_cost = tracker.cost(key=compute_key) if compute_key else 0.0
    usage = {
        "total_input_tokens": tracker.total_input_tokens,
        "total_output_tokens": tracker.total_output_tokens,
        "total_cache_input_tokens": tracker.total_cached_tokens,
        "cost": tracker.cost(),
        "system_cost": retrieve_cost + compute_cost,
        "retrieve_cost": retrieve_cost,
        "compute_cost": compute_cost,
        "embed_tokens": tracker.total_embed_tokens,
        "embed_calls": tracker.total_embed_calls,
        "embed_cost": tracker.embed_cost(),
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

    if rc.trace_dir:
        try:
            dump_trace(
                str(rc.trace_dir / f"{q.qid}.txt"),
                qid=q.qid, benchmark=rc.benchmark.name, system=rc.system.name,
                question=q.text, predicted=predicted, gold=q.gold, score=score,
                failed=failed, reason=reason, events=list(ctx.events), model=rc.model,
            )
        except Exception:  # noqa: BLE001 — a trace-dump failure must not lose the row
            pass
    ctx.close()

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


async def _run_all(qs: list[Question], rc: _RunCtx) -> list[Result]:
    return []

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

def _execute_all(rc: _RunCtx, qid_to_result: dict[str, Result], todo: list[Question], results_path: Path, exp_config: ExperimentConfig) -> None:
    """Run the benchmark questions in sequence; good for experiments where state is shared / reused across questions."""
    results = asyncio.run(_run_all(todo, rc))
    with results_path.open("a", encoding="utf-8") as rf:
        for r in results:
            rf.write(json.dumps(asdict(r)) + "\n")
            rf.flush()
            qid_to_result[r.qid] = r

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
    elif RunMode(exp_config.run_mode) == RunMode.ALL:
        _execute_all(rc, qid_to_result, todo, results_path, exp_config)
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
    system_cfg = system_config_factory(cfg)

    # build benchmark and system
    benchmark = build_benchmark(bench_cfg)
    system = build_system(system_cfg, inference_cfg)

    if not cfg.dry_run:
        run(benchmark, system, exp_cfg, cfg, overrides=overrides)


if __name__ == "__main__":
    main()
