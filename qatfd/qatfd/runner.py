"""Shared parallel runner: pairs a Benchmark with a System, runs the selected
questions concurrently, scores them, and writes report.csv + per-question traces.

Configuration is composed by Hydra from `configs/` — four groups (experiments /
benchmarks / systems / skunk) selected by the root `configs/config.yaml`. `main()` is
the Hydra entry point; it builds a SkunkConfig (env defaults + the `skunk` overrides),
constructs the benchmark + system via `qatfd.registry`, then calls `run()`.

Mirrors skunk eval/eval_e2e.py's patterns: ThreadPoolExecutor over questions with
one asyncio.run per worker, a timestamped run directory, a JSONL event sink, and a
pre-allocated results list that preserves input order.
"""

# ruff: noqa: E402 — qatfd.env.load_env() MUST run before any `import skunk` (it
# populates the API keys / config skunk reads at import), so the skunk + qatfd
# imports below deliberately follow the load_env() call rather than sitting at the top.
from __future__ import annotations

import asyncio
import csv
import datetime
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from skunk.common import ExecutionContext
from skunk.errors import MissingData, StepFailed
from skunk.llm_client import LLMClient
from skunk.trace import configure_obs

from qatfd.benchmarks.base import Benchmark, BenchmarkResources
from qatfd.config import ExperimentConfig, benchmark_config_factory, system_config_factory
from qatfd.registry import build_benchmark, build_system
from qatfd.systems.base import System
from qatfd.trace_util import dump_trace
from qatfd.types import Question, Result, report_columns, result_to_row

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
    assert len(test_qids) > 0, "benchmark must define at least one test qid for a meaningful dev/test split"

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

    pool = (
        [q for q in all_q if q.qid in test_qids]
        if config.split == "test" else
        [q for q in all_q if q.qid not in test_qids]
    )
    if config.sample:
        pool = random.sample(pool, min(config.sample, len(pool)))
    print(f"[qatfd] split '{config.split}': running {len(pool)} question(s).")

    return pool


# ---------------------------------------------------------------------------
# Per-question execution
# ---------------------------------------------------------------------------

async def _run_one(q: Question, rc: _RunCtx) -> Result:
    llm_client = LLMClient(rc.system.config)
    tracker = llm_client.usage
    log_path = str(rc.trace_dir / f"{q.qid}.log") if rc.trace_dir else None
    ctx = ExecutionContext(
        question=q.text, uid=q.qid, config=rc.system.config, llm_client=llm_client,
        log_path=log_path, verbose=rc.verbose,
        prompt_overrides=rc.resources.prompt_overrides,
    )

    predicted, failed, reason, retrieved = "", False, "", None
    t0 = time.monotonic()
    try:
        out = await rc.system.answer(q, rc.resources, ctx)
        predicted = out.answer
        retrieved = out.retrieved_doc_ids
    except (StepFailed, MissingData) as e:
        failed, reason = True, f"{type(e).__name__}: {e}"
    except Exception as e:  # noqa: BLE001 — record any failure as a row, never crash the run
        failed, reason = True, f"{type(e).__name__}: {e}"
    wall_s = time.monotonic() - t0

    # Snapshot system usage BEFORE scoring so the judge's tokens are not counted.
    # `cost` is the all-in total (generation + embeddings); embed_* break out the
    # query-embedding portion (tokens are exact on OpenRouter, char/4-estimated local).
    usage = {
        "total_input_tokens": tracker.input_tokens,
        "total_output_tokens": tracker.output_tokens,
        "total_cache_input_tokens": tracker.cache_input_tokens,
        "cost": tracker.cost(),
        "embed_tokens": tracker.embed_tokens,
        "embed_calls": tracker.n_embed_calls,
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
        judge_rationale=judge_rationale, wall_s=round(wall_s, 3), **usage,
    )


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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _persist_run_config(
    run_dir: Path, cfg: DictConfig | None, overrides: list[str] | None
) -> None:
    """Snapshot the EXACT config this run used into its own results dir, so a historical
    run is self-describing without depending on Hydra's separate `.hydra/` output tree.
    Writes `config.yaml` (the fully-composed, interpolation-resolved config printed at
    startup) and `overrides.txt` (the CLI overrides that produced it). Best-effort: a
    snapshot failure must never abort the run."""
    try:
        if cfg is not None:
            try:
                text = OmegaConf.to_yaml(cfg, resolve=True)
            except Exception:  # noqa: BLE001 — fall back to unresolved if interpolation fails
                text = OmegaConf.to_yaml(cfg)
            (run_dir / "config.yaml").write_text(text, encoding="utf-8")
        if overrides:
            (run_dir / "overrides.txt").write_text("\n".join(overrides) + "\n", encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        print(f"[qatfd] WARN: failed to persist run config: {e}")


def run(
    benchmark: Benchmark, system: System, exp_config: ExperimentConfig, results_root: str,
    cfg: DictConfig | None = None, overrides: list[str] | None = None,
) -> None:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore

    # retrieve questions to run based on configuration
    questions = _select_questions(benchmark, exp_config)
    if not questions:
        raise Exception("[qatfd] ABORT: no questions selected for the run. Check your split, qids, and sample settings.")

    # set up run directory and event logging; resolve a relative value against the current working directory
    # (absolute values pass through).
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_label = f"{exp_config.run_name}_{ts}" if exp_config.run_name else ts
    run_dir = Path(results_root).expanduser().resolve() / benchmark.name / system.name / run_label
    trace_dir = run_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    _persist_run_config(run_dir, cfg, overrides)
    configure_obs(jsonl_path=str(trace_dir / "events.jsonl") if trace_dir else None)

    print(f"[qatfd] Trace dir: {trace_dir}")
    print(f"[qatfd] benchmark={benchmark.name} system={system.name} questions={len(questions)} workers={exp_config.workers}")

    # build the shared retrieval substrate once (chroma + document_map).
    resources = benchmark.get_resources()
    model = system.config.agent_model_id
    rc = _RunCtx(
        benchmark=benchmark, system=system, resources=resources,
        trace_dir=trace_dir, model=model, verbose=exp_config.console,
    )

    # run the questions and collect results
    results: list[Result | None] = [None] * len(questions)
    if exp_config.workers <= 1:
        for i, q in enumerate(questions):
            results[i] = asyncio.run(_run_one(q, rc))
    else:
        with ThreadPoolExecutor(max_workers=exp_config.workers) as pool:
            futures = {
                pool.submit(lambda q: asyncio.run(_run_one(q, rc)), question): i
                for i, question in enumerate(questions)
            }
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()

    # write results to report.csv
    rows = [r for r in results if r is not None]
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
    print(f"[qatfd] composed config:\n{OmegaConf.to_yaml(cfg)}")

    # the CLI overrides that produced this composed config (snapshotted alongside it).
    try:
        overrides = list(HydraConfig.get().overrides.task)
    except Exception:  # noqa: BLE001 — HydraConfig is unset outside a @hydra.main job
        overrides = []

    exp_cfg = ExperimentConfig(**cast(dict, OmegaConf.to_container(cfg.experiments, resolve=True)))
    bench_cfg = benchmark_config_factory(cfg)
    system_cfg = system_config_factory(cfg)

    # build benchmark and system
    benchmark = build_benchmark(bench_cfg)
    system = build_system(system_cfg)

    if not cfg.dry_run:
        run(benchmark, system, exp_cfg, results_root=cfg.results_root, cfg=cfg, overrides=overrides)


if __name__ == "__main__":
    main()
