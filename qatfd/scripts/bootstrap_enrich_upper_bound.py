"""Upper-bound experiment for the bootstrap / enrich idea.

One long-lived `SearchAgentSystem` answers the SAME X dev questions `passes` times (default twice), in
sequence, so the second pass sees whatever the first pass left behind on the chroma server: the search
agents' own trajectory working sets (the `docs_for_q*` collections), the BootstrapAgent's collections
(`enrich_working_sets=before` / `both`, built once before the first question), and/or the EnrichAgent's
curation (`enrich_working_sets=after` / `both`, run every `enrich_query_batch_size` questions). The stock
runner cannot express this: it keys results by qid (a repeated qid is skipped), and a second invocation
would build a fresh system (the bootstrap would run again, the enrich history would be empty).

The one knob the stock configs do not have is whether the search agent PERSISTS its trajectory as a
collection. `systems.retrieve.working_set_collection_off` only changes what the agent is told (base
collection only vs. base + every managed collection); the `docs_for_q*` collection is created and filled
either way. `+ub.trajectory_working_sets=false` drops that collection right after the agent is built and
swallows its writes, so with bootstrap / enrich on and trajectory working sets off the agent still sees
the collection agents' collections (keep `working_set_collection_off=false`) but never its predecessors'
trajectories.

Config: the stock Hydra config (`configs/config.yaml`) plus an `ub` group added on the command line:

    +ub.label=<run label>                 # run dir name prefix (results/<bench>/search_agent_ub/<label>_<ts>)
    +ub.num_queries=<X>                   # X dev questions: a seeded permutation of the split, first X taken,
    +ub.sample_seed=0                     #   so the X=1 / 5 / 10 subsets nest. `experiments.qids` overrides both.
    +ub.passes=2                          # how many times the sequence is answered
    +ub.trajectory_working_sets=true      # persist each search agent's trajectory as a collection?
    +ub.wipe_collections=true             # delete every non-corpus collection before the run
    +ub.keep_collections=[a,b]            # corpus collections never deleted (the active one is always kept)

Outputs (in the run dir): config.yaml / overrides.txt (the composed config, as the stock runner writes),
results.jsonl + report.csv (one row per (pass, question), a `pass` and `order` column ahead of the stock
columns), traces/pass<p>/<qid>.jsonl, collections_after_pass<p>.json (the managed collections on the server
at the end of each pass) and summary.json (per-pass means + a per-question pass-over-pass table).

Usage (from qatfd/; see scripts/run_bootstrap_enrich_upper_bound.sh for the full sweep):

    python3 scripts/bootstrap_enrich_upper_bound.py benchmarks=officeqa systems=search_agent \
        benchmarks.collection_name=officeqa-qwen-8b-v1 experiments.run_mode=sequential experiments.workers=1 \
        systems.retrieve.enrich_working_sets=before \
        +ub.label=x5_bs +ub.num_queries=5 +ub.trajectory_working_sets=false \
        "+ub.keep_collections=[officeqa-qwen-8b,officeqa-qwen-8b-v1]"
"""

from __future__ import annotations

import asyncio
import copy
import csv
import dataclasses
import json
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import cast

import hydra
from chromadb.api import ClientAPI
from chromadb.errors import NotFoundError
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from skunk.common import ExecutionContext
from skunk.config import InferenceConfig

from qatfd.agents.search_agent import SearchAgent
from qatfd.benchmarks.base import Benchmark, BenchmarkResources
from qatfd.config import ExperimentConfig, benchmark_config_factory, system_config_factory
from qatfd.registry import build_benchmark
from qatfd.runner import _RunCtx, _persist_run_config, _resolve_run_dir, _run_one, _select_questions
from qatfd.systems.search_agent import SearchAgentSystem
from qatfd.tools import ListCollectionsTool, is_managed_collection
from qatfd.types import Question, Result, report_columns, result_to_row

# results land under results/<benchmark>/<SYSTEM_DIR>/ so they never collide with the stock sweeps'
# results/<benchmark>/search_agent/ run dirs (whose scripts glob those by label)
SYSTEM_DIR = "search_agent_ub"

# chromadb caps list_collections() at 100 per call
_LIST_LIMIT = 100


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@dataclass
class UpperBoundConfig:
    """The `ub` config group (added on the CLI with `+ub.<key>=<value>`)."""
    # run label; the run dir is results/<benchmark>/search_agent_ub/<label>_<timestamp>
    label: str
    # X: how many dev questions to answer (ignored when experiments.qids is given)
    num_queries: int | None = None
    # seed of the permutation of the split the X questions are the head of
    sample_seed: int = 0
    # how many times the X-question sequence is answered by the one system instance
    passes: int = 2
    # persist each search agent's trajectory (the chunks it searched / fetched) as a collection?
    trajectory_working_sets: bool = True
    # delete every collection except the corpus collections before the run
    wipe_collections: bool = True
    # corpus collections that are never deleted (the active benchmarks.collection_name is always kept)
    keep_collections: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# the system: a SearchAgentSystem whose search agents can run without a trajectory collection
# ---------------------------------------------------------------------------

class _NullCollection:
    """Stand-in for the search agent's trajectory collection when trajectory working sets are off: the
    agent's result upserts and its action ledger go nowhere (`SearchAgent._handle_tool_call` /
    `_update_collection_actions` only ever call `upsert`, `metadata` and `modify` on it)."""

    name = "<no trajectory collection>"

    @property
    def metadata(self) -> dict:
        return {}

    def upsert(self, **_) -> None:
        pass

    def modify(self, **_) -> None:
        pass

    def count(self) -> int:
        return 0


class UpperBoundSearchAgentSystem(SearchAgentSystem):
    """`SearchAgentSystem` with one extra switch: whether each search agent keeps its trajectory
    collection. Everything else (bootstrap / enrich scheduling, usage keys, prompts) is inherited."""

    def __init__(self, retrieve_config, compute_config, inference_cfg: InferenceConfig, *, trajectory_working_sets: bool) -> None:
        super().__init__(retrieve_config, compute_config, inference_cfg)
        self._trajectory_working_sets = trajectory_working_sets

    def _build_search_agent(self, ctx: ExecutionContext, resources: BenchmarkResources, q: Question) -> SearchAgent:
        agent = super()._build_search_agent(ctx, resources, q)
        if not self._trajectory_working_sets:
            # the SearchAgent created its (still empty) docs_for_q* collection in __init__; drop it before
            # any result is written so no later agent (or the EnrichAgent) ever sees this trajectory
            name = agent._collection.name
            try:
                resources.chroma_client.delete_collection(name)
            except NotFoundError:
                pass
            agent._collection = _NullCollection()  # type: ignore[assignment]
            ctx.tracer.emit(id="trajectory_collection_dropped", kind="lifecycle", data={"collection": name})
        return agent


# ---------------------------------------------------------------------------
# chroma helpers
# ---------------------------------------------------------------------------

def _list_all_collections(client: ClientAPI) -> list:
    out, offset = [], 0
    while True:
        page = list(client.list_collections(limit=_LIST_LIMIT, offset=offset))
        out.extend(page)
        if len(page) < _LIST_LIMIT:
            break
        offset += _LIST_LIMIT
    return out


def wipe_collections(client: ClientAPI, keep: set[str]) -> None:
    """Delete every agent-managed collection (`is_working_set` metadata flag: the search agents'
    docs_for_q* trajectories and the Bootstrap / Enrich agents' collections) except `keep`, retrying
    transient 500s the way scripts/run_collection_agent_ablation.sh does. Corpus collections carry no
    such flag, so they are never deleted even when they are missing from `keep`; they are only reported.
    Aborts if a managed collection is left over."""
    all_collections = _list_all_collections(client)
    unmanaged = sorted(c.name for c in all_collections if c.name not in keep and not is_managed_collection(c))
    if unmanaged:
        print(f"[ub] leaving {len(unmanaged)} unmanaged (corpus-like) collection(s) alone: {unmanaged}")
    names = [c.name for c in all_collections if c.name not in keep and is_managed_collection(c)]
    failed: list[tuple[str, str]] = []
    for name in names:
        for attempt in range(3):
            try:
                client.delete_collection(name)
                break
            except NotFoundError:
                break
            except Exception as e:  # noqa: BLE001 — a delete can 500 transiently (server still compacting it)
                if attempt == 2:
                    failed.append((name, f"{type(e).__name__}: {e}"))
                else:
                    time.sleep(5)
    left = [c.name for c in _list_all_collections(client) if c.name not in keep and is_managed_collection(c)]
    print(f"[ub] wiped {len(names) - len(failed)} managed collection(s); kept {sorted(keep)}; {len(left)} managed collection(s) remain")
    for name, err in failed:
        print(f"[ub] delete failed for {name!r}: {err}", file=sys.stderr)
    if left:
        raise Exception(f"[ub] ABORT: managed collections left on the server after the wipe: {left}")


def snapshot_collections(client: ClientAPI, base_collection_name: str, path: Path) -> list[dict]:
    """Write the managed collections on the server (name / description / size / fields / actions)."""
    summaries = ListCollectionsTool(client, base_collection_name)()["collections"]
    path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return summaries


# ---------------------------------------------------------------------------
# question selection
# ---------------------------------------------------------------------------

def select_questions(benchmark: Benchmark, exp_cfg: ExperimentConfig, ub: UpperBoundConfig) -> list[Question]:
    """`experiments.qids` (in the given order) if set; otherwise the first `num_queries` of a seeded
    permutation of the split, so the subsets for increasing X nest (X=1 ⊂ X=5 ⊂ X=10 for one seed)."""
    if exp_cfg.qids:
        return _select_questions(benchmark, exp_cfg)
    if not ub.num_queries or ub.num_queries < 1:
        raise Exception("[ub] ABORT: set +ub.num_queries=<X> (or experiments.qids=[...])")
    whole_split = dataclasses.replace(exp_cfg, sample=None, sample_seed=None, shuffle_seed=None, shuffle_group_key=None, qids=None)
    pool = sorted(_select_questions(benchmark, whole_split), key=lambda q: q.qid)
    if ub.num_queries > len(pool):
        raise Exception(f"[ub] ABORT: num_queries={ub.num_queries} exceeds the {len(pool)}-question '{exp_cfg.split}' split")
    order = random.Random(ub.sample_seed).sample(pool, len(pool))
    return order[: ub.num_queries]


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

_MEAN_FIELDS = [
    "score", "cost", "retrieve_cost", "compute_cost", "wall_s", "retrieve_wall_s", "compute_wall_s",
    "total_input_tokens", "total_output_tokens", "total_cache_input_tokens",
]
_SUM_FIELDS = ["precompute_cost", "enrich_cost", "precompute_wall_s", "enrich_wall_s"]


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def summarize(rows: list[tuple[int, int, Result]]) -> dict:
    passes = sorted({p for p, _, _ in rows})
    per_pass: dict[str, dict] = {}
    for p in passes:
        rs = [r for pp, _, r in rows if pp == p]
        metric_keys = sorted({k for r in rs for k in r.recall_metrics})
        s: dict = {
            "n": len(rs),
            "n_failed": sum(1 for r in rs if r.failed),
            "n_perfect": sum(1 for r in rs if r.score >= 1.0),
            "n_out_of_steps": sum(1 for r in rs if r.out_of_steps),
        }
        s.update({f"mean_{k}": _mean([getattr(r, k) for r in rs]) for k in _MEAN_FIELDS})
        s.update({f"mean_{k}": _mean([r.recall_metrics.get(k, 0.0) for r in rs]) for k in metric_keys})
        s.update({f"sum_{k}": sum(getattr(r, k) for r in rs) for k in _SUM_FIELDS})
        per_pass[str(p)] = s

    # per-question, pass over pass: did the repeat get cheaper / faster / better?
    per_question: dict[str, dict] = {}
    for p, order, r in rows:
        q = per_question.setdefault(r.qid, {"order": order})
        q[f"pass{p}"] = {
            "score": r.score, "failed": r.failed, "retrieve_cost": r.retrieve_cost, "retrieve_wall_s": r.retrieve_wall_s,
            "input_tokens": r.total_input_tokens, "output_tokens": r.total_output_tokens,
            "n_retrieved": len(json.loads(r.retrieved_docs or "[]")), "out_of_steps": r.out_of_steps,
            **{k: v for k, v in r.recall_metrics.items()},
        }
    return {"per_pass": per_pass, "per_question": per_question}


def print_summary(summary: dict, report_path: Path) -> None:
    per_pass = summary["per_pass"]
    keys = ["n", "n_failed", "n_perfect", "mean_score"]
    keys += [k for k in next(iter(per_pass.values())) if k.startswith("mean_") and k not in ("mean_score",) and k[5:] not in _MEAN_FIELDS]
    keys += ["mean_retrieve_cost", "mean_compute_cost", "sum_precompute_cost", "sum_enrich_cost",
             "mean_retrieve_wall_s", "sum_precompute_wall_s", "sum_enrich_wall_s",
             "mean_total_input_tokens", "mean_total_output_tokens", "n_out_of_steps"]
    print(f"\n[ub] Wrote {report_path}")
    print(f"[ub] {'metric':<28}" + "".join(f"{'pass ' + p:>16}" for p in per_pass))
    for k in keys:
        cells = []
        for s in per_pass.values():
            v = s.get(k, 0)
            cells.append(f"{v:>16.4f}" if isinstance(v, float) else f"{v:>16}")
        print(f"[ub] {k:<28}" + "".join(cells))


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def run(
    benchmark: Benchmark,
    system: UpperBoundSearchAgentSystem,
    exp_cfg: ExperimentConfig,
    ub: UpperBoundConfig,
    cfg: DictConfig,
    run_dir: Path,
    overrides: list[str] | None,
) -> None:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore
    run_dir.mkdir(parents=True, exist_ok=False)
    _persist_run_config(run_dir, cfg, overrides)

    questions = select_questions(benchmark, exp_cfg, ub)
    print(f"[ub] {len(questions)} question(s) x {ub.passes} pass(es): {[q.qid for q in questions]}")

    resources = benchmark.get_resources()
    base_collection_name = benchmark.config.storage.collection_name
    keep = {base_collection_name, *ub.keep_collections}
    if ub.wipe_collections:
        wipe_collections(resources.chroma_client, keep)
    else:
        others = [c.name for c in _list_all_collections(resources.chroma_client) if c.name not in keep]
        print(f"[ub] wipe_collections=false: {len(others)} pre-existing non-corpus collection(s) stay visible to the agents")
    snapshot_collections(resources.chroma_client, base_collection_name, run_dir / "collections_before.json")

    results_path = run_dir / "results.jsonl"
    rows: list[tuple[int, int, Result]] = []
    with results_path.open("w", encoding="utf-8") as rf:
        for p in range(1, ub.passes + 1):
            trace_dir = run_dir / "traces" / f"pass{p}"
            trace_dir.mkdir(parents=True, exist_ok=True)
            rc = _RunCtx(
                benchmark=benchmark, system=system, resources=resources, exp_config=exp_cfg,
                trace_dir=trace_dir, model=system.inference_cfg.llm_model, verbose=exp_cfg.console,
            )
            for i, q in enumerate(questions, 1):
                print(f"[ub] pass {p}/{ub.passes} question {i}/{len(questions)} qid={q.qid}")
                # _run_one mutates q.text (prefixes "Question: "), so every pass gets its own copy
                r = asyncio.run(_run_one(copy.deepcopy(q), rc))
                rf.write(json.dumps({"pass": p, "order": i, **asdict(r)}) + "\n")
                rf.flush()
                rows.append((p, i, r))
                print(
                    f"[ub]   score={r.score:.2f} failed={r.failed} retrieve_cost=${r.retrieve_cost:.4f} "
                    f"retrieve_wall={r.retrieve_wall_s:.1f}s precompute=${r.precompute_cost:.4f}/{r.precompute_wall_s:.0f}s "
                    f"enrich=${r.enrich_cost:.4f}/{r.enrich_wall_s:.0f}s{' reason=' + r.reason if r.reason else ''}"
                )
            cols = snapshot_collections(resources.chroma_client, base_collection_name, run_dir / f"collections_after_pass{p}.json")
            print(f"[ub] end of pass {p}: {sum(1 for c in cols if not c['is_base'])} managed collection(s) on the server")

    report_path = run_dir / "report.csv"
    results = [r for _, _, r in rows]
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pass", "order"] + report_columns(results))
        writer.writeheader()
        for p, i, r in rows:
            writer.writerow({"pass": p, "order": i, **result_to_row(r)})

    summary = summarize(rows)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_summary(summary, report_path)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    print(f"[ub] composed config:\n{OmegaConf.to_yaml(cfg)}")
    try:
        overrides = list(HydraConfig.get().overrides.task)
    except Exception:  # noqa: BLE001 — HydraConfig is unset outside a @hydra.main job
        overrides = []

    if "ub" not in cfg:
        raise Exception("[ub] ABORT: add the experiment group on the CLI, e.g. +ub.label=x5_bs +ub.num_queries=5 (see the module docstring)")
    ub = UpperBoundConfig(**cast(dict, OmegaConf.to_container(cfg.ub, resolve=True)))
    exp_cfg = ExperimentConfig(**cast(dict, OmegaConf.to_container(cfg.experiments, resolve=True)))
    inference_cfg = InferenceConfig(**cast(dict, OmegaConf.to_container(cfg.inference, resolve=True)))
    bench_cfg = benchmark_config_factory(cfg)
    retrieve_cfg, compute_cfg, codex_cfg = system_config_factory(cfg)
    if codex_cfg is not None or retrieve_cfg is None or retrieve_cfg.name != SearchAgentSystem.name:
        raise Exception("[ub] ABORT: this experiment only runs the search_agent system (systems=search_agent)")
    if exp_cfg.run_mode != "sequential" or exp_cfg.workers != 1:
        raise Exception("[ub] ABORT: set experiments.run_mode=sequential experiments.workers=1 (the passes must be answered in order)")
    if exp_cfg.resume_dir:
        raise Exception("[ub] ABORT: resume is not supported (the second pass depends on the first pass's collections)")
    if retrieve_cfg.enrich_working_sets in ("after", "both") and not retrieve_cfg.enrich_query_batch_size:
        raise Exception("[ub] ABORT: systems.retrieve.enrich_query_batch_size is required with enrich_working_sets=after|both")

    exp_cfg.run_name = ub.label
    run_dir = _resolve_run_dir(cfg.results_root, exp_cfg, bench_cfg.name, SYSTEM_DIR)
    benchmark = build_benchmark(bench_cfg)
    system = UpperBoundSearchAgentSystem(retrieve_cfg, compute_cfg, inference_cfg, trajectory_working_sets=ub.trajectory_working_sets)
    print(
        f"[ub] label={ub.label} passes={ub.passes} trajectory_working_sets={ub.trajectory_working_sets} "
        f"working_set_collection_off={retrieve_cfg.working_set_collection_off} enrich_working_sets={retrieve_cfg.enrich_working_sets} "
        f"enrich_query_batch_size={retrieve_cfg.enrich_query_batch_size} collection={bench_cfg.storage.collection_name} run_dir={run_dir}"
    )

    if cfg.dry_run:
        questions = select_questions(benchmark, exp_cfg, ub)
        print(f"[ub] dry_run: would answer {[q.qid for q in questions]} x {ub.passes} pass(es)")
        return

    run(benchmark, system, exp_cfg, ub, cfg, run_dir, overrides)


if __name__ == "__main__":
    main()
