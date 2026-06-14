"""Retrieve-only eval — measures the retrieval stage in isolation (no extract,
no compute, no planner).

For each UID it gets the plan's retrieve branches — from the cache
(`cache/retrieve_bench_plans.jsonl`) if present, else by running the planner here (which
rebuilds the cache in the current canonical-period format; `--replan` forces it) — and
runs `RetrieveOp` over the `page_index` backend (per-era ToC pick → year filter →
coarse summary filter on flash-lite). The retrieved pages are scored against the
benchmark's gold `source_docs` pages.

Reports, per UID:
  - `pct_eliminated` — fraction of the filter's input pages the coarse filter
    dropped (summed over the UID's branches: (pre - kept) / pre).
  - `recall` — fraction of the UID's gold pages present in the retrieved set.
And aggregate recall (micro = total gold hit / total gold; macro = mean).

Guardrails (this run spends real money on flash-lite):
  - HELD-OUT TEST SET is excluded by default (see CLAUDE.md). `--include-test-set`
    overrides for a deliberate final run; `--uids` aborts on any test-set match.
  - Cost: cumulative token spend is estimated after every UID and the run ABORTS
    before exceeding `--budget` (default $50). The estimate uses the conservative
    Flash rate so realistic flash-lite spend stays comfortably under budget.
  - 429s: a logging probe counts every RESOURCE_EXHAUSTED / 429 the LLM client
    hits (it retries internally); the count is reported and a spike aborts.

Usage:
  python3 -m eval.eval_retrieve --csv data/officeqa_pro.csv                 # full dev sweep
  python3 -m eval.eval_retrieve --csv data/officeqa_pro.csv --limit 2       # sanity check
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Reuse the env loader + helpers from the e2e harness. Importing it also runs its
# top-level `_load_env(.env)`, so GEMINI_API_KEY is present before skunk imports.
from eval.eval_e2e import _load_test_set, load_golden  # noqa: E402  (import also runs its .env load)

# --- Run knobs (hard-set; win over .env, which only setdefault's). Must land
# --- before the first config build / LLM call so the rate limiter picks them up.
import os  # noqa: E402

_MODEL = "gemini-3.1-flash-lite"  # default filter model (override: --filter-model)
_RUN_ENV = {
    "SKUNK_RETRIEVER": "page_index",
    "SKUNK_SEMFILTER_BATCH": "32",
}
for _k, _v in _RUN_ENV.items():
    os.environ[_k] = _v
# SKUNK_SEMFILTER_MODEL + RPM/TPM are set from CLI args in main() (the filter model
# is tunable via --filter-model), before the first LLM call.
# RPM/TPM throttles (experiment-scoped) are set from CLI args in main(), before
# the first LLM call, so they can be tuned per run (e.g. --tpm 30000000).

from skunk.common import ExecutionContext, PageRef  # noqa: E402
from skunk.config import SkunkConfig  # noqa: E402
from skunk.errors import StepFailed  # noqa: E402
from skunk.plan import Planner, RetrieveBranch  # noqa: E402
from skunk.prompted_call import load_prompt_overrides  # noqa: E402
from skunk.retrieve import RetrieveOp  # noqa: E402

PLANS_PATH = REPO_ROOT / "cache" / "retrieve_bench_plans.jsonl"

# Token pricing. Flash rate is the conservative guard (overestimates flash-lite
# spend, so the budget cap stops us before real spend reaches it); the lite rate
# is the realistic figure we report.
_FLASH_IN, _FLASH_OUT = 0.30e-6, 2.50e-6
_LITE_IN, _LITE_OUT = 0.10e-6, 0.40e-6

_CALL_RE = re.compile(r"call call_site=(\S+) .*?latency_s=([0-9.]+) in_tok=(\d+) out_tok=(\d+)")
# The page-index query emits one "pick_chapters … pages=N" per era and a final
# "semantic_filter … kept=K/S" (S = pages entering the filter); sum across a UID's
# branches + eras for the per-stage candidate counts.
_PICK_RE = re.compile(r"pick_chapters .* pages=(\d+)")
# "semantic_filter … kept=K/S blocks_kept=B judged=J cached=C blocks=N" — kept/S are PAGES
# (S = pages entering the filter); judged/cached present in the live filter path (absent in the
# SKIPPED ceiling emit), so they're optional groups. `.*` skips the intervening blocks_kept field.
_SEMFILTER_RE = re.compile(
    r"semantic_filter .* kept=(\d+)/(\d+)(?:.* judged=(\d+) cached=(\d+))?"
)
# Final retrieve emit carries the full catalog size — used for the ToC+date elimination
# rate (fraction of the whole index NOT in the candidate set). Output is now block-granular
# (`block_count`); the page-level candidate count is taken from the returned refs, not here.
_RETR_RE = re.compile(r"page_index_retrieve .*catalog_size=(\d+) .*block_count=(\d+)")


# ---------------------------------------------------------------------------
# 429 probe
# ---------------------------------------------------------------------------

class _RateLimitProbe(logging.Handler):
    """Counts RESOURCE_EXHAUSTED / 429 warnings the LLM client emits (it retries
    internally; we just observe)."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.count = 0
        self.samples: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:
            return
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            self.count += 1
            if len(self.samples) < 5:
                self.samples.append(msg[:200])


# ---------------------------------------------------------------------------
# Plan / branch loading
# ---------------------------------------------------------------------------

def _load_plans() -> dict[str, list[RetrieveBranch]]:
    """uid → list of retrieve branches parsed from the cached plans."""
    if not PLANS_PATH.exists():
        sys.exit(f"[retrieve] missing cached plans: {PLANS_PATH}")
    out: dict[str, list[RetrieveBranch]] = {}
    for line in PLANS_PATH.open():
        rec = json.loads(line)
        plan = rec["plan_json"]
        if isinstance(plan, str):
            plan = json.loads(plan)
        branches: list[RetrieveBranch] = []
        for b in plan.get("branches", []):
            if b.get("kind") != "retrieve":
                continue  # lookup_external branches don't touch the page corpus
            branches.append(
                RetrieveBranch(key=b["key"], period=b.get("period"))
            )
        out[rec["uid"]] = branches
    return out


def _write_plans(plans: dict[str, list[RetrieveBranch]]) -> None:
    """(Re)write the plan cache from {uid: retrieve branches}, one record per line, sorted by
    uid — same `{uid, plan_json:{branches:[...]}}` shape `_load_plans` reads."""
    PLANS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PLANS_PATH.open("w") as f:
        for uid in sorted(plans):
            payload = {"branches": [b.model_dump(mode="json") for b in plans[uid]]}
            f.write(json.dumps({"uid": uid, "plan_json": payload}) + "\n")


async def _build_plans(
    uids: list[str], questions: dict[str, str], config: SkunkConfig,
    overrides: tuple, concurrency: int,
) -> tuple[dict[str, list[RetrieveBranch]], int, int]:
    """Run the planner for each UID → {uid: retrieve branches} + planner token totals (in, out).
    The planner expands CY/FY/quarter periods to canonical YYYY-MM via its corpus prompt, so this
    rebuilds a cleared/stale plan cache in the current format. Hooked in before retrieval so the
    eval is self-contained (the offline plan cache is otherwise built by the full e2e eval)."""
    planner = Planner()
    sem = asyncio.Semaphore(concurrency)
    out: dict[str, list[RetrieveBranch]] = {}
    lock = asyncio.Lock()
    totals = {"in": 0, "out": 0, "done": 0}
    n = len(uids)
    t0 = time.perf_counter()

    async def one(uid: str) -> None:
        t = time.perf_counter()
        async with sem:
            ctx = ExecutionContext(question=questions[uid], uid=uid, config=config,
                                   prompt_overrides=overrides)
            err = None
            try:
                plan = await planner.plan(questions[uid], ctx)
                branches = [b for b in plan.branches if b.kind == "retrieve"]
                itok = otok = 0
                for e in ctx.events:
                    m = _CALL_RE.search(e.get("message", ""))
                    if m:
                        itok += int(m.group(3))
                        otok += int(m.group(4))
            except Exception as e:  # noqa: BLE001 — surface, count, keep going
                err, branches, itok, otok = e, [], 0, 0
            finally:
                ctx.close()
        async with lock:
            out[uid] = branches  # type: ignore[assignment]  (retrieve-kind only)
            totals["in"] += itok
            totals["out"] += otok
            totals["done"] += 1
            tag = f"FAILED {type(err).__name__}: {str(err)[:80]}" if err else f"{len(branches)} branch(es)"
            print(f"[plan] {totals['done']}/{n} {uid}: {tag} "
                  f"({time.perf_counter() - t:.1f}s, {time.perf_counter() - t0:.0f}s total)",
                  flush=True)

    await asyncio.gather(*[one(u) for u in uids])
    return out, totals["in"], totals["out"]


def _refkeys(refs: list[PageRef]) -> set[tuple[str, int]]:
    return {(r.month, r.page) for r in refs}


# ---------------------------------------------------------------------------
# Per-UID retrieval
# ---------------------------------------------------------------------------

async def _retrieve_uid(
    question: str, uid: str, branches: list[RetrieveBranch], op: RetrieveOp, config: SkunkConfig,
) -> dict:
    """Run all of a UID's retrieve branches in ONE `run_all` pass — so the semantic
    filter sees the full target list ("fits any target") and scans the deduped union of
    candidates once. `op` is shared across UIDs so the page index/catalog is loaded once,
    not once per UID. Returns retrieved pages plus PER-STAGE stats (ToC pick → year filter
    → semfilter), parsed from the event stream: candidate counts at each stage and tokens
    per stage."""
    t0 = time.perf_counter()   # actual per-UID wall (its branches + parallel batches)
    ctx = ExecutionContext(question=question, uid=uid, config=config)
    retrieved: set[tuple[str, int]] = set()
    n_failed = 0
    # Per-stage accumulators (summed across the UID's retrieve branches).
    s = {
        "chapter_pages": 0,  # candidates after ToC pick (stage 1 out)
        "pre": 0,            # candidates after year filter (stage 2 out, semfilter in)
        "kept": 0,           # candidates after semfilter (stage 3 out)
        "catalog_size": 0,   # full index size (constant; for ToC+date elimination rate)
        "toc_calls": 0, "toc_in": 0, "toc_out": 0,
        "sem_calls": 0, "sem_in": 0, "sem_out": 0,
        "sem_judged": 0, "sem_cached": 0,
        "latency_s": 0.0,    # summed LLM-call latency across this UID's calls
        "toc_latency": 0.0,  # summed latency of toc_pick calls (per-era picks)
        "sem_latency": 0.0,  # summed latency of semfilter calls
    }
    try:
        # One pass over all branches: the filter judges the full target list and the union
        # of candidates is scanned once. A per-branch slot may be a StepFailed (no pages) —
        # attribute it to that branch, keep the rest.
        # A non-failed slot is a `BranchRetrieval`; recall scores against the physical pages
        # its blocks span.
        for slot in await op.run_all(ctx, branches):
            if isinstance(slot, StepFailed):
                n_failed += 1
            else:
                retrieved |= _refkeys(
                    [r for b in slot.blocks for r in b.member_refs]
                )
        for evt in ctx.events:
            msg = evt.get("message", "")
            m = _CALL_RE.search(msg)
            if m:
                site, lat, itok, otok = m.group(1), float(m.group(2)), int(m.group(3)), int(m.group(4))
                s["latency_s"] += lat
                if site == "toc_pick":
                    s["toc_calls"] += 1
                    s["toc_in"] += itok
                    s["toc_out"] += otok
                    s["toc_latency"] += lat
                elif site == "semfilter":
                    s["sem_calls"] += 1
                    s["sem_in"] += itok
                    s["sem_out"] += otok
                    s["sem_latency"] += lat
                continue
            pm = _PICK_RE.search(msg)
            if pm:
                s["chapter_pages"] += int(pm.group(1))   # per-era picked pages (pre-filter)
                continue
            sm = _SEMFILTER_RE.search(msg)
            if sm:
                s["kept"] += int(sm.group(1))            # kept after the semantic filter
                s["pre"] += int(sm.group(2))             # pages entering the semantic filter
                if sm.group(3) is not None:              # live filter path (not the SKIP emit)
                    s["sem_judged"] += int(sm.group(3))  # pages actually sent to the LLM
                    s["sem_cached"] += int(sm.group(4))  # pages served from the decision cache
                continue
            rm = _RETR_RE.search(msg)
            if rm:
                s["catalog_size"] = int(rm.group(1))     # constant across branches
    finally:
        ctx.close()
    return {"retrieved": retrieved, "n_failed_branches": n_failed,
            "wall_s": round(time.perf_counter() - t0, 2), **s}


async def _run_all(uids, questions, plans, gold, config, probe, args, t0) -> tuple[list[dict], int, int]:
    """Process UIDs `args.concurrency`-at-a-time on one event loop, so the global
    RPM/TPM limiters stay saturated. A lock serializes accumulation + printing;
    `stop` halts new work on a budget/429 breach (in-flight UIDs finish)."""
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    rows: list[dict] = []
    st = {"in": 0, "out": 0, "done": 0, "stop": False}
    total = len(uids)
    op = RetrieveOp(config)  # shared across UIDs: the page index/catalog loads once

    async def worker(uid: str) -> None:
        if st["stop"] or not plans.get(uid):
            return
        async with sem:
            if st["stop"]:
                return
            res = await _retrieve_uid(questions[uid], uid, plans[uid], op, config)
        async with lock:
            st["done"] += 1
            i = st["done"]
            st["in"] += res["toc_in"] + res["sem_in"]
            st["out"] += res["toc_out"] + res["sem_out"]
            g = gold.get(uid, set())
            hits = len(res["retrieved"] & g)
            recall = (hits / len(g)) if g else None
            chap, pre, kept = res["chapter_pages"], res["pre"], res["kept"]
            cat, cand = res["catalog_size"], len(res["retrieved"])
            yf_elim = (100.0 * (chap - pre) / chap) if chap else None
            sf_elim = (100.0 * (pre - kept) / pre) if pre else None
            # ToC+date elimination: fraction of the full index NOT in the candidate set.
            elim = (100.0 * (cat - cand) / cat) if cat else None
            rows.append({
                "uid": uid, "n_branches": len(plans[uid]),
                "catalog_size": cat, "candidates": cand, "elim_pct": elim,
                "chapter_pages": chap, "pre": pre, "kept": kept,
                "yearfilter_elim_pct": yf_elim, "semfilter_elim_pct": sf_elim,
                "toc_calls": res["toc_calls"], "toc_in_tok": res["toc_in"], "toc_out_tok": res["toc_out"],
                "sem_calls": res["sem_calls"], "sem_in_tok": res["sem_in"], "sem_out_tok": res["sem_out"],
                "sem_judged": res["sem_judged"], "sem_cached": res["sem_cached"],
                "input_tok": res["toc_in"] + res["sem_in"], "total_latency_s": round(res["latency_s"], 2),
                "toc_latency_s": round(res["toc_latency"], 2), "sem_latency_s": round(res["sem_latency"], 2),
                "wall_s": res["wall_s"],   # actual per-UID wall (parallel batches), measured
                "gold_n": len(g), "hit_n": hits, "recall": recall,
                "n_failed_branches": res["n_failed_branches"],
            })
            lite_cost = st["in"] * _LITE_IN + st["out"] * _LITE_OUT
            rec_s = "n/a " if recall is None else f"{recall:.2f}"
            elim_s = "n/a  " if elim is None else f"{elim:5.2f}%"
            tpm_now = 60.0 * st["in"] / max(1e-6, time.perf_counter() - t0)
            print(f"[{i}/{total}] {uid}: recall={rec_s} elim={elim_s} "
                  f"chap={chap:<5} pre={pre:<5} cand={cand:<5} gold={len(g)} hit={hits} "
                  f"| ${lite_cost:.2f} | {tpm_now / 1e6:.1f}M tok/min | 429s={probe.count}")
            if lite_cost >= args.budget:
                st["stop"] = True
                print(f"[retrieve] ABORT: cost guard (~${lite_cost:.2f} ≥ ${args.budget:.0f}).")
            if probe.count > args.max_429:
                st["stop"] = True
                print(f"[retrieve] ABORT: 429 count {probe.count} > {args.max_429}. Samples: {probe.samples}")

    await asyncio.gather(*[worker(u) for u in uids])
    rows.sort(key=lambda r: r["uid"])
    return rows, st["in"], st["out"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    ap = argparse.ArgumentParser(description="Retrieve-only eval (recall + %% eliminated)")
    ap.add_argument("--csv", default="data/officeqa_pro.csv")
    ap.add_argument("--limit", type=int, help="Run only the first N dev UIDs (sanity check)")
    ap.add_argument("--uids", help="Comma-separated UIDs (aborts on any held-out test UID)")
    ap.add_argument("--include-test-set", action="store_true",
                    help="Include the 32 held-out UIDs (deliberate final run only)")
    ap.add_argument("--budget", type=float, default=float("inf"),
                    help="USD cap; abort before exceeding (default: no cap)")
    ap.add_argument("--concurrency", type=int, default=32, help="UIDs processed in parallel")
    ap.add_argument("--filter-model", default=_MODEL, help="Model for the summary coarse filter")
    ap.add_argument("--rpm", type=int, default=4000, help="Requests/min cap for the filter model")
    ap.add_argument("--tpm", type=int, default=25_000_000,
                    help="Tokens/min cap for the filter model. 25M sits just under the flash-lite "
                         "quota — 30M induced sporadic 429s on the heavy-candidate UIDs.")
    ap.add_argument("--max-429", type=int, default=200, help="Abort if 429 count exceeds this")
    ap.add_argument("--batch", type=int, default=32,
                    help="Semfilter batch size = max CONTENT BLOCKS per LLM call (flat-block filter; "
                         "pages exploded into blocks, packed page-coherently up to this cap)")
    ap.add_argument("--no-semfilter", action="store_true",
                    help="ToC pick + year filter only (skip the semantic filter) — measures the "
                         "ToC+date recall ceiling and elimination rate in isolation")
    ap.add_argument("--replan", action="store_true",
                    help="Force re-running the planner for the selected UIDs (rebuild plan cache)")
    ap.add_argument("--out", default=None, help="Report CSV path")
    args = ap.parse_args()

    # ToC+date-only mode: skip the semantic filter in the retriever (set before config build).
    if args.no_semfilter:
        os.environ["SKUNK_RETRIEVE_SKIP_SEMFILTER"] = "1"

    # Filter model + throttles from CLI (before any LLM call / config build).
    os.environ["SKUNK_SEMFILTER_MODEL"] = args.filter_model
    os.environ["SKUNK_MODEL_RPM"] = f"{args.filter_model}={args.rpm}"
    os.environ["SKUNK_MODEL_TPM"] = f"{args.filter_model}={args.tpm}"
    os.environ["SKUNK_SEMFILTER_BATCH"] = str(args.batch)

    probe = _RateLimitProbe()
    logging.getLogger().addHandler(probe)

    import pandas as pd
    df = pd.read_csv(args.csv)
    questions = {str(r["uid"]): str(r["question"]) for _, r in df.iterrows()}
    gold = {u: _refkeys(p) for u, p in load_golden(args.csv).items()}
    test_set = set() if args.include_test_set else _load_test_set()

    config = SkunkConfig.from_env()
    assert config.retriever == "page_index", "run env not applied"
    overrides_path = Path(config.prompt_overrides_path)
    overrides = load_prompt_overrides(overrides_path) if overrides_path.exists() else ()

    # --- Select UIDs (contamination guard) ---
    if args.uids:
        uids = [u.strip() for u in args.uids.split(",") if u.strip()]
        bad = sorted(set(uids) & _load_test_set())
        if bad and not args.include_test_set:
            sys.exit(f"[retrieve] ABORT: --uids includes held-out test UIDs: {bad}")
    else:
        uids = sorted(u for u in questions if u not in test_set)
        if args.limit:
            uids = uids[: args.limit]

    # --- Plans: hook the planner in so a cleared/stale cache is rebuilt in canonical form.
    # Plan any selected UID missing from the cache (or all, with --replan), then persist.
    plans = _load_plans() if PLANS_PATH.exists() else {}
    need = [u for u in uids if args.replan or u not in plans]
    plan_in = plan_out = 0
    plan_wall = 0.0
    if need:
        print(f"[retrieve] planning {len(need)} UID(s) via the planner "
              f"(model={config.llm_model}) — rebuilding canonical plans...")
        tp = time.perf_counter()
        generated, plan_in, plan_out = asyncio.run(
            _build_plans(need, questions, config, overrides, args.concurrency))
        plan_wall = time.perf_counter() - tp
        plans.update(generated)
        _write_plans(plans)
        print(f"[retrieve] planned {len(generated)} UID(s) in {plan_wall:.1f}s | "
              f"planner tokens in={plan_in:,} out={plan_out:,} | "
              f"~${plan_in * _FLASH_IN + plan_out * _FLASH_OUT:.2f}")

    budget_s = "none" if args.budget == float("inf") else f"${args.budget:.0f}"
    print(f"[retrieve] UIDs: {len(uids)} | test-set excluded: {len(test_set)} "
          f"| budget: {budget_s} | filter model: {os.environ['SKUNK_SEMFILTER_MODEL']} "
          f"| batch: {os.environ['SKUNK_SEMFILTER_BATCH']} "
          f"| rpm: {os.environ['SKUNK_MODEL_RPM']} | tpm: {os.environ['SKUNK_MODEL_TPM']}")
    print(f"[retrieve] concurrency: {args.concurrency} UIDs in parallel")

    t0 = time.perf_counter()
    rows, tot_in, tot_out = asyncio.run(
        _run_all(uids, questions, plans, gold, config, probe, args, t0)
    )

    # --- Aggregate ---
    scored = [r for r in rows if r["recall"] is not None]
    tot_gold = sum(r["gold_n"] for r in scored)
    tot_hit = sum(r["hit_n"] for r in scored)
    micro = (tot_hit / tot_gold) if tot_gold else float("nan")
    macro = (sum(r["recall"] for r in scored) / len(scored)) if scored else float("nan")
    elim_vals = [r["elim_pct"] for r in rows if r["elim_pct"] is not None]
    mean_elim = (sum(elim_vals) / len(elim_vals)) if elim_vals else float("nan")
    wall = time.perf_counter() - t0
    n = max(1, len(rows))

    # Cost: planner (default model) + retrieval (filter model, realistic flash-lite rate with
    # the flash rate as a conservative upper bound).
    ret_lite = tot_in * _LITE_IN + tot_out * _LITE_OUT
    ret_flash = tot_in * _FLASH_IN + tot_out * _FLASH_OUT
    plan_cost = plan_in * _FLASH_IN + plan_out * _FLASH_OUT
    per_uid = (plan_cost + ret_lite) / n

    def _pct(vals: list[float], p: int) -> float:
        vals = sorted(vals)
        return vals[min(len(vals) - 1, round(p / 100 * (len(vals) - 1)))] if vals else float("nan")
    walls = [r["wall_s"] for r in rows]
    toc_lat = [r["toc_latency_s"] for r in rows]
    sem_lat = [r["sem_latency_s"] for r in rows]

    print("\n" + "=" * 72)
    print(f"UIDs run: {len(rows)} | scored (gold present): {len(scored)}")
    print(f"RECALL    micro (Σhit/Σgold) = {micro:.4f}   macro (mean per-UID) = {macro:.4f}")
    stage_lbl = "ToC+date" if os.environ.get("SKUNK_RETRIEVE_SKIP_SEMFILTER") not in (None, "", "0") else "ToC+date+semfilter"
    print(f"ELIM      mean {stage_lbl} pct_eliminated (of full index) = {mean_elim:.2f}%")
    print("-- LATENCY  (per-UID retrieval wall, seconds) --")
    print(f"  wall/UID   mean={statistics.mean(walls):.2f}  median={statistics.median(walls):.2f}  "
          f"p90={_pct(walls, 90):.2f}  max={max(walls):.2f}")
    print(f"  LLM call latency/UID (summed over a UID's calls): "
          f"toc-pick={statistics.mean(toc_lat):.2f}s  semfilter={statistics.mean(sem_lat):.2f}s")
    print("-- COST --")
    print(f"  planner    in={plan_in:,} out={plan_out:,}  ~${plan_cost:.2f}  ({config.llm_model})")
    print(f"  retrieval  in={tot_in:,} out={tot_out:,}  ~${ret_lite:.2f} flash-lite "
          f"(upper-bound ${ret_flash:.2f} flash)")
    print(f"  TOTAL ~${plan_cost + ret_lite:.2f}  |  per-UID ~${per_uid:.3f}  |  "
          f"project: 101-dev ~${per_uid * 101:.2f}  32-test ~${per_uid * 32:.2f}")
    print(f"429s      {probe.count}")
    print(f"WALL      retrieval {wall:.1f}s ({wall / 60:.1f} min)  +  planning {plan_wall:.1f}s")
    print("=" * 72)

    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "eval" / f"retrieve_recall_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[retrieve] per-UID report → {out_path}")


if __name__ == "__main__":
    main()
