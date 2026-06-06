"""Retrieve-only eval — measures the retrieval stage in isolation (no extract,
no compute, no planner).

For each dev UID it reads the cached plan (`cache/retrieve_bench_plans.jsonl`,
built offline so NO planner runs here), pulls out the retrieve branches, and
runs `RetrieveOp` over the `page_index_old` backend (ToC pick → year filter →
full-text coarse filter on flash-lite). The retrieved pages are scored against
the benchmark's gold `source_docs` pages.

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
import ast
import asyncio
import json
import logging
import re
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
    "SKUNK_RETRIEVER": "page_index_old",
    "SKUNK_SEMFILTER_BATCH": "16",
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
from skunk.plan import RetrieveBranch  # noqa: E402
from skunk.retrieve import RetrieveOp  # noqa: E402

PLANS_PATH = REPO_ROOT / "cache" / "retrieve_bench_plans.jsonl"

# Token pricing. Flash rate is the conservative guard (overestimates flash-lite
# spend, so the budget cap stops us before real spend reaches it); the lite rate
# is the realistic figure we report.
_FLASH_IN, _FLASH_OUT = 0.30e-6, 2.50e-6
_LITE_IN, _LITE_OUT = 0.10e-6, 0.40e-6

_CALL_RE = re.compile(r"call call_site=(\S+) .*?latency_s=([0-9.]+) in_tok=(\d+) out_tok=(\d+)")
_FT_RE = re.compile(r"fulltext_filter pages=(\d+) judged=(\d+) cached=(\d+)")
_CHAP_RE = re.compile(r"chapter_size=(\d+)")
_SEMFILTER_RE = re.compile(r"semfilter=(\{.*\})\s*$")


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
                RetrieveBranch(key=b["key"], period=b.get("period"), as_of=b.get("as_of"))
            )
        out[rec["uid"]] = branches
    return out


def _refkeys(refs: list[PageRef]) -> set[tuple[str, int]]:
    return {(r.month, r.page) for r in refs}


# ---------------------------------------------------------------------------
# Per-UID retrieval
# ---------------------------------------------------------------------------

async def _retrieve_uid(
    question: str, uid: str, branches: list[RetrieveBranch], op: RetrieveOp, config: SkunkConfig,
) -> dict:
    """Run every retrieve branch for one UID (sequentially, so the per-question
    decision cache is shared across branches). `op` is shared across UIDs so the
    page index/catalog is loaded once, not once per UID. Returns retrieved pages
    plus PER-STAGE stats (ToC pick → year filter → semfilter), parsed from the
    event stream: candidate counts at each stage and tokens per stage."""
    t0 = time.perf_counter()   # actual per-UID wall (its branches + parallel batches)
    ctx = ExecutionContext(question=question, uid=uid, config=config)
    retrieved: set[tuple[str, int]] = set()
    n_failed = 0
    # Per-stage accumulators (summed across the UID's retrieve branches).
    s = {
        "chapter_pages": 0,  # candidates after ToC pick (stage 1 out)
        "pre": 0,            # candidates after year filter (stage 2 out, semfilter in)
        "kept": 0,           # candidates after semfilter (stage 3 out)
        "toc_calls": 0, "toc_in": 0, "toc_out": 0,
        "sem_calls": 0, "sem_in": 0, "sem_out": 0,
        "sem_judged": 0, "sem_cached": 0,
        "latency_s": 0.0,    # summed LLM-call latency across this UID's calls
    }
    try:
        for branch in branches:
            try:
                refs = await op.run(ctx, branch)
                retrieved |= _refkeys(refs)
            except StepFailed:
                n_failed += 1  # e.g. no pages matched this branch
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
                elif site == "semfilter":
                    s["sem_calls"] += 1
                    s["sem_in"] += itok
                    s["sem_out"] += otok
                continue
            ft = _FT_RE.search(msg)
            if ft:
                s["sem_judged"] += int(ft.group(2))
                s["sem_cached"] += int(ft.group(3))
                continue
            cm = _CHAP_RE.search(msg)
            if cm:
                s["chapter_pages"] += int(cm.group(1))
            sm = _SEMFILTER_RE.search(msg)
            if sm:
                try:
                    meta = ast.literal_eval(sm.group(1))
                    s["pre"] += int(meta.get("pre") or 0)
                    s["kept"] += int(meta.get("kept") or 0)
                except (ValueError, SyntaxError):
                    pass
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
            yf_elim = (100.0 * (chap - pre) / chap) if chap else None
            sf_elim = (100.0 * (pre - kept) / pre) if pre else None
            rows.append({
                "uid": uid, "n_branches": len(plans[uid]),
                "chapter_pages": chap, "pre": pre, "kept": kept,
                "yearfilter_elim_pct": yf_elim, "semfilter_elim_pct": sf_elim,
                "toc_calls": res["toc_calls"], "toc_in_tok": res["toc_in"], "toc_out_tok": res["toc_out"],
                "sem_calls": res["sem_calls"], "sem_in_tok": res["sem_in"], "sem_out_tok": res["sem_out"],
                "sem_judged": res["sem_judged"], "sem_cached": res["sem_cached"],
                "input_tok": res["toc_in"] + res["sem_in"], "total_latency_s": round(res["latency_s"], 2),
                "wall_s": res["wall_s"],   # actual per-UID wall (parallel batches), measured
                "gold_n": len(g), "hit_n": hits, "recall": recall,
                "n_failed_branches": res["n_failed_branches"],
            })
            lite_cost = st["in"] * _LITE_IN + st["out"] * _LITE_OUT
            rec_s = "n/a " if recall is None else f"{recall:.2f}"
            elim_s = "n/a " if sf_elim is None else f"{sf_elim:5.1f}%"
            tpm_now = 60.0 * st["in"] / max(1e-6, time.perf_counter() - t0)
            print(f"[{i}/{total}] {uid}: recall={rec_s} sf_elim={elim_s} "
                  f"chap={chap:<5} pre={pre:<5} kept={kept:<4} gold={len(g)} hit={hits} "
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
    ap.add_argument("--tpm", type=int, default=4_000_000, help="Tokens/min cap for the filter model")
    ap.add_argument("--max-429", type=int, default=200, help="Abort if 429 count exceeds this")
    ap.add_argument("--batch", type=int, default=16, help="Semfilter batch size (pages per LLM call)")
    ap.add_argument("--out", default=None, help="Report CSV path")
    args = ap.parse_args()

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
    plans = _load_plans()
    test_set = set() if args.include_test_set else _load_test_set()

    # --- Select UIDs (contamination guard) ---
    if args.uids:
        uids = [u.strip() for u in args.uids.split(",") if u.strip()]
        bad = sorted(set(uids) & _load_test_set())
        if bad and not args.include_test_set:
            sys.exit(f"[retrieve] ABORT: --uids includes held-out test UIDs: {bad}")
    else:
        uids = [u for u in plans if u not in test_set]
        uids.sort()
        if args.limit:
            uids = uids[: args.limit]

    budget_s = "none" if args.budget == float("inf") else f"${args.budget:.0f}"
    print(f"[retrieve] dev UIDs: {len(uids)} | test-set excluded: {len(test_set)} "
          f"| budget: {budget_s} | model: {os.environ['SKUNK_SEMFILTER_MODEL']} "
          f"| batch: {os.environ['SKUNK_SEMFILTER_BATCH']} "
          f"| rpm: {os.environ['SKUNK_MODEL_RPM']} | tpm: {os.environ['SKUNK_MODEL_TPM']}")

    config = SkunkConfig.from_env()
    assert config.retriever == "page_index_old", "run env not applied"
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
    elim_vals = [r["semfilter_elim_pct"] for r in rows if r["semfilter_elim_pct"] is not None]
    mean_elim = (sum(elim_vals) / len(elim_vals)) if elim_vals else float("nan")
    lite_cost = tot_in * _LITE_IN + tot_out * _LITE_OUT
    flash_cost = tot_in * _FLASH_IN + tot_out * _FLASH_OUT

    print("\n" + "=" * 72)
    print(f"UIDs run: {len(rows)} | scored (gold present): {len(scored)}")
    print(f"RECALL  micro (Σhit/Σgold) = {micro:.4f}   macro (mean per-UID) = {macro:.4f}")
    print(f"FILTER  mean pct_eliminated = {mean_elim:.1f}%")
    print(f"TOKENS  in={tot_in:,} out={tot_out:,}")
    print(f"COST    flash-lite ~${lite_cost:.2f}  (flash upper-bound ${flash_cost:.2f})")
    print(f"429s    {probe.count}")
    wall = time.perf_counter() - t0
    print(f"WALL    {wall:.1f}s ({wall / 60:.1f} min) | achieved {60 * tot_in / max(1e-6, wall) / 1e6:.2f}M in-tok/min")
    print("=" * 72)

    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "eval" / f"retrieve_recall_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[retrieve] per-UID report → {out_path}")


if __name__ == "__main__":
    main()
