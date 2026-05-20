"""Page-index report harness — full build-and-query measurement.

End-to-end report on the page-index retriever:

  - Build: read `<page-index>/build_stats.json` written by the pipeline
    (per-stage wall time + token usage + index-structure stats).
  - Retrieve: walk the 101-UID dev set through the production retrieve
    path (L1 chapter pick + year-window filter) and capture per-step
    recall and elimination rate.
  - BM25 rank-only: for every golden page in a branch's year-filtered
    candidate set, record its rank BEFORE and AFTER BM25 ordering
    (no `top_k` truncation — pure ranking quality probe).
  - Cost / latency: tokens + wall-clock on the retrieve path
    (planner + L1 chapter pick), per branch and per UID.

Outputs land under `--output-dir`:

    report.md              top-line summary
    build_stats.md         render of build_stats.json
    retrieve_stats.md      per-step recall + elimination tables
    bm25_rank.md           golden rank distribution shift
    latency_cost.md        end-to-end cost breakdown
    retrieve_results.jsonl raw per-UID rows
    bm25_rank_rows.jsonl   raw per-(branch,golden) rank rows
    build_stats.json       copy of the build's build_stats.json

The held-out test set (`eval/test_set_uids.json`) is filtered out by
default — pass `--include-test-set` only for a deliberate final
measurement. Per CLAUDE.md, contaminating the dev set against the test
UIDs invalidates every downstream tuning decision.

Usage
-----
  python -m eval.report_page_index \\
      --page-index-dir cache/page_index_report \\
      --output-dir eval/reports/page_index_<ts>/

"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


# Load .env before importing skunk so LLMClient sees the API keys.
_load_env(REPO_ROOT / ".env")

from skunk.common import LLMClient  # noqa: E402
from skunk.config import SkunkConfig  # noqa: E402
from skunk.models import HarnessContext  # noqa: E402
from skunk.page_index import default_profile  # noqa: E402
from skunk.page_index.bm25 import Bm25Index, tokenize  # noqa: E402
from skunk.page_index.bm25_runtime import build_chapter_index  # noqa: E402
from skunk.page_index.retrieve_probe import (  # noqa: E402
    load_catalog, load_concept_tree, one_shot_parent_chapter_retrieve,
)
from skunk.plan import Plan, PlannerPromptedCall, RetrieveBranch  # noqa: E402
from skunk.prompt_overrides import load_prompt_overrides  # noqa: E402


_PERIOD = default_profile().period_parser

# Held-out test set — see CLAUDE.md 🚨 banner. Required at module-import time
# so the filter is provably wired into any code path that might run an LLM
# call. Search this file for `test_set_uids` to satisfy the doc invariant.
_TEST_SET_PATH = REPO_ROOT / "eval" / "test_set_uids.json"


def _load_test_set_uids() -> set[str]:
    if not _TEST_SET_PATH.exists():
        return set()
    return set(json.loads(_TEST_SET_PATH.read_text()).get("uids", []))


# ---------------------------------------------------------------------------
# Benchmark loader (mirrors eval/eval_e2e.py:_parse_source_docs)
# ---------------------------------------------------------------------------

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
}

_URL_RE = re.compile(
    r"/(?P<month>january|february|march|april|may|june|july|august|september|october|november|december)"
    r"-(?P<year>\d{4})[^?]*\?page=(?P<page>\d+)",
    re.IGNORECASE,
)


def _parse_source_doc(url: str) -> tuple[str, int] | None:
    m = _URL_RE.search(url)
    if not m:
        return None
    mn = _MONTHS.get(m.group("month").lower())
    if mn is None:
        return None
    return (f"{m.group('year')}-{mn:02d}", int(m.group("page")))


def _load_benchmark(csv_path: Path) -> list[dict]:
    out: list[dict] = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            goldens: list[tuple[str, int]] = []
            for url in (row.get("source_docs") or "").splitlines():
                ref = _parse_source_doc(url.strip())
                if ref:
                    goldens.append(ref)
            out.append({"uid": row["uid"], "question": row["question"],
                        "goldens": goldens})
    return out


# ---------------------------------------------------------------------------
# Plan cache (reused from the previous retrieve-bench cycle)
# ---------------------------------------------------------------------------

def _load_plan_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        out[d["uid"]] = d["plan_json"]
    return out


def _append_plan_cache(path: Path, uid: str, question: str,
                       plan_json: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps({"uid": uid, "question": question,
                            "plan_json": plan_json}))
        f.write("\n")


# ---------------------------------------------------------------------------
# Per-UID work
# ---------------------------------------------------------------------------

@dataclass
class TokenStats:
    n_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0

    def add(self, in_t: int | None, out_t: int | None,
            lat_s: float | None) -> None:
        self.n_calls += 1
        self.input_tokens += int(in_t or 0)
        self.output_tokens += int(out_t or 0)
        self.latency_s += float(lat_s or 0.0)


@dataclass
class BranchStep:
    """Per-step counts for one RetrieveBranch's walk through the pipeline.
    All sizes are page counts at each gate."""
    branch_idx: int
    key: str
    period: str | None
    picked_chapter: str | None
    after_chapter_pick: int       # chapter_top — pages in the picked chapter
    after_year_filter: int        # year-filtered survivors
    period_parseable: bool
    # Goldens at each gate (intersection with this branch's filter set).
    goldens_in_branch: list[tuple[str, int]] = field(default_factory=list)
    goldens_after_chapter: list[tuple[str, int]] = field(default_factory=list)
    goldens_after_year: list[tuple[str, int]] = field(default_factory=list)
    # Branch's predicted page set (after year filter). Materialized so the
    # UID-level union can compute recall + precision + elimination on the
    # real predicted set. ~3k pages per branch × ~2 branches × 101 UIDs
    # is well under 1 MB.
    predicted_pages: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class UIDResult:
    uid: str
    question: str
    goldens: list[tuple[str, int]]
    goldens_in_index: list[tuple[str, int]]
    branches: list[BranchStep]
    predicted: list[tuple[str, int]]   # union over branches, post year-filter
    planner: TokenStats = field(default_factory=TokenStats)
    retrieve: TokenStats = field(default_factory=TokenStats)
    wall_s: float = 0.0
    error: str | None = None

    @property
    def hits(self) -> list[tuple[str, int]]:
        pred = set(self.predicted)
        return [g for g in self.goldens if g in pred]


@dataclass
class GoldenRankRow:
    """One (branch, golden) row. `pre_rank` is the golden's position in the
    year-filtered list as the production path returns it (chapter-page
    enumeration order, which is effectively chronological).
    `post_rank` is the position after BM25 score sort, descending."""
    uid: str
    branch_idx: int
    chapter: str
    period: str | None
    golden: tuple[str, int]
    pre_rank: int                # 1-based, in input order
    post_rank: int               # 1-based, BM25-sort canonical (strictly-higher count + 1)
    n_candidates: int
    bm25_score: float
    unrankable: bool             # bm25_score == 0


# Process-wide BM25 cache: built once per (catalog_dir, chapter).
_BM25_CACHE: dict[tuple[Path, str], Bm25Index] = {}
_BM25_CACHE_LOCK = threading.Lock()
_BM25_CHAPTER_LOCKS: dict[tuple[Path, str], threading.Lock] = {}


def _bm25_for(chapter: str, tree: dict, catalog_index: dict,
              catalog_dir: Path) -> Bm25Index:
    key = (catalog_dir.resolve(), chapter)
    cached = _BM25_CACHE.get(key)
    if cached is not None:
        return cached
    with _BM25_CACHE_LOCK:
        lock = _BM25_CHAPTER_LOCKS.setdefault(key, threading.Lock())
    with lock:
        cached = _BM25_CACHE.get(key)
        if cached is not None:
            return cached
        pages = tree.get("chapters", {}).get(chapter, {}).get("pages", [])
        idx = build_chapter_index(pages, catalog_index)
        _BM25_CACHE[key] = idx
        return idx


def _plan_question(
    uid: str, question: str, llm: LLMClient, ctx_cfg: SkunkConfig,
    prompt_overrides, plan_cache: dict[str, str], plan_cache_path: Path,
    stats: TokenStats, cache_lock: threading.Lock,
) -> str | None:
    if uid in plan_cache:
        return plan_cache[uid]

    ctx = HarnessContext(
        question=question, verbose=False, config=ctx_cfg,
        llm_client=llm, prompt_overrides=prompt_overrides,
    )
    try:
        plan_obj = PlannerPromptedCall().plan(question, ctx)
    except Exception:
        # Drain planner-side LLM events before bailing so we still see
        # tokens for the failed attempt.
        for ev in ctx.events:
            if ev.get("source") == "llm" and ev.get("message") == "call":
                stats.add(ev.get("input_tokens"), ev.get("output_tokens"),
                          ev.get("latency_s"))
        return None
    for ev in ctx.events:
        if ev.get("source") == "llm" and ev.get("message") == "call":
            stats.add(ev.get("input_tokens"), ev.get("output_tokens"),
                      ev.get("latency_s"))
    plan_json = plan_obj.model_dump_json()
    with cache_lock:
        _append_plan_cache(plan_cache_path, uid, question, plan_json)
        plan_cache[uid] = plan_json
    return plan_json


def _year_filter(
    chapter_top: list[dict], catalog_index: dict, period: str | None,
) -> tuple[list[dict], bool]:
    """Same date-filter the production path applies. Returns (filtered,
    period_parseable). `period_parseable=False` means the filter was a
    no-op (no ISO interval resolvable from the period string)."""
    period_intervals = _PERIOD.intervals(period)
    if not period_intervals:
        return list(chapter_top), False
    kept = []
    for c in chapter_top:
        row = catalog_index.get((c["bulletin"], int(c["page"])))
        if row is None or not row.dates:
            kept.append(c)
            continue
        if _PERIOD.dates_overlap_period(row.dates, period_intervals):
            kept.append(c)
    return kept, True


def _bm25_rank_rows(
    uid: str, branch_idx: int, chapter: str, period: str | None,
    question: str, key: str, year_filtered: list[dict],
    goldens_in_set: set[tuple[str, int]],
    tree: dict, catalog_index: dict, catalog_dir: Path,
) -> list[GoldenRankRow]:
    """Score every page in `year_filtered` with the BM25 index for
    `chapter` and emit one row per golden in the candidate set.

    `pre_rank` is the 1-based position of the golden in `year_filtered`
    as returned by the production path (preserving input order).
    `post_rank` is the canonical 1-based BM25 rank: count of pages with
    a strictly higher score, plus one. Ties are broken canonically
    rather than by stable-sort so the number is reproducible across
    runs without dependence on input order.
    """
    if not year_filtered or not goldens_in_set:
        return []
    pages: list[tuple[str, int]] = [
        (c["bulletin"], int(c["page"])) for c in year_filtered
    ]
    pre_rank_of: dict[tuple[str, int], int] = {
        p: i + 1 for i, p in enumerate(pages)
    }

    index = _bm25_for(chapter, tree, catalog_index, catalog_dir)
    q_toks = tokenize(question) + tokenize(key) + tokenize(key)
    score_all = index.score(q_toks)
    scores = [score_all.get(p, 0.0) for p in pages]
    N = len(pages)

    out: list[GoldenRankRow] = []
    for g in goldens_in_set:
        if g not in pre_rank_of:
            continue
        s_g = score_all.get(g, 0.0)
        # Canonical rank: strictly-higher count + 1.
        higher = sum(1 for s in scores if s > s_g)
        post_rank = higher + 1
        out.append(GoldenRankRow(
            uid=uid, branch_idx=branch_idx, chapter=chapter, period=period,
            golden=g, pre_rank=pre_rank_of[g], post_rank=post_rank,
            n_candidates=N, bm25_score=round(float(s_g), 6),
            unrankable=(s_g == 0.0),
        ))
    return out


def _run_one_uid(
    rec: dict, indexed: set[tuple[str, int]], plan_cache: dict[str, str],
    plan_cache_path: Path, cache_lock: threading.Lock,
    tree: dict, catalog_index: dict, catalog_dir: Path,
    llm: LLMClient, cfg: SkunkConfig, prompt_overrides,
    retrieve_workers: int,
) -> tuple[UIDResult, list[GoldenRankRow]]:
    t0 = time.monotonic()
    uid, question, goldens = rec["uid"], rec["question"], rec["goldens"]
    result = UIDResult(
        uid=uid, question=question, goldens=goldens,
        goldens_in_index=[g for g in goldens if g in indexed],
        branches=[], predicted=[],
    )
    plan_json = _plan_question(
        uid, question, llm, cfg, prompt_overrides,
        plan_cache, plan_cache_path, result.planner, cache_lock,
    )
    if plan_json is None:
        result.error = "plan failed"
        result.wall_s = time.monotonic() - t0
        return result, []
    try:
        plan_obj = Plan.model_validate_json(plan_json)
    except Exception as e:
        result.error = f"plan parse: {type(e).__name__}: {e}"
        result.wall_s = time.monotonic() - t0
        return result, []

    retrieve_branches: list[tuple[int, RetrieveBranch]] = [
        (i, b) for i, b in enumerate(plan_obj.branches)
        if isinstance(b, RetrieveBranch)
    ]

    bm25_rows: list[GoldenRankRow] = []
    predicted: set[tuple[str, int]] = set()
    branch_steps: list[BranchStep] = []
    goldens_set = set(goldens)

    def _one_branch(idx: int, b: RetrieveBranch) -> tuple[
        BranchStep, list[GoldenRankRow], TokenStats,
    ]:
        chapter_top, trace = one_shot_parent_chapter_retrieve(
            tree, question=question, concept=b.key, period=b.period,
            llm=llm, catalog_index=catalog_index,
            uid=uid, retrieve_idx=idx,
        )
        filtered, parseable = _year_filter(chapter_top, catalog_index, b.period)
        picked = trace.picked_chapters[0] if trace.picked_chapters else None

        chap_set = {(c["bulletin"], int(c["page"])) for c in chapter_top}
        year_set = {(c["bulletin"], int(c["page"])) for c in filtered}
        step = BranchStep(
            branch_idx=idx, key=b.key, period=b.period,
            picked_chapter=picked,
            after_chapter_pick=len(chap_set),
            after_year_filter=len(year_set),
            period_parseable=parseable,
            goldens_in_branch=sorted(goldens_set),
            goldens_after_chapter=sorted(goldens_set & chap_set),
            goldens_after_year=sorted(goldens_set & year_set),
            predicted_pages=sorted(year_set),
        )
        # Tokens for this branch's LLM call.
        ts = TokenStats()
        for lvl in trace.levels:
            ts.add(lvl.input_tokens, lvl.output_tokens, lvl.latency_s)

        # BM25 rank-only probe (no top_k truncation). Only rows for goldens
        # actually in the year-filtered candidate set — anything earlier
        # was already eliminated.
        rows: list[GoldenRankRow] = []
        if picked and filtered:
            gold_in_year = goldens_set & year_set
            if gold_in_year:
                rows = _bm25_rank_rows(
                    uid=uid, branch_idx=idx, chapter=picked, period=b.period,
                    question=question, key=b.key, year_filtered=filtered,
                    goldens_in_set=gold_in_year,
                    tree=tree, catalog_index=catalog_index,
                    catalog_dir=catalog_dir,
                )
        return step, rows, ts

    if retrieve_branches:
        with ThreadPoolExecutor(max_workers=max(1, retrieve_workers)) as ex:
            futs = [ex.submit(_one_branch, i, b)
                    for i, b in retrieve_branches]
            for f in as_completed(futs):
                step, rows, ts = f.result()
                branch_steps.append(step)
                bm25_rows.extend(rows)
                # Production predicted = union of year-filtered per branch.
                predicted.update(step.predicted_pages)
                result.retrieve.n_calls += ts.n_calls
                result.retrieve.input_tokens += ts.input_tokens
                result.retrieve.output_tokens += ts.output_tokens
                result.retrieve.latency_s += ts.latency_s

    branch_steps.sort(key=lambda s: s.branch_idx)
    result.branches = branch_steps
    result.predicted = sorted(predicted)
    result.wall_s = time.monotonic() - t0
    return result, bm25_rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(results: list[UIDResult], indexed: set[tuple[str, int]],
               bm25_rows: list[GoldenRankRow]) -> dict:
    corpus_size = len(indexed)

    # Per-step recall: hit = at least one golden is in the surviving set
    # for the branch list of this UID. Micro-recall = hits / total goldens
    # across UIDs.
    total_g = 0
    total_g_in_idx = 0
    hits_chapter = 0
    hits_year = 0
    hits_pred = 0
    # Page-count distributions (across branches): aggregate sizes at each
    # gate so we can report elimination rates.
    chapter_sizes: list[int] = []
    year_sizes: list[int] = []
    branch_counts: list[int] = []
    pred_sizes_per_uid: list[int] = []   # UID-level predicted set size (post-union)
    n_uids_with_at_least_one_in_idx = 0

    for r in results:
        total_g += len(r.goldens)
        total_g_in_idx += len(r.goldens_in_index)
        if r.goldens_in_index:
            n_uids_with_at_least_one_in_idx += 1
        # Aggregate across branches for THIS UID.
        chap_union: set[tuple[str, int]] = set()
        year_union: set[tuple[str, int]] = set()
        for st in r.branches:
            chap_union.update(st.goldens_after_chapter)
            year_union.update(st.goldens_after_year)
            chapter_sizes.append(st.after_chapter_pick)
            year_sizes.append(st.after_year_filter)
        branch_counts.append(len(r.branches))
        hits_chapter += sum(1 for g in r.goldens if g in chap_union)
        hits_year += sum(1 for g in r.goldens if g in year_union)
        hits_pred += len(r.hits)
        pred_sizes_per_uid.append(len(r.predicted))

    n = len(results)

    def _stats(xs: list[int]) -> dict:
        if not xs:
            return {"n": 0, "mean": 0.0, "p50": 0, "p25": 0, "p75": 0,
                    "p95": 0, "min": 0, "max": 0}
        xs = sorted(xs)
        return {
            "n": len(xs),
            "mean": round(sum(xs) / len(xs), 1),
            "p25": xs[int(0.25 * (len(xs) - 1))],
            "p50": xs[int(0.50 * (len(xs) - 1))],
            "p75": xs[int(0.75 * (len(xs) - 1))],
            "p95": xs[int(0.95 * (len(xs) - 1))],
            "min": xs[0], "max": xs[-1],
        }

    def _elim(sizes: list[int]) -> float:
        if not sizes or corpus_size == 0:
            return 0.0
        kept = sum(sizes) / len(sizes)  # mean kept per branch
        return 1.0 - kept / corpus_size

    # BM25 rank stats: only rows where the golden is in the year-filtered
    # set. Split scored vs unrankable.
    scored_rows = [r for r in bm25_rows if not r.unrankable]
    pre_ranks = [r.pre_rank for r in scored_rows]
    post_ranks = [r.post_rank for r in scored_rows]
    pre_norm = [r.pre_rank / max(1, r.n_candidates) for r in scored_rows]
    post_norm = [r.post_rank / max(1, r.n_candidates) for r in scored_rows]
    rank_improvements = [r.pre_rank - r.post_rank for r in scored_rows]

    def _fstats(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0, "mean": 0.0, "p50": 0.0, "p25": 0.0, "p75": 0.0,
                    "p95": 0.0}
        return {
            "n": len(xs),
            "mean": round(statistics.mean(xs), 3),
            "p25": round(statistics.quantiles(xs, n=4)[0]
                         if len(xs) >= 4 else xs[len(xs) // 4], 3),
            "p50": round(statistics.median(xs), 3),
            "p75": round(statistics.quantiles(xs, n=4)[2]
                         if len(xs) >= 4 else xs[3 * len(xs) // 4], 3),
            "p95": round(sorted(xs)[int(0.95 * (len(xs) - 1))], 3),
        }

    # Cost / latency aggregation.
    tot_planner = TokenStats()
    tot_retrieve = TokenStats()
    per_branch_latency: list[float] = []
    wall_times: list[float] = []
    for r in results:
        tot_planner.n_calls += r.planner.n_calls
        tot_planner.input_tokens += r.planner.input_tokens
        tot_planner.output_tokens += r.planner.output_tokens
        tot_planner.latency_s += r.planner.latency_s
        tot_retrieve.n_calls += r.retrieve.n_calls
        tot_retrieve.input_tokens += r.retrieve.input_tokens
        tot_retrieve.output_tokens += r.retrieve.output_tokens
        tot_retrieve.latency_s += r.retrieve.latency_s
        wall_times.append(r.wall_s)
        if r.branches:
            per_branch_latency.append(r.retrieve.latency_s /
                                      max(1, r.retrieve.n_calls))

    def _dollars(t: TokenStats) -> float:
        # Indicative Flash rate: matches pipeline + previous bench harness.
        return t.input_tokens * 0.30e-6 + t.output_tokens * 2.50e-6

    return {
        "n_uids": n,
        "n_failed_plans": sum(1 for r in results if r.error),
        "corpus_size": corpus_size,
        "goldens": {
            "total": total_g,
            "in_index": total_g_in_idx,
            "in_index_pct": round(total_g_in_idx / max(1, total_g), 4),
        },
        "per_step_recall": {
            "after_chapter_pick": {
                "hits": hits_chapter,
                "recall_of_total": round(hits_chapter / max(1, total_g), 4),
                "recall_of_in_index": round(
                    hits_chapter / max(1, total_g_in_idx), 4),
            },
            "after_year_filter": {
                "hits": hits_year,
                "recall_of_total": round(hits_year / max(1, total_g), 4),
                "recall_of_in_index": round(
                    hits_year / max(1, total_g_in_idx), 4),
            },
            "end_to_end_predicted": {
                "hits": hits_pred,
                "recall_of_total": round(hits_pred / max(1, total_g), 4),
                "recall_of_in_index": round(
                    hits_pred / max(1, total_g_in_idx), 4),
            },
        },
        "page_counts": {
            "chapter_size": _stats(chapter_sizes),
            "year_filtered_size": _stats(year_sizes),
            "predicted_per_uid": _stats(pred_sizes_per_uid),
            "n_branches_per_uid": _stats(branch_counts),
        },
        "elimination_rate": {
            "after_chapter_pick": round(_elim(chapter_sizes), 4),
            "after_year_filter": round(_elim(year_sizes), 4),
            "end_to_end_per_uid": (
                round(1.0 - (sum(pred_sizes_per_uid) /
                             len(pred_sizes_per_uid)) / max(1, corpus_size), 4)
                if pred_sizes_per_uid else 0.0),
        },
        "precision": {
            "micro_after_year_filter": round(
                hits_pred / max(1, sum(pred_sizes_per_uid)), 6),
        },
        "bm25_rank": {
            "n_rows_total": len(bm25_rows),
            "n_unrankable": len(bm25_rows) - len(scored_rows),
            "pre_rank": _fstats([float(x) for x in pre_ranks]),
            "post_rank": _fstats([float(x) for x in post_ranks]),
            "pre_norm_rank": _fstats(pre_norm),
            "post_norm_rank": _fstats(post_norm),
            "rank_improvement": _fstats([float(x) for x in rank_improvements]),
            "n_improved": sum(1 for x in rank_improvements if x > 0),
            "n_worsened": sum(1 for x in rank_improvements if x < 0),
            "n_unchanged": sum(1 for x in rank_improvements if x == 0),
        },
        "cost_latency": {
            "planner": {
                "calls": tot_planner.n_calls,
                "input_tokens": tot_planner.input_tokens,
                "output_tokens": tot_planner.output_tokens,
                "approx_cost_usd": round(_dollars(tot_planner), 4),
                "sum_latency_s": round(tot_planner.latency_s, 2),
            },
            "retrieve": {
                "calls": tot_retrieve.n_calls,
                "input_tokens": tot_retrieve.input_tokens,
                "output_tokens": tot_retrieve.output_tokens,
                "approx_cost_usd": round(_dollars(tot_retrieve), 4),
                "sum_latency_s": round(tot_retrieve.latency_s, 2),
            },
            "total_approx_cost_usd": round(
                _dollars(tot_planner) + _dollars(tot_retrieve), 4),
            "per_uid_wall_s": _fstats(wall_times),
            "per_call_latency_s": _fstats(per_branch_latency),
        },
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _md_table(headers: list[str], rows: list[list]) -> str:
    out = "| " + " | ".join(headers) + " |\n"
    out += "| " + " | ".join("---" for _ in headers) + " |\n"
    for r in rows:
        out += "| " + " | ".join(str(c) for c in r) + " |\n"
    return out


def render_build_md(build_stats: dict) -> str:
    lines = ["# Build stats\n"]
    lines.append(f"- Built at: `{build_stats.get('built_at')}`")
    lines.append(f"- Git SHA: `{build_stats.get('git_sha')}`")
    lines.append(f"- Pipeline wall: **{build_stats['pipeline_wall_s']}s**")
    t = build_stats["totals"]
    lines.append(f"- Total LLM calls: **{t['n_llm_calls']:,}**  "
                 f"input={t['input_tokens']:,}  out={t['output_tokens']:,}  "
                 f"~**${t['approx_cost_usd']:.3f}**\n")

    lines.append("## Per-stage breakdown\n")
    rows = []
    for name, s in build_stats["stages"].items():
        rows.append([name, f"{s['wall_s']}s", s["n_llm_calls"],
                     f"{s['input_tokens']:,}", f"{s['output_tokens']:,}",
                     f"${s['approx_cost_usd']:.4f}", s["n_llm_errors"]])
    lines.append(_md_table(
        ["stage", "wall", "calls", "in_tokens", "out_tokens", "~cost",
         "errors"], rows))

    ix = build_stats["index_structure"]
    lines.append("\n## Index structure\n")
    lines.append(f"- Chapters: **{ix['n_chapters']}**  "
                 f"Indexed pages: **{ix['n_pages_indexed']:,}**  "
                 f"Bulletins: **{ix['n_bulletins']}**  "
                 f"Year range: {ix['year_range']}")
    lines.append(f"- Catalog rows: **{ix['catalog_rows']:,}**  "
                 f"(with content blocks: {ix['catalog_rows_with_content']:,}; "
                 f"year-envelope coverage: "
                 f"{ix['catalog_year_envelope_coverage']:.1%})")
    lines.append(f"- Chapter page-count distribution: "
                 f"min={ix['chapter_pages_min']}  p50={ix['chapter_pages_p50']}  "
                 f"mean={ix['chapter_pages_mean']}  max={ix['chapter_pages_max']}\n")
    lines.append("### Pages per chapter\n")
    rows = []
    for ch, n in ix["chapter_page_counts"].items():
        rows.append([ch, f"{n:,}", ix["chapter_examples_count"].get(ch, 0)])
    lines.append(_md_table(["chapter", "pages", "n_examples"], rows))

    if ix.get("placement_tier_counts"):
        lines.append("\n### Placement tier mix (place_pages)\n")
        tier_rows = sorted(ix["placement_tier_counts"].items(),
                           key=lambda x: -x[1])
        total = sum(n for _, n in tier_rows) or 1
        rows = [[k, f"{n:,}", f"{n/total:.1%}"] for k, n in tier_rows]
        lines.append(_md_table(["tier", "count", "%"], rows))

    return "\n".join(lines) + "\n"


def render_retrieve_md(agg: dict) -> str:
    lines = ["# Retrieve stats — per-step recall + elimination\n"]
    lines.append(f"- UIDs run: **{agg['n_uids']}**  "
                 f"(failed plans: {agg['n_failed_plans']})")
    lines.append(f"- Corpus size: **{agg['corpus_size']:,}** retrievable pages")
    lines.append(f"- Total goldens: **{agg['goldens']['total']}**  "
                 f"in-index: **{agg['goldens']['in_index']} "
                 f"({agg['goldens']['in_index_pct']:.1%})**\n")

    lines.append("## Recall at each gate (micro)\n")
    psr = agg["per_step_recall"]
    rows = [
        ["after chapter pick", psr["after_chapter_pick"]["hits"],
         f"{psr['after_chapter_pick']['recall_of_total']:.1%}",
         f"{psr['after_chapter_pick']['recall_of_in_index']:.1%}"],
        ["after year filter", psr["after_year_filter"]["hits"],
         f"{psr['after_year_filter']['recall_of_total']:.1%}",
         f"{psr['after_year_filter']['recall_of_in_index']:.1%}"],
        ["end-to-end predicted", psr["end_to_end_predicted"]["hits"],
         f"{psr['end_to_end_predicted']['recall_of_total']:.1%}",
         f"{psr['end_to_end_predicted']['recall_of_in_index']:.1%}"],
    ]
    lines.append(_md_table(
        ["gate", "hits", "recall (of all golden)",
         "recall (of in-index)"], rows))

    lines.append("\n## Page counts (per branch)\n")
    pc = agg["page_counts"]
    rows = [
        ["chapter size", pc["chapter_size"]["min"], pc["chapter_size"]["p25"],
         pc["chapter_size"]["p50"], pc["chapter_size"]["mean"],
         pc["chapter_size"]["p75"], pc["chapter_size"]["p95"],
         pc["chapter_size"]["max"]],
        ["year-filtered", pc["year_filtered_size"]["min"],
         pc["year_filtered_size"]["p25"], pc["year_filtered_size"]["p50"],
         pc["year_filtered_size"]["mean"], pc["year_filtered_size"]["p75"],
         pc["year_filtered_size"]["p95"], pc["year_filtered_size"]["max"]],
    ]
    lines.append(_md_table(
        ["gate", "min", "p25", "p50", "mean", "p75", "p95", "max"], rows))

    lines.append("\n## Elimination rate (1 − kept / corpus)\n")
    er = agg["elimination_rate"]
    lines.append(f"- After chapter pick (per branch): **{er['after_chapter_pick']:.2%}**")
    lines.append(f"- After year filter (per branch): **{er['after_year_filter']:.2%}**")
    lines.append(f"- End-to-end (per-UID predicted union): "
                 f"**{er['end_to_end_per_uid']:.2%}**")
    pp = agg["page_counts"]["predicted_per_uid"]
    lines.append(f"- Predicted per UID: mean={pp['mean']}  p50={pp['p50']}  "
                 f"p95={pp['p95']}  max={pp['max']}")
    p = agg["precision"]
    lines.append(f"- Micro-precision (hits / sum predicted): "
                 f"**{p['micro_after_year_filter']:.4%}**\n")

    lines.append(f"### Branches per UID\n")
    nb = agg["page_counts"]["n_branches_per_uid"]
    lines.append(
        f"- mean={nb['mean']}  p50={nb['p50']}  max={nb['max']}\n")
    return "\n".join(lines) + "\n"


def render_bm25_md(agg: dict) -> str:
    b = agg["bm25_rank"]
    lines = ["# BM25 rank-only probe (no top_k truncation)\n"]
    lines.append(
        "Rank of each golden in its branch's year-filtered candidate set, "
        "before vs after BM25 score-descending sort. `post_rank` is the "
        "canonical 1-based rank (strictly-higher count + 1).\n")
    lines.append(f"- Golden rank rows: **{b['n_rows_total']}** "
                 f"(unrankable / BM25 score = 0: **{b['n_unrankable']}**)")
    lines.append(f"- improved: **{b['n_improved']}**  "
                 f"worsened: **{b['n_worsened']}**  "
                 f"unchanged: **{b['n_unchanged']}**\n")

    rows = [
        ["pre_rank (1-based)", b["pre_rank"]["mean"], b["pre_rank"]["p25"],
         b["pre_rank"]["p50"], b["pre_rank"]["p75"], b["pre_rank"]["p95"]],
        ["post_rank (1-based)", b["post_rank"]["mean"], b["post_rank"]["p25"],
         b["post_rank"]["p50"], b["post_rank"]["p75"], b["post_rank"]["p95"]],
        ["pre_norm_rank (rank/N)", b["pre_norm_rank"]["mean"],
         b["pre_norm_rank"]["p25"], b["pre_norm_rank"]["p50"],
         b["pre_norm_rank"]["p75"], b["pre_norm_rank"]["p95"]],
        ["post_norm_rank (rank/N)", b["post_norm_rank"]["mean"],
         b["post_norm_rank"]["p25"], b["post_norm_rank"]["p50"],
         b["post_norm_rank"]["p75"], b["post_norm_rank"]["p95"]],
        ["rank_improvement (pre−post)", b["rank_improvement"]["mean"],
         b["rank_improvement"]["p25"], b["rank_improvement"]["p50"],
         b["rank_improvement"]["p75"], b["rank_improvement"]["p95"]],
    ]
    lines.append(_md_table(
        ["metric", "mean", "p25", "p50", "p75", "p95"], rows))

    return "\n".join(lines) + "\n"


def render_latency_md(agg: dict) -> str:
    cl = agg["cost_latency"]
    lines = ["# Latency & cost — retrieval path\n"]
    lines.append(
        "Planner is one LLM call per UID; retrieve is one L1-chapter-pick "
        "call per `RetrieveBranch`. BM25 reranking is in-process (no LLM). "
        "Cost uses indicative Flash rates: $0.30 / M input, $2.50 / M output.\n")
    if cl["planner"]["calls"] == 0:
        lines.append(
            "_Planner shows 0 calls because every UID hit the on-disk plan "
            "cache (`cache/retrieve_bench_plans.jsonl`). Delete the cache "
            "or pass a fresh path to re-plan from scratch._\n")
    rows = [
        ["planner", cl["planner"]["calls"], f"{cl['planner']['input_tokens']:,}",
         f"{cl['planner']['output_tokens']:,}",
         f"${cl['planner']['approx_cost_usd']:.4f}",
         f"{cl['planner']['sum_latency_s']}s"],
        ["retrieve", cl["retrieve"]["calls"],
         f"{cl['retrieve']['input_tokens']:,}",
         f"{cl['retrieve']['output_tokens']:,}",
         f"${cl['retrieve']['approx_cost_usd']:.4f}",
         f"{cl['retrieve']['sum_latency_s']}s"],
    ]
    lines.append(_md_table(
        ["component", "calls", "in_tokens", "out_tokens", "~cost",
         "sum_latency"], rows))
    lines.append(f"\nTotal approx cost across both: "
                 f"**${cl['total_approx_cost_usd']:.4f}**\n")
    wt = cl["per_uid_wall_s"]
    lines.append(f"\n## Per-UID wall clock\n")
    lines.append(f"- mean={wt['mean']}s  p50={wt['p50']}s  p75={wt['p75']}s  "
                 f"p95={wt['p95']}s")
    pl = cl["per_call_latency_s"]
    lines.append(f"\n## Per-branch LLM latency\n")
    lines.append(f"- mean={pl['mean']}s  p50={pl['p50']}s  p95={pl['p95']}s\n")
    return "\n".join(lines) + "\n"


def render_top_report(agg: dict, build_stats: dict, n_uids: int,
                      include_test_set: bool) -> str:
    psr = agg["per_step_recall"]
    er = agg["elimination_rate"]
    b = agg["bm25_rank"]
    cl = agg["cost_latency"]
    lines = ["# Page-index retrieval report — summary\n"]
    scope = ("dev + held-out test set" if include_test_set
             else "dev only (101 UIDs)")
    lines.append(f"- Eval scope: **{scope}**  ({n_uids} UIDs run)")
    lines.append(f"- Build artifact: `{build_stats.get('built_at')}` "
                 f"(git {build_stats.get('git_sha')})\n")

    lines.append("## Headline numbers\n")
    ix = build_stats["index_structure"]
    rows = [
        ["Chapters", f"{ix['n_chapters']}"],
        ["Indexed pages", f"{ix['n_pages_indexed']:,}"],
        ["Bulletins", f"{ix['n_bulletins']}"],
        ["Build wall-clock", f"{build_stats['pipeline_wall_s']}s"],
        ["Build LLM cost (~Flash)",
         f"${build_stats['totals']['approx_cost_usd']:.3f}"],
        ["Goldens in index",
         f"{agg['goldens']['in_index']}/{agg['goldens']['total']} "
         f"({agg['goldens']['in_index_pct']:.1%})"],
        ["Recall @ chapter-pick (of in-index)",
         f"{psr['after_chapter_pick']['recall_of_in_index']:.1%}"],
        ["Recall @ year-filter (of in-index)",
         f"{psr['after_year_filter']['recall_of_in_index']:.1%}"],
        ["Mean elimination @ year-filter",
         f"{er['after_year_filter']:.2%}"],
        ["BM25 rank improvement (mean, scored rows)",
         f"{b['rank_improvement']['mean']}"],
        ["BM25 normalized post-rank (p50)",
         f"{b['post_norm_rank']['p50']:.3f}  (vs pre {b['pre_norm_rank']['p50']:.3f})"],
        ["Retrieval cost across UIDs",
         f"${cl['total_approx_cost_usd']:.4f}"],
    ]
    lines.append(_md_table(["metric", "value"], rows))

    lines.append("\n## Sub-reports\n")
    for sub in ("build_stats.md", "retrieve_stats.md", "bm25_rank.md",
                "latency_cost.md"):
        lines.append(f"- [`{sub}`]({sub})")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    ap = argparse.ArgumentParser(
        description="Full page-index report (build + retrieve + BM25 + cost).")
    ap.add_argument("--page-index-dir", type=Path,
                    default=REPO_ROOT / "cache" / "page_index_report",
                    help="Page-index build dir containing concept_tree.json, "
                         "catalog/, manifest.json, build_stats.json.")
    ap.add_argument("--benchmark", type=Path,
                    default=REPO_ROOT / "data" / "officeqa_pro.csv")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Where to write report.md and sub-reports.")
    ap.add_argument("--plan-cache", type=Path,
                    default=REPO_ROOT / "cache"
                    / "retrieve_bench_plans.jsonl")
    ap.add_argument("--uids", type=str, default=None,
                    help="Comma-separated UIDs to run (debug; default: all "
                         "dev UIDs).")
    ap.add_argument("--sample", type=int, default=None,
                    help="Random sample N UIDs from the dev pool.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--uid-workers", type=int, default=4)
    ap.add_argument("--retrieve-workers", type=int, default=4)
    ap.add_argument("--include-test-set", action="store_true",
                    help="Include held-out test UIDs (see CLAUDE.md). "
                         "Default excludes them.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Build stats ------------------------------------------------------
    build_stats_path = args.page_index_dir / "build_stats.json"
    if not build_stats_path.exists():
        print(f"[ABORT] {build_stats_path} missing. Run pipeline.py with the "
              f"instrumented build first.", file=sys.stderr)
        return 2
    build_stats = json.loads(build_stats_path.read_text())
    shutil.copy(build_stats_path, args.output_dir / "build_stats.json")

    # --- Load tree + catalog ----------------------------------------------
    tree_path = args.page_index_dir / "concept_tree.json"
    tree = load_concept_tree(tree_path)
    indexed: set[tuple[str, int]] = set()
    for cdata in tree["chapters"].values():
        for p in cdata.get("pages", []):
            indexed.add((p["bulletin"], p["page"]))
    catalog_dir = args.page_index_dir / "catalog"
    rows = load_catalog(catalog_dir)
    catalog_index = {(r.bulletin, r.page): r for r in rows}
    print(f"Tree: {len(tree['chapters'])} chapters, {len(indexed)} pages")
    print(f"Catalog: {len(catalog_index)} rows")

    # --- Pick UIDs (with test-set filter) ---------------------------------
    test_uids = _load_test_set_uids()
    benchmark = _load_benchmark(args.benchmark)

    if args.uids:
        wanted = {u.strip() for u in args.uids.split(",") if u.strip()}
        if not args.include_test_set:
            collision = sorted(wanted & test_uids)
            if collision:
                print(f"[ABORT] --uids names held-out test UIDs: "
                      f"{collision}. Pass --include-test-set to override "
                      f"(see CLAUDE.md).", file=sys.stderr)
                return 2
        sampled = [r for r in benchmark if r["uid"] in wanted]
        missing = wanted - {r["uid"] for r in sampled}
        if missing:
            print(f"[WARN] UIDs not in benchmark: {sorted(missing)}",
                  file=sys.stderr)
    else:
        eligible = benchmark
        if test_uids and not args.include_test_set:
            before = len(eligible)
            eligible = [r for r in eligible if r["uid"] not in test_uids]
            print(f"[report] Excluded {before - len(eligible)} held-out "
                  f"test UIDs (see CLAUDE.md). {len(eligible)} dev UIDs "
                  f"remain.", file=sys.stderr)
        if args.sample is not None:
            rng = random.Random(args.seed)
            sampled = rng.sample(eligible, min(args.sample, len(eligible)))
        else:
            sampled = eligible

    sampled.sort(key=lambda r: r["uid"])
    print(f"Running {len(sampled)} UID(s)\n")

    if args.include_test_set and test_uids:
        print(f"[report] WARNING: --include-test-set is ON; numbers "
              f"include the held-out {len(test_uids)}-UID test set. "
              f"Use only for final measurement.", file=sys.stderr)

    # --- Plan cache + LLM client ------------------------------------------
    plan_cache = _load_plan_cache(args.plan_cache)
    cache_lock = threading.Lock()
    print(f"Plan cache: {len(plan_cache)} prior entries\n")

    cfg = SkunkConfig.from_env()
    overrides_path = Path(cfg.prompt_overrides_path)
    prompt_overrides = (load_prompt_overrides(overrides_path)
                        if overrides_path.exists() else ())
    llm = LLMClient(cfg)
    print(f"Model: {cfg.llm_model}\n")

    # --- Run --------------------------------------------------------------
    results: list[UIDResult] = []
    all_bm25_rows: list[GoldenRankRow] = []
    t_bench = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, args.uid_workers)) as ex:
        futs = {
            ex.submit(_run_one_uid, r, indexed, plan_cache, args.plan_cache,
                      cache_lock, tree, catalog_index, catalog_dir, llm, cfg,
                      prompt_overrides, args.retrieve_workers): r["uid"]
            for r in sampled
        }
        for f in as_completed(futs):
            r, bm25_rows = f.result()
            results.append(r)
            all_bm25_rows.extend(bm25_rows)
            n_g = len(r.goldens_in_index)
            chap_hits = sum(
                1 for g in r.goldens
                if any(g in set(s.goldens_after_chapter) for s in r.branches))
            year_hits = sum(
                1 for g in r.goldens
                if any(g in set(s.goldens_after_year) for s in r.branches))
            print(f"  {r.uid}  br={len(r.branches)}  "
                  f"g={len(r.goldens)}({n_g} in idx)  "
                  f"chap={chap_hits}  year={year_hits}  "
                  f"wall={r.wall_s:.1f}s "
                  f"{'[ERR ' + r.error + ']' if r.error else ''}",
                  flush=True)
    bench_wall = time.monotonic() - t_bench
    results.sort(key=lambda r: r.uid)

    # --- Aggregate + persist ----------------------------------------------
    agg = _aggregate(results, indexed, all_bm25_rows)

    out = args.output_dir
    (out / "aggregate.json").write_text(json.dumps(agg, indent=2,
                                                   default=str))
    with (out / "retrieve_results.jsonl").open("w") as f:
        for r in results:
            d = asdict(r)
            f.write(json.dumps(d, default=str))
            f.write("\n")
    with (out / "bm25_rank_rows.jsonl").open("w") as f:
        for row in all_bm25_rows:
            f.write(json.dumps(asdict(row), default=str))
            f.write("\n")

    # Markdown sub-reports.
    (out / "build_stats.md").write_text(render_build_md(build_stats))
    (out / "retrieve_stats.md").write_text(render_retrieve_md(agg))
    (out / "bm25_rank.md").write_text(render_bm25_md(agg))
    (out / "latency_cost.md").write_text(
        render_latency_md(agg))
    (out / "report.md").write_text(render_top_report(
        agg, build_stats, len(results), args.include_test_set))

    print(f"\nBench wall: {bench_wall:.1f}s")
    print(f"Wrote {out}/report.md (+ build_stats.md, retrieve_stats.md, "
          f"bm25_rank.md, latency_cost.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
