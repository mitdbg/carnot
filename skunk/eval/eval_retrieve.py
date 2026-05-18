"""Retrieval-only benchmark over a dev sample of OfficeQA UIDs.

For each sampled UID:
  1. Plan the question via skunk.plan.PlannerExecutor (cached in --plan-cache JSONL).
  2. Walk the Plan's branches → enumerate every RetrieveBranch.
  3. Run the chosen `--retriever` on each branch (in parallel within a UID).
  4. Aggregate the union of returned (bulletin, page) pairs as the prediction.
  5. Compare to the golden set from officeqa_pro.csv → recall + precision.

Retriever modes:
  - `one-shot-parent-chapter` (default): L1 chapter pick only; returns
    every page in the picked chapter. Useful as a recall ceiling probe.
  - `one-shot-section`: alias for parent-chapter (back-compat).

(Vector / embedding-based retrievers were removed once we established
that they did not improve recall over LLM-only L1 + leaf-rank.)

Outputs a JSONL of per-UID results plus a printed summary table.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from skunk.common import HarnessContext, LLMClient, load_env_file  # noqa: E402
load_env_file(_REPO_ROOT / ".env")
from skunk.config import SkunkConfig  # noqa: E402
from skunk.plan import Plan, RetrieveBranch  # noqa: E402
from skunk.page_index import default_profile  # noqa: E402
_PERIOD_PARSER = default_profile().period_parser
period_year_window = _PERIOD_PARSER.year_window
from skunk.page_index.retrieve_probe import (  # noqa: E402
    load_catalog, load_concept_tree,
    one_shot_parent_chapter_retrieve, one_shot_section_retrieve,
    write_trace_jsonl,
)
from skunk.page_index.bm25 import Bm25Index  # noqa: E402
from skunk.page_index.bm25_runtime import bm25_rerank, build_chapter_index  # noqa: E402
import threading  # noqa: E402

# Process-wide BM25 cache: shared across all UIDs in a run so each
# chapter's index is built once. Built lazily under a per-chapter lock.
_BM25_CACHE: dict[str, Bm25Index] = {}
_BM25_CACHE_LOCK = threading.Lock()
_BM25_CHAPTER_LOCKS: dict[str, threading.Lock] = {}


def _bm25_get(chapter: str, tree: dict, catalog_index: dict) -> Bm25Index:
    cached = _BM25_CACHE.get(chapter)
    if cached is not None:
        return cached
    with _BM25_CACHE_LOCK:
        lock = _BM25_CHAPTER_LOCKS.setdefault(chapter, threading.Lock())
    with lock:
        cached = _BM25_CACHE.get(chapter)
        if cached is not None:
            return cached
        pages = tree.get("chapters", {}).get(chapter, {}).get("pages", [])
        idx = build_chapter_index(pages, catalog_index)
        _BM25_CACHE[chapter] = idx
        return idx


_MONTHS = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
           "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12}


def _parse_source_doc(url: str) -> tuple[str, int] | None:
    m = re.search(r"/([a-z]+)-(\d{4})-\d+\?page=(\d+)", url)
    if not m:
        return None
    mn = _MONTHS.get(m.group(1).lower())
    if mn is None:
        return None
    return (f"{m.group(2)}-{mn:02d}", int(m.group(3)))


def _load_benchmark(csv_path: Path) -> list[dict]:
    out: list[dict] = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            uid = row["uid"]
            q = row["question"]
            goldens: list[tuple[str, int]] = []
            for url in (row.get("source_docs") or "").splitlines():
                ref = _parse_source_doc(url.strip())
                if ref:
                    goldens.append(ref)
            out.append({"uid": uid, "question": q, "goldens": goldens})
    return out


@dataclass
class TokenStats:
    n_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0

    def add(self, in_t: int | None, out_t: int | None, lat_s: float | None) -> None:
        self.n_calls += 1
        self.input_tokens += int(in_t or 0)
        self.output_tokens += int(out_t or 0)
        self.latency_s += float(lat_s or 0.0)


@dataclass
class UIDResult:
    uid: str
    question: str
    plan_text: str
    goldens: list[tuple[str, int]]
    goldens_in_index: list[tuple[str, int]]
    predicted: list[tuple[str, int]]
    n_retrieve_branches: int
    planner: TokenStats = field(default_factory=TokenStats)
    retrieve: TokenStats = field(default_factory=TokenStats)
    wall_s: float = 0.0
    error: str | None = None

    @property
    def hits(self) -> list[tuple[str, int]]:
        pred = set(self.predicted)
        return [g for g in self.goldens if g in pred]

    @property
    def recall(self) -> float:
        return len(self.hits) / max(1, len(self.goldens))

    @property
    def recall_indexed(self) -> float:
        pred = set(self.predicted)
        if not self.goldens_in_index:
            return 0.0
        return sum(1 for g in self.goldens_in_index if g in pred) / len(self.goldens_in_index)

    @property
    def precision(self) -> float:
        return len(self.hits) / max(1, len(self.predicted))


def _load_plan_cache(path: Path) -> dict[str, str]:
    """Return {uid: plan_json_string}. One {uid, question, plan_json} per JSONL line."""
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


def _append_plan_cache(path: Path, uid: str, question: str, plan_json: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps({"uid": uid, "question": question, "plan_json": plan_json}))
        f.write("\n")


def _plan_question(uid: str, question: str, llm: LLMClient,
                   plan_cache: dict[str, str], cache_path: Path,
                   stats: TokenStats) -> str | None:
    """Return plan_json string, caching new plans. Accumulate token stats."""
    from skunk.plan import PlannerExecutor

    if uid in plan_cache:
        return plan_cache[uid]

    cfg = SkunkConfig.from_env()
    from skunk.prompt_overrides import load_prompt_overrides
    overrides_path = Path(cfg.prompt_overrides_path)
    prompt_overrides = load_prompt_overrides(overrides_path) if overrides_path.exists() else ()
    ctx = HarnessContext(question=question, verbose=False, config=cfg, llm_client=llm,
                         prompt_overrides=prompt_overrides)
    try:
        plan_obj = PlannerExecutor().plan(question, ctx)
    except Exception:
        return None
    finally:
        for ev in ctx.events:
            if ev.get("source") == "llm" and ev.get("message") == "call":
                stats.add(ev.get("input_tokens"), ev.get("output_tokens"),
                          ev.get("latency_s"))
    plan_json = plan_obj.model_dump_json()
    _append_plan_cache(cache_path, uid, question, plan_json)
    plan_cache[uid] = plan_json
    return plan_json


def _retrieve_branches(plan_obj: Plan) -> list[RetrieveBranch]:
    return [b for b in plan_obj.branches if isinstance(b, RetrieveBranch)]


def _run_one_uid(
    uid: str, question: str, goldens: list[tuple[str, int]],
    indexed_pages: set[tuple[str, int]],
    plan_cache: dict[str, str], plan_cache_path: Path,
    tree: dict, catalog_index: dict, llm: LLMClient, retrieve_workers: int,
    trace_path: Path | None = None,
    retriever: str = "one-shot-parent-chapter",
    bm25_top_k: int = 20,
    bm25_threshold: float = 2.0,
) -> UIDResult:
    t0 = time.monotonic()
    result = UIDResult(uid=uid, question=question, plan_text="",
                       goldens=goldens,
                       goldens_in_index=[g for g in goldens if g in indexed_pages],
                       predicted=[], n_retrieve_branches=0)

    plan_json = _plan_question(uid, question, llm, plan_cache, plan_cache_path,
                               result.planner)
    if plan_json is None:
        result.error = "plan failed"
        result.wall_s = time.monotonic() - t0
        return result
    result.plan_text = plan_json

    try:
        plan_obj = Plan.model_validate_json(plan_json)
    except Exception as e:
        result.error = f"plan parse: {type(e).__name__}: {e}"
        result.wall_s = time.monotonic() - t0
        return result

    branches = _retrieve_branches(plan_obj)
    result.n_retrieve_branches = len(branches)

    predicted: set[tuple[str, int]] = set()

    def _one_branch(idx: int, b: RetrieveBranch):
        if retriever == "one-shot-parent-chapter":
            top, trace = one_shot_parent_chapter_retrieve(
                tree, question=question, concept=b.key, period=b.period,
                llm=llm, catalog_index=catalog_index,
                uid=uid, retrieve_idx=idx,
            )
        elif retriever == "one-shot-section":
            top, trace = one_shot_section_retrieve(
                tree, question=question, concept=b.key, period=b.period,
                llm=llm, catalog_index=catalog_index,
                uid=uid, retrieve_idx=idx,
            )
        elif retriever in ("chapter-year", "chapter-year-bm25"):
            # chapter-year: production pipeline (mirrors src/skunk/retrieve.py).
            # chapter-year-bm25: same + experimental BM25 rerank scaffold.
            chapter_top, trace = one_shot_parent_chapter_retrieve(
                tree, question=question, concept=b.key, period=b.period,
                llm=llm, catalog_index=catalog_index,
                uid=uid, retrieve_idx=idx,
            )
            window = period_year_window(b.period)
            if window is None:
                filtered = chapter_top
            else:
                q_lo, q_hi = window
                filtered = []
                for c in chapter_top:
                    row = catalog_index.get((c["bulletin"], int(c["page"])))
                    if row is None or row.min_year is None or row.max_year is None:
                        filtered.append(c)
                    elif row.max_year >= q_lo and row.min_year <= q_hi:
                        filtered.append(c)
            if retriever == "chapter-year-bm25" and filtered and trace.picked_chapters:
                try:
                    index = _bm25_get(trace.picked_chapters[0], tree, catalog_index)
                    filtered, _bm25_meta = bm25_rerank(
                        filtered, index, question=question, key=b.key,
                        top_k=bm25_top_k, threshold=bm25_threshold,
                    )
                except Exception as e:  # scaffold safety: pass through on failure
                    print(f"  [bm25 error uid={uid} branch={idx}] "
                          f"{type(e).__name__}: {e}", flush=True)
            trace.candidate_count = len(filtered)
            trace.top_k = filtered[:50]
            top = filtered
        else:
            raise ValueError(f"unknown retriever: {retriever!r}")
        return top, trace

    if branches:
        with ThreadPoolExecutor(max_workers=max(1, retrieve_workers)) as ex:
            futs = [ex.submit(_one_branch, i, b) for i, b in enumerate(branches)]
            for f in as_completed(futs):
                top, trace = f.result()
                if trace_path is not None:
                    write_trace_jsonl(trace, trace_path)
                for r in top:
                    predicted.add((r["bulletin"], r["page"]))
                for lvl in trace.levels:
                    result.retrieve.add(lvl.input_tokens, lvl.output_tokens,
                                        lvl.latency_s)

    result.predicted = sorted(predicted)
    result.wall_s = time.monotonic() - t0
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="32-UID retrieval-only benchmark.")
    ap.add_argument("--benchmark", type=Path, default=_REPO_ROOT / "data/officeqa_pro.csv")
    ap.add_argument("--catalog-dir", type=Path,
                    default=_REPO_ROOT / "cache/page_index_v4/catalog",
                    help="Catalog jsonl directory. Defaults to the latest "
                         "build under cache/page_index_v4/.")
    ap.add_argument("--n", type=int, default=32, help="Number of UIDs to sample.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--uids", type=str, default=None,
                    help="Comma-separated UID list; overrides --n/--seed sampling.")
    ap.add_argument("--plan-cache", type=Path,
                    default=_REPO_ROOT / "cache/retrieve_bench_plans.jsonl",
                    help="JSONL cache: one {uid, question, plan_text} per line.")
    ap.add_argument("--uid-workers", type=int, default=4,
                    help="Parallel UIDs in flight.")
    ap.add_argument("--retrieve-workers", type=int, default=4,
                    help="Parallel retrieve branches within one UID.")
    ap.add_argument("--tree-path", type=Path, default=None,
                    help="Tree file path (default: shipped artifact at "
                         "data/page_index/concept_tree.json; falls back to "
                         "<catalog-dir>/concept_tree.json if --catalog-dir "
                         "is overridden).")
    ap.add_argument("--retriever",
                    choices=("one-shot-parent-chapter", "one-shot-section",
                             "chapter-year", "chapter-year-bm25"),
                    default="chapter-year",
                    help="chapter-year (default): production pipeline — "
                         "L1 chapter pick + year-window filter. Mirrors "
                         "src/skunk/retrieve.py. "
                         "chapter-year-bm25: experimental scaffold — "
                         "chapter-year + BM25 rerank with confidence-based "
                         "bypass (see src/skunk/page_index/bm25_runtime.py). "
                         "one-shot-parent-chapter: L1 chapter pick only "
                         "(returns every page in the picked chapter; recall "
                         "ceiling probe). "
                         "one-shot-section: alias for parent-chapter under "
                         "the flat-tree shape.")
    ap.add_argument("--out", type=Path,
                    default=_REPO_ROOT / "cache/retrieve_bench_results.jsonl")
    ap.add_argument("--include-test-set", action="store_true",
                    help="Include held-out test UIDs (see CLAUDE.md). Default "
                         "excludes them. Required for any --uids that names "
                         "a test UID.")
    ap.add_argument("--trace-out", type=Path,
                    default=_REPO_ROOT / "cache/retrieve_bench_traces.jsonl",
                    help="Per-retrieve-call trace JSONL for offline analysis.")
    ap.add_argument("--bm25-top-k", type=int, default=20,
                    help="(chapter-year-bm25 only) Truncate to this many pages "
                         "when BM25's top score clearly dominates the field.")
    ap.add_argument("--bm25-threshold", type=float, default=2.0,
                    help="(chapter-year-bm25 only) Dominance ratio "
                         "(top1 / median20) required to trigger truncation. "
                         "Below this, return all year-filtered survivors in "
                         "BM25 order without truncating.")
    args = ap.parse_args()
    # Clear prior trace from this benchmark file.
    if args.trace_out.exists():
        args.trace_out.unlink()

    # Load corpus tree + indexed page set (for in-index recall).
    # Default tree path follows the shipped artifact layout
    # (data/page_index/concept_tree.json sits next to catalog/). If
    # --catalog-dir was overridden but --tree-path wasn't, look for the
    # tree alongside the catalog dir's parent.
    default_artifact_tree = _REPO_ROOT / "data/page_index/concept_tree.json"
    if args.tree_path is not None:
        tree_path = args.tree_path
    elif args.catalog_dir == _REPO_ROOT / "data/page_index/catalog":
        tree_path = default_artifact_tree
    else:
        # Catalog dir overridden — try sibling locations.
        cand = args.catalog_dir.parent / "concept_tree.json"
        if cand.exists():
            tree_path = cand
        else:
            tree_path = args.catalog_dir / "concept_tree.json"
    tree = load_concept_tree(tree_path)
    indexed: set[tuple[str, int]] = set()
    if "chapters" in tree:
        # New flat shape (Phase 3 merge output)
        for chapter, cdata in tree["chapters"].items():
            for p in cdata.get("pages", []):
                indexed.add((p["bulletin"], p["page"]))
        n_top = len(tree["chapters"])
        top_label = "chapters"
    else:
        # Legacy nested shape (sections → clusters → keywords → postings)
        for sec, sdata in tree.get("sections", {}).items():
            for cluster, cdata in sdata.get("clusters", {}).items():
                for kw, postings in cdata.get("keywords", {}).items():
                    for p in postings:
                        indexed.add((p["bulletin"], p["page"]))
        n_top = len(tree.get("sections", {}))
        top_label = "sections"
    print(f"Tree loaded from {tree_path}: "
          f"{n_top} {top_label}, {len(indexed)} indexed pages")

    catalog_rows = load_catalog(args.catalog_dir)
    catalog_index = {(r.bulletin, r.page): r for r in catalog_rows}
    print(f"Catalog index: {len(catalog_index)} (bulletin,page) entries")

    # Load the held-out test set (see CLAUDE.md). Filter it out unless the
    # operator explicitly opts in via --include-test-set.
    test_set_path = _REPO_ROOT / "eval" / "test_set_uids.json"
    test_uids: set[str] = set()
    if test_set_path.exists():
        test_uids = set(json.loads(test_set_path.read_text()).get("uids", []))

    # Sample UIDs.
    benchmark = _load_benchmark(args.benchmark)
    if args.uids:
        wanted = {u.strip() for u in args.uids.split(",") if u.strip()}
        if not args.include_test_set:
            overlap = wanted & test_uids
            if overlap:
                print(f"[ABORT] --uids names held-out test UIDs: "
                      f"{sorted(overlap)}. Pass --include-test-set to "
                      f"override (see CLAUDE.md).", file=sys.stderr)
                return 2
        sampled = [r for r in benchmark if r["uid"] in wanted]
        missing = wanted - {r["uid"] for r in sampled}
        if missing:
            print(f"Requested UIDs not in benchmark: {sorted(missing)}",
                  file=sys.stderr)
        sampled.sort(key=lambda r: r["uid"])
        print(f"Running {len(sampled)} explicit UIDs\n")
    else:
        eligible = benchmark
        if not args.include_test_set and test_uids:
            before = len(eligible)
            eligible = [r for r in eligible if r["uid"] not in test_uids]
            print(f"[retrieve_eval] Excluded {before - len(eligible)} "
                  f"held-out test UIDs (see CLAUDE.md). {len(eligible)} "
                  f"dev UIDs remain.", file=sys.stderr)
        rng = random.Random(args.seed)
        sampled = rng.sample(eligible, min(args.n, len(eligible)))
        sampled.sort(key=lambda r: r["uid"])
        print(f"Sampled {len(sampled)} UIDs (seed={args.seed})\n")

    if args.include_test_set and test_uids:
        print(f"[retrieve_eval] WARNING: --include-test-set is on. Numbers "
              f"will include the held-out {len(test_uids)}-UID test set.",
              file=sys.stderr)

    plan_cache = _load_plan_cache(args.plan_cache)
    print(f"Plan cache: {len(plan_cache)} prior entries at {args.plan_cache}\n")

    cfg = SkunkConfig.from_env()
    print(f"Model: {cfg.llm_model}\n")
    llm = LLMClient(cfg)

    results: list[UIDResult] = []
    t_bench = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, args.uid_workers)) as ex:
        futs = {
            ex.submit(_run_one_uid, r["uid"], r["question"], r["goldens"],
                      indexed, plan_cache, args.plan_cache, tree,
                      catalog_index, llm,
                      args.retrieve_workers, args.trace_out,
                      args.retriever,
                      args.bm25_top_k, args.bm25_threshold): r["uid"]
            for r in sampled
        }
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            status = "ok" if r.error is None else f"ERR {r.error}"
            print(f"  {r.uid}  branches={r.n_retrieve_branches}  "
                  f"recall={r.recall:.0%}  P={r.precision:.0%}  "
                  f"wall={r.wall_s:.1f}s  [{status}]", flush=True)
    total_wall = time.monotonic() - t_bench

    results.sort(key=lambda r: r.uid)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in results:
            d = asdict(r)
            f.write(json.dumps(d, default=str))
            f.write("\n")

    # Aggregate summary
    print("\n" + "=" * 80)
    print("PER-UID RESULTS")
    print("=" * 80)
    print(f"{'UID':<10}{'br':>3}{'gold':>5}{'idx':>5}{'pred':>5}{'hit':>4}"
          f"{'recall':>8}{'r/idx':>7}{'P':>6}{'wall':>7}")
    for r in results:
        print(f"{r.uid:<10}{r.n_retrieve_branches:>3}"
              f"{len(r.goldens):>5}{len(r.goldens_in_index):>5}"
              f"{len(r.predicted):>5}{len(r.hits):>4}"
              f"{r.recall*100:>7.1f}%"
              f"{r.recall_indexed*100:>6.1f}%"
              f"{r.precision*100:>5.1f}%"
              f"{r.wall_s:>6.1f}s")

    # Totals
    tot_g = sum(len(r.goldens) for r in results)
    tot_gi = sum(len(r.goldens_in_index) for r in results)
    tot_p = sum(len(r.predicted) for r in results)
    tot_h = sum(len(r.hits) for r in results)
    tot_planner = TokenStats()
    tot_retr = TokenStats()
    for r in results:
        tot_planner.n_calls += r.planner.n_calls
        tot_planner.input_tokens += r.planner.input_tokens
        tot_planner.output_tokens += r.planner.output_tokens
        tot_planner.latency_s += r.planner.latency_s
        tot_retr.n_calls += r.retrieve.n_calls
        tot_retr.input_tokens += r.retrieve.input_tokens
        tot_retr.output_tokens += r.retrieve.output_tokens
        tot_retr.latency_s += r.retrieve.latency_s

    print("\n" + "=" * 80)
    print("AGGREGATE")
    print("=" * 80)
    print(f"UIDs: {len(results)}    failed plans: {sum(1 for r in results if r.error)}")
    print(f"Total goldens: {tot_g}    in-index: {tot_gi} ({tot_gi/max(1,tot_g):.1%})")
    print(f"Predicted pages: {tot_p}    Hits: {tot_h}")
    print(f"Micro-recall (hits / total_goldens):         {tot_h/max(1,tot_g):.1%}")
    print(f"Micro-recall_indexed (hits / in_index):       {tot_h/max(1,tot_gi):.1%}")
    print(f"Micro-precision (hits / predicted):           {tot_h/max(1,tot_p):.1%}")
    macro_r = sum(r.recall for r in results) / max(1, len(results))
    macro_p = sum(r.precision for r in results) / max(1, len(results))
    print(f"Macro-recall (mean per-UID):                 {macro_r:.1%}")
    print(f"Macro-precision (mean per-UID):              {macro_p:.1%}")

    # Elimination rate: fraction of the indexed corpus the retriever
    # filtered out per query, averaged across UIDs. corpus_size is the
    # number of retrievable pages in the index (the union of every
    # chapter's `pages`). A retriever returning N pages eliminates
    # 1 − N/corpus_size of the corpus for that query.
    corpus_size = len(indexed)
    per_uid_elim = [
        1.0 - (len(r.predicted) / corpus_size) for r in results if r.predicted
    ]
    mean_elim = (sum(per_uid_elim) / len(per_uid_elim)) if per_uid_elim else 0.0
    mean_pred = (sum(len(r.predicted) for r in results) /
                 max(1, len(results)))
    print(f"Indexed corpus size:                          {corpus_size}")
    print(f"Mean predicted pages / UID:                   {mean_pred:.1f}")
    print(f"Mean elimination rate:                        {mean_elim:.2%}")
    print()
    print("LLM cost breakdown:")
    print(f"  PLANNER:   calls={tot_planner.n_calls:>4}  "
          f"in={tot_planner.input_tokens:>9,}  out={tot_planner.output_tokens:>9,}  "
          f"sum_latency={tot_planner.latency_s:.1f}s")
    print(f"  RETRIEVE:  calls={tot_retr.n_calls:>4}  "
          f"in={tot_retr.input_tokens:>9,}  out={tot_retr.output_tokens:>9,}  "
          f"sum_latency={tot_retr.latency_s:.1f}s")
    # Standard Flash rate cost estimate (purely indicative).
    def _dollars(t: TokenStats) -> float:
        return t.input_tokens * 0.30e-6 + t.output_tokens * 2.50e-6
    print(f"  ~$ planner  {_dollars(tot_planner):.3f}")
    print(f"  ~$ retrieve {_dollars(tot_retr):.3f}")
    print(f"  ~$ total    {_dollars(tot_planner) + _dollars(tot_retr):.3f}")
    print()
    print(f"Wall clock end-to-end ({args.uid_workers} UID workers, "
          f"{args.retrieve_workers} retrieve workers): {total_wall:.1f}s")
    print(f"\nPer-UID results written to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
