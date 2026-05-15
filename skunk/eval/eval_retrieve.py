"""32-UID retrieval-only benchmark.

For each sampled UID:
  1. Plan the question via skunk.planner.plan (cached in --plan-cache JSONL).
  2. Walk the plan's ComputeNodes → enumerate every RetrieveBranch.
  3. Call retrieve_hierarchical on each branch (in parallel within a UID).
  4. Aggregate the union of returned (bulletin, page) pairs as the prediction.
  5. Compare to the golden set from officeqa_pro.csv → recall + precision.

Tracks per-UID and aggregate:
  - LLM calls (planner vs retrieve)
  - input / output tokens
  - wall-clock latency (planning, retrieve, end-to-end)

Outputs a JSONL of per-UID results plus a printed summary table.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


_REPO_ROOT = Path(__file__).resolve().parent.parent
_load_env(_REPO_ROOT / ".env")
sys.path.insert(0, str(_REPO_ROOT / "src"))

from skunk.common import HarnessContext, LLMClient  # noqa: E402
from skunk.config import SkunkConfig  # noqa: E402
from skunk.dsl import Plan, RetrieveBranch, serialize  # noqa: E402
from skunk.page_index.concept_tree import load_catalog  # noqa: E402
from skunk.page_index.retrieve_probe import (  # noqa: E402
    agentic_traverse_retrieve, build_bucket_section_index, load_concept_tree,
    one_shot_parent_chapter_retrieve, one_shot_section_retrieve,
    retrieve_hierarchical, single_bucket_retrieve, write_trace_jsonl,
)


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
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        out[d["uid"]] = d["plan_text"]
    return out


def _append_plan_cache(path: Path, uid: str, question: str, plan_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps({"uid": uid, "question": question, "plan_text": plan_text}))
        f.write("\n")


def _plan_question(uid: str, question: str, llm: LLMClient,
                   plan_cache: dict[str, str], cache_path: Path,
                   stats: TokenStats) -> str | None:
    """Return plan_text, caching new plans. Accumulate token stats."""
    from skunk.planner import plan as planner_plan
    from skunk.dsl import serialize as dsl_serialize

    if uid in plan_cache:
        return plan_cache[uid]

    cfg = SkunkConfig.from_env()
    ctx = HarnessContext(question=question, verbose=False, config=cfg, llm_client=llm)
    try:
        plan_obj = planner_plan(question, ctx)
    except Exception as e:
        return None
    finally:
        for ev in ctx.events:
            if ev.get("source") == "llm" and ev.get("message") == "call":
                stats.add(ev.get("input_tokens"), ev.get("output_tokens"),
                          ev.get("latency_s"))
    plan_text = dsl_serialize(plan_obj)
    _append_plan_cache(cache_path, uid, question, plan_text)
    plan_cache[uid] = plan_text
    return plan_text


def _retrieve_branches(plan_obj: Plan) -> list[RetrieveBranch]:
    branches: list[RetrieveBranch] = []
    for compute in plan_obj.computes:
        for b in compute.branches:
            if isinstance(b, RetrieveBranch):
                branches.append(b)
    return branches


def _run_one_uid(
    uid: str, question: str, goldens: list[tuple[str, int]],
    indexed_pages: set[tuple[str, int]],
    plan_cache: dict[str, str], plan_cache_path: Path,
    tree: dict, catalog_index: dict, llm: LLMClient, retrieve_workers: int,
    trace_path: Path | None = None,
    skip_leaf_rank: bool = False,
    retriever: str = "hierarchical",
    cell_index: dict | None = None,
) -> UIDResult:
    from skunk.dsl import parse as dsl_parse

    t0 = time.monotonic()
    result = UIDResult(uid=uid, question=question, plan_text="",
                       goldens=goldens,
                       goldens_in_index=[g for g in goldens if g in indexed_pages],
                       predicted=[], n_retrieve_branches=0)

    plan_text = _plan_question(uid, question, llm, plan_cache, plan_cache_path,
                               result.planner)
    if plan_text is None:
        result.error = "plan failed"
        result.wall_s = time.monotonic() - t0
        return result
    result.plan_text = plan_text

    try:
        plan_obj = dsl_parse(plan_text)
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
                tree, question=question, concept=b.concept, period=b.period,
                llm=llm, catalog_index=catalog_index,
                uid=uid, retrieve_idx=idx,
            )
        elif retriever == "one-shot-section":
            top, trace = one_shot_section_retrieve(
                tree, question=question, concept=b.concept, period=b.period,
                llm=llm, catalog_index=catalog_index,
                uid=uid, retrieve_idx=idx,
            )
        elif retriever == "agentic-traverse":
            top, trace = agentic_traverse_retrieve(
                tree, question=question, concept=b.concept, period=b.period,
                llm=llm, catalog_index=catalog_index,
                cell_index=cell_index or {},
                uid=uid, retrieve_idx=idx,
            )
        elif retriever == "single-bucket":
            top, trace = single_bucket_retrieve(
                tree, question=question, concept=b.concept, period=b.period,
                llm=llm, catalog_index=catalog_index,
                uid=uid, retrieve_idx=idx,
            )
        else:
            top, trace = retrieve_hierarchical(
                tree, question=question, concept=b.concept, period=b.period,
                llm=llm, catalog_index=catalog_index,
                uid=uid, retrieve_idx=idx,
                skip_leaf_rank=skip_leaf_rank,
            )
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
    ap.add_argument("--catalog-dir", type=Path, default=_REPO_ROOT / "cache/page_index")
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
                    help="Override tree file location "
                         "(default: <catalog-dir>/concept_tree.json).")
    ap.add_argument("--skip-leaf-rank", action="store_true",
                    help="Return all (bulletin, page) keys under picked "
                         "clusters as the prediction; skip the L3 LLM "
                         "rerank. Measures the navigator's recall ceiling.")
    ap.add_argument("--retriever",
                    choices=("hierarchical", "single-bucket",
                             "agentic-traverse", "one-shot-section",
                             "one-shot-parent-chapter"),
                    default="hierarchical",
                    help="hierarchical: section_pick → cluster_pick → "
                         "leaf_rank (default). single-bucket: section_pick "
                         "then an agentic open/commit loop over (section, "
                         "cluster) leaves. agentic-traverse: open/commit "
                         "loop over (time_bucket, section) cells directly, "
                         "no section_pick prefix; predicted = all pages in "
                         "the committed cell.")
    ap.add_argument("--out", type=Path,
                    default=_REPO_ROOT / "cache/retrieve_bench_results.jsonl")
    ap.add_argument("--include-test-set", action="store_true",
                    help="Include held-out test UIDs (see CLAUDE.md). Default "
                         "excludes them. Required for any --uids that names "
                         "a test UID.")
    ap.add_argument("--trace-out", type=Path,
                    default=_REPO_ROOT / "cache/retrieve_bench_traces.jsonl",
                    help="Per-retrieve-call trace JSONL for offline analysis.")
    args = ap.parse_args()
    # Clear prior trace from this benchmark file.
    if args.trace_out.exists():
        args.trace_out.unlink()

    # Load corpus tree + indexed page set (for in-index recall).
    tree_path = args.tree_path or (args.catalog_dir / "concept_tree.json")
    tree = load_concept_tree(tree_path)
    indexed: set[tuple[str, int]] = set()
    for sec, sdata in tree.get("sections", {}).items():
        for cluster, cdata in sdata.get("clusters", {}).items():
            for kw, postings in cdata.get("keywords", {}).items():
                for p in postings:
                    indexed.add((p["bulletin"], p["page"]))
    print(f"Tree loaded from {tree_path}: "
          f"{len(tree.get('sections', {}))} sections, "
          f"{len(indexed)} indexed pages")

    catalog_rows = load_catalog(args.catalog_dir)
    catalog_index = {(r.bulletin, r.page): r for r in catalog_rows}
    print(f"Catalog index: {len(catalog_index)} (bulletin,page) entries")

    # For the agentic-traverse retriever: build the (bucket, section) cell
    # index once using the tree's banner_rewrite for canonical section names.
    cell_index: dict[tuple[str, str], list[tuple[str, int]]] = {}
    if args.retriever == "agentic-traverse":
        banner_rewrite = tree.get("meta", {}).get("banner_rewrite") or {}
        cell_index = build_bucket_section_index(
            catalog_rows, banner_rewrite,
        )
        print(f"Cell index: {len(cell_index)} non-empty (bucket, section) cells")

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
                      args.skip_leaf_rank, args.retriever,
                      cell_index): r["uid"]
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
