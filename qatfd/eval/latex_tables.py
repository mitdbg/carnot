"""Generate the main-result LaTeX tables from ``results/``.

Layout (decided with the paper's main table in mind):

* **One table per benchmark** (OfficeQA, BrowseComp-Plus, TREC-BioGen,
  FinanceBench, QAMPARI, FreshStack).
* **Rows = (variant x LLM)**: RAG-LLM (k=10/100/1000) plus one row per SearchAgent
  retrieval tool set — ``SearchAgent`` (grep+vector) and ``QATFD`` (grep+sem) are two
  points in that lattice — each repeated per LLM it was run with (gemini-3.5-flash,
  opus-4.8, ...). See ``variants.py``: the SearchAgent variants are a single system
  configured by its tool flags, so the row identity is read from each run's
  ``config.yaml``, not from its results/ directory name.
* **Columns = metrics**: Accuracy, optional retrieval recall column(s), avg.
  per-question cost, avg. per-question latency.
* **Best per column is bolded** (highest for accuracy/recall, lowest for
  cost/latency).

How the numbers are computed
----------------------------
Each ``report.csv`` is one *run* (one full pass over the benchmark's questions).
For a run we take the per-question mean of each metric (``score`` -> accuracy,
``cost``, ``wall_s`` -> latency, and each per-benchmark recall column). When the
same (benchmark, variant, LLM, judge, top_k) config was run multiple times, we
**average across runs** (equal weight per run) and report the run count.

Run discovery walks ``results/<benchmark>/<system>/<run_name>_<ts>/`` and reads each
run's ``config.yaml`` to recover the variant, the agent LLM (``inference.llm_model``),
the semantic-filter judge model, and ``top_k``.

Usage
-----
    python qatfd/eval/latex_tables.py                 # all tables -> stdout
    python qatfd/eval/latex_tables.py --out tables.tex
    python qatfd/eval/latex_tables.py --benchmarks officeqa,qampari --no-recall
    python qatfd/eval/latex_tables.py --llm gemini-3.5-flash
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

# Sibling module: relative import as a package, plain import when run as a script.
try:
    from .variants import Variant, judge_of, llm_of, model_label, variant_of
except ImportError:
    from variants import Variant, judge_of, llm_of, model_label, variant_of

# --------------------------------------------------------------------------- #
# Display config
# --------------------------------------------------------------------------- #

# Benchmark dir-name -> (display name, latex label suffix). Order = table order.
BENCHMARKS: dict[str, str] = {
    "officeqa": "OfficeQA",
    "browsecomp_plus": "BrowseComp-Plus",
    "trec_biogen": "TREC-BioGen",
    "financebench": "FinanceBench",
    "qampari": "QAMPARI",
    "freshstack": "FreshStack",
}

# Row labels and their order come from `variants.py`: the SearchAgent variants are now one
# system distinguished by its tool flags, so a row's identity is read from each run's
# config.yaml rather than from its results/ directory name.

# Pretty headers for the dynamic per-benchmark recall columns. Order = column
# order in the table (fine-grained -> coarse-grained).
RECALL_DISPLAY: dict[str, str] = {
    "chunk_recall": "Chunk Rec.",
    "page_recall": "Page Rec.",
    "doc_recall": "Doc Rec.",
    "file_recall": "File Rec.",
    "gold_doc_recall": "Gold Doc Rec.",
    "evidence_doc_recall": "Ev. Doc Rec.",
}

_TS_RE = re.compile(r"_(\d{8}_\d{6})$")  # trailing _YYYYMMDD_HHMMSS in run dir names


# --------------------------------------------------------------------------- #
# Metric specs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Metric:
    """One reported column. ``higher_is_better`` drives which value gets bolded."""

    key: str  # report.csv column to aggregate
    header: str  # LaTeX column header
    higher_is_better: bool
    fmt: str  # python format applied to the (scaled) value
    scale: float = 1.0  # multiply the raw mean before formatting (e.g. acc -> %)


ACCURACY = Metric("score", r"Acc.\ (\%)", higher_is_better=True, fmt="{:.1f}", scale=100.0)
COST = Metric("cost", r"Cost (\$)", higher_is_better=False, fmt="{:.3f}")
LATENCY = Metric("wall_s", r"Lat.\ (s)", higher_is_better=False, fmt="{:.1f}")


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class Run:
    """A single ``report.csv`` reduced to its per-question mean metrics."""

    benchmark: str
    system: str
    variant: Variant  # row identity (tool set + working-set mode), from config.yaml
    llm: str  # agent model
    judge: str | None  # semantic-filter judge model; None when the run has no sem filter
    top_k: int | None
    run_dir: Path
    n_questions: int
    means: dict[str, float]  # metric key (incl. recall cols) -> per-question mean


@dataclass
class Row:
    """An averaged (system, llm, top_k) cell-row in a benchmark table."""

    variant: Variant
    llm: str
    judge: str | None
    top_k: int | None
    n_runs: int
    n_questions: int
    means: dict[str, float]  # metric key -> mean across runs

    @property
    def system_label(self) -> str:
        if self.variant.system == "rag_llm" and self.top_k is not None:
            return f"{self.variant.label} (k={self.top_k})"
        return self.variant.label

    @property
    def model_label(self) -> str:
        """Agent model, or ``agent / judge`` when the semantic filter used its own model."""
        return model_label(self.llm, self.judge)

    def sort_key(self) -> tuple:
        return (self.variant.order, self.top_k if self.top_k is not None else -1, self.llm, self.judge or "")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _recall_keys_from_header(header: list[str]) -> list[str]:
    """Per-benchmark recall columns are exactly those between ``scorer`` and
    ``retrieved_docs`` (see qatfd.types.report_columns)."""
    try:
        i = header.index("scorer")
        j = header.index("retrieved_docs")
    except ValueError:
        return []
    return header[i + 1 : j]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def load_run(run_dir: Path, *, exclude_failed: bool = False) -> Run | None:
    """Read one run dir into a :class:`Run`, or return None if it has no usable
    report.csv / config.yaml."""
    report = run_dir / "report.csv"
    config = run_dir / "config.yaml"
    if not report.exists() or not config.exists():
        return None

    cfg = yaml.safe_load(config.read_text()) or {}
    systems = cfg.get("systems", {}) or {}
    benchmarks = cfg.get("benchmarks", {}) or {}
    benchmark = benchmarks.get("name", run_dir.parent.parent.name)
    # The run dir is named after the system, so it is the fallback when the snapshot predates
    # `systems.name`. Everything else that identifies the row comes from the config itself.
    variant = variant_of(cfg, fallback_system=run_dir.parent.name)
    llm = llm_of(cfg)
    top_k = systems.get("top_k")

    with report.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)

    recall_keys = _recall_keys_from_header(header)
    if exclude_failed:
        rows = [r for r in rows if str(r.get("failed", "")).strip().lower() != "true"]
    if not rows:
        return None

    metric_keys = ["score", "cost", "wall_s", *recall_keys]
    means: dict[str, float] = {}
    for key in metric_keys:
        vals: list[float] = []
        for r in rows:
            raw = r.get(key, "")
            if raw is None or raw == "":
                continue
            try:
                vals.append(float(raw))
            except ValueError:
                continue
        m = _mean(vals)
        if m is not None:
            means[key] = m

    return Run(
        benchmark=benchmark,
        system=variant.system,
        variant=variant,
        llm=llm,
        judge=judge_of(cfg, llm),
        top_k=int(top_k) if top_k is not None else None,
        run_dir=run_dir,
        n_questions=len(rows),
        means=means,
    )


def discover_runs(results_root: Path, *, exclude_failed: bool = False) -> list[Run]:
    """Find every run dir (one containing report.csv) under results_root."""
    runs: list[Run] = []
    for report in sorted(results_root.rglob("report.csv")):
        run = load_run(report.parent, exclude_failed=exclude_failed)
        if run is not None:
            runs.append(run)
    return runs


# --------------------------------------------------------------------------- #
# Aggregation (average across runs of the same config)
# --------------------------------------------------------------------------- #


def aggregate_rows(runs: list[Run], *, warn: bool = True) -> dict[str, list[Row]]:
    """Group runs by (benchmark, variant, llm, judge, top_k) and average across runs.

    The judge model is part of the key so two semantic-filter runs that differ only in
    `semantic_filter_model` stay separate rows instead of silently averaging together.

    Returns benchmark -> sorted list of Rows.
    """
    groups: dict[tuple, list[Run]] = defaultdict(list)
    for r in runs:
        groups[(r.benchmark, r.variant, r.llm, r.judge, r.top_k)].append(r)

    by_bench: dict[str, list[Row]] = defaultdict(list)
    for (benchmark, variant, llm, judge, top_k), grp in groups.items():
        if warn and len({r.n_questions for r in grp}) > 1:
            counts = ", ".join(f"{r.run_dir.name}={r.n_questions}q" for r in grp)
            print(
                f"[latex_tables] warn: averaging runs with differing question counts "
                f"for {benchmark}/{variant.label}/{model_label(llm, judge)}/k={top_k}: {counts}",
                file=sys.stderr,
            )
        all_keys = sorted({k for r in grp for k in r.means})
        means: dict[str, float] = {}
        for key in all_keys:
            vals = [r.means[key] for r in grp if key in r.means]
            if vals:
                means[key] = sum(vals) / len(vals)
        by_bench[benchmark].append(
            Row(
                variant=variant,
                llm=llm,
                judge=judge,
                top_k=top_k,
                n_runs=len(grp),
                n_questions=max(r.n_questions for r in grp),
                means=means,
            )
        )

    for rows in by_bench.values():
        rows.sort(key=lambda r: r.sort_key())
    return by_bench


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _ordered_recall_keys(rows: list[Row]) -> list[str]:
    """Union of recall keys present across a benchmark's rows, in a stable order
    (known keys first by RECALL_DISPLAY order, then any unknown ones)."""
    present = {k for r in rows for k in r.means if k not in ("score", "cost", "wall_s")}
    known = [k for k in RECALL_DISPLAY if k in present]
    unknown = sorted(present - set(known))
    return known + unknown


def _best_values(rows: list[Row], metrics: list[Metric]) -> dict[str, float | None]:
    """For each metric key, the best (max/min) mean across rows."""
    best: dict[str, float | None] = {}
    for m in metrics:
        vals = [r.means[m.key] for r in rows if m.key in r.means]
        best[m.key] = (max(vals) if m.higher_is_better else min(vals)) if vals else None
    return best


def _fmt_cell(row: Row, m: Metric, best: float | None) -> str:
    if m.key not in row.means:
        return "--"
    val = row.means[m.key]
    text = m.fmt.format(val * m.scale)
    if best is not None and math.isclose(val, best, rel_tol=0, abs_tol=1e-9):
        text = rf"\textbf{{{text}}}"
    return text


def render_benchmark_table(
    bench_key: str,
    rows: list[Row],
    *,
    include_recall: bool = True,
    label_prefix: str = "tab:main",
) -> str:
    """Render one benchmark's booktabs table. Empty ``rows`` -> placeholder."""
    display = BENCHMARKS.get(bench_key, bench_key)
    recall_keys = _ordered_recall_keys(rows) if (include_recall and rows) else []
    recall_metrics = [
        Metric(k, RECALL_DISPLAY.get(k, k.replace("_", " ").title()), higher_is_better=True, fmt="{:.1f}", scale=100.0)
        for k in recall_keys
    ]
    # Column order: System | Model | Acc | recall... | Cost | Latency
    metrics = [ACCURACY, *recall_metrics, COST, LATENCY]
    colspec = "ll" + "r" * len(metrics)
    headers = ["System", "Model"] + [m.header for m in metrics]

    out: list[str] = []
    out.append("% --- auto-generated by qatfd/eval/latex_tables.py ---")
    out.append(r"\begin{table}[t]")
    out.append(r"  \centering")
    out.append(rf"  \begin{{tabular}}{{{colspec}}}")
    out.append(r"    \toprule")
    out.append("    " + " & ".join(headers) + r" \\")
    out.append(r"    \midrule")

    if not rows:
        out.append(rf"    \multicolumn{{{len(headers)}}}{{c}}{{\emph{{No results yet}}}} \\")
    else:
        best = _best_values(rows, metrics)
        for row in rows:
            cells = [row.system_label, _tex_escape(row.model_label)]
            cells += [_fmt_cell(row, m, best[m.key]) for m in metrics]
            comment = f"  % n_runs={row.n_runs}, n_q={row.n_questions}"
            out.append("    " + " & ".join(cells) + r" \\" + comment)

    out.append(r"    \bottomrule")
    out.append(r"  \end{tabular}")
    out.append(
        rf"  \caption{{Main results on {display}. Accuracy and recall are \%; cost is "
        r"avg.\ USD per question; latency is avg.\ seconds per question. Best per column "
        r"in \textbf{bold} (highest accuracy/recall, lowest cost/latency). Values are "
        r"averaged across repeated runs of each configuration.}"
    )
    out.append(rf"  \label{{{label_prefix}:{bench_key}}}")
    out.append(r"\end{table}")
    return "\n".join(out)


def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #


def generate_latex_tables(
    results_root: Path | str,
    *,
    benchmarks: list[str] | None = None,
    include_recall: bool = True,
    include_empty: bool = True,
    llm: str | None = None,
    exclude_failed: bool = False,
) -> dict[str, str]:
    """Build the per-benchmark LaTeX tables.

    Args:
        results_root: the ``results/`` directory.
        benchmarks: ordered benchmark keys to emit (default: all in BENCHMARKS).
        include_recall: include the per-benchmark recall column(s).
        include_empty: emit a placeholder table for benchmarks with no runs
            (e.g. FreshStack until it is implemented).
        llm: if set, only include runs whose ``llm_model`` matches.
        exclude_failed: drop rows where ``failed == True`` before averaging.

    Returns:
        Ordered dict benchmark_key -> LaTeX string.
    """
    results_root = Path(results_root)
    runs = discover_runs(results_root, exclude_failed=exclude_failed)
    if llm is not None:
        runs = [r for r in runs if r.llm == llm]
    by_bench = aggregate_rows(runs)

    bench_keys = benchmarks if benchmarks is not None else list(BENCHMARKS)
    tables: dict[str, str] = {}
    for key in bench_keys:
        rows = by_bench.get(key, [])
        if not rows and not include_empty:
            continue
        tables[key] = render_benchmark_table(key, rows, include_recall=include_recall)
    return tables


def _default_results_root() -> Path:
    return Path(__file__).resolve().parent.parent / "results"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=_default_results_root(),
        help="results/ directory (default: ../results relative to this script)",
    )
    parser.add_argument("--out", type=str, default="-", help="output .tex file, or '-' for stdout (default)")
    parser.add_argument(
        "--benchmarks",
        type=str,
        default=None,
        help="comma-separated benchmark keys to emit (default: all)",
    )
    parser.add_argument("--no-recall", action="store_true", help="omit retrieval recall column(s)")
    parser.add_argument(
        "--no-empty",
        action="store_true",
        help="skip benchmarks with no results instead of emitting a placeholder table",
    )
    parser.add_argument("--llm", type=str, default=None, help="only include runs for this llm_model")
    parser.add_argument("--exclude-failed", action="store_true", help="drop failed questions before averaging")
    args = parser.parse_args(argv)

    if not args.results_root.exists():
        print(f"[latex_tables] results root not found: {args.results_root}", file=sys.stderr)
        return 1

    bench_list = [b.strip() for b in args.benchmarks.split(",")] if args.benchmarks else None
    tables = generate_latex_tables(
        args.results_root,
        benchmarks=bench_list,
        include_recall=not args.no_recall,
        include_empty=not args.no_empty,
        llm=args.llm,
        exclude_failed=args.exclude_failed,
    )

    text = "\n\n".join(tables[k] for k in tables)
    if args.out == "-":
        print(text)
    else:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"[latex_tables] wrote {len(tables)} table(s) to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
