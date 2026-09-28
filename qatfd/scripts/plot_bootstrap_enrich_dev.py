"""Plots for the dev-split bootstrap / enrich sweep (qatfd.k8s sweep `bootstrap_enrich_dev`: the stock runner, one
pass over the split, one run per (cell, seed) under results/<benchmark>/search_agent/<prefix>_<cell>_s<seed>_<ts>/).

Emits two figures, quality vs cost per question and quality vs wall-clock per question, one point per cell at
the mean over seeds with +-1 sample-std bars on both axes. One panel per `--prefixes` entry (e.g. `dev` and
`devq` for the two model configurations side by side), sharing the y axis.

Cost / latency scope (`--cost-scope`): `retrieval` (default) is the search agent alone (`retrieve_cost` /
`retrieve_wall_s`); `retrieval+agents` adds the Bootstrap / Enrich agents' spend amortized over the questions
of the run (it lands on the row of the question they ran before / after); `total` is the report's `cost` /
`wall_s`, which also folds in the answer agent. Failed questions (score 0, e.g. the search agent returned no
well-formed doc ids) stay in the means, as they do in the report's summary.

The latest complete run (rows == --num-questions) per (prefix, cell, seed) wins.

Usage (from qatfd/):
    python3 scripts/plot_bootstrap_enrich_dev.py [--prefixes dev devq] [--cost-scope retrieval|retrieval+agents|total]
                                                 [--quality score|doc_recall|page_recall] [--cells ...]
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

sys.path.insert(0, str(Path(__file__).parent))
from plot_bootstrap_enrich_ub import (  # noqa: E402
    CELL_STYLE, DEFAULT_CELLS, QUALITY_LABEL, SCOPES, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, _legend_outside, _mean,
    _save, _style_axis,
)

RUN_RE = re.compile(r"^(?P<prefix>[A-Za-z0-9]+)_(?P<cell>exp\d+_[a-z_]+?)_s(?P<seed>\d+)_(?P<stamp>\d{8}_\d{6})$")


def load_runs(results_dir: Path, prefixes: list[str], quality: str, scope: str, n_questions: int) -> dict:
    """(prefix, cell, seed) -> per-question means of the latest complete run."""
    runs: dict[tuple[str, str, int], tuple[str, dict]] = {}
    for report in sorted(results_dir.glob("*/report.csv")):
        m = RUN_RE.match(report.parent.name)
        if not m or m["prefix"] not in prefixes:
            continue
        rows = list(csv.DictReader(report.open()))
        if len(rows) != n_questions or sum(float(r["cost"] or 0) for r in rows) == 0:
            continue  # incomplete (still running / pulled mid-run) or crashed at startup
        metrics = {
            "quality": _mean(rows, quality),
            "cost": sum(_mean(rows, c) for c in SCOPES[scope]["cost"]),
            "latency": sum(_mean(rows, c) for c in SCOPES[scope]["latency"]),
            "failed": sum(r["failed"] == "True" for r in rows),
        }
        key = (m["prefix"], m["cell"], int(m["seed"]))
        if key not in runs or m["stamp"] > runs[key][0]:
            runs[key] = (m["stamp"], metrics)
    return {k: v[1] for k, v in runs.items()}


def aggregate(runs: dict, prefixes: list[str], cells: list[str]) -> dict:
    """(prefix, cell) -> {metric: (mean, std), 'seeds': n, 'failed': total failed rows}."""
    out: dict = {}
    for prefix in prefixes:
        for cell in cells:
            pts = [m for (p, c, _), m in runs.items() if p == prefix and c == cell]
            if not pts:
                continue
            agg: dict = {"seeds": len(pts), "failed": sum(p["failed"] for p in pts)}
            for k in ("quality", "cost", "latency"):
                vals = [p[k] for p in pts]
                agg[k] = (statistics.fmean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0)
            out[(prefix, cell)] = agg
    return out


def figure(agg: dict, prefixes: list[str], titles: dict[str, str], cells: list[str], metric: str, xlabel: str,
           quality_label: str, suptitle: str, out: Path) -> None:
    fig, axes = plt.subplots(1, len(prefixes), figsize=(5.2 * len(prefixes), 4.2), facecolor=SURFACE, sharey=True,
                             squeeze=False)
    top = max((a["quality"][0] + a["quality"][1] for a in agg.values()), default=1.0)
    for ax, prefix in zip(axes[0], prefixes, strict=True):
        for cell in cells:
            a = agg.get((prefix, cell))
            if a is None:
                continue
            label, color, marker = CELL_STYLE[cell]
            (mx, sx), (my, sy) = a[metric], a["quality"]
            ax.errorbar(mx, my, xerr=sx or None, yerr=sy or None, fmt="none", ecolor=color, elinewidth=1.0,
                        capsize=3, capthick=1.0, alpha=0.55, zorder=2)
            ax.scatter(mx, my, s=80, color=color, marker=marker, label=label, edgecolors=SURFACE, linewidths=1.2, zorder=3)
            if a["seeds"] < 3:  # partial: say so next to the point rather than silently showing a thin mean
                ax.annotate(f"{a['seeds']} seed{'s' if a['seeds'] > 1 else ''}", (mx, my), xytext=(6, -12),
                            textcoords="offset points", fontsize=7, color=TEXT_SECONDARY)
        ax.set_xscale("log")
        lo, hi = ax.get_xlim()
        subs = (1.0, 2.0, 5.0) if np.log10(hi / lo) > 1.4 else (1.0, 1.5, 2.0, 3.0, 5.0, 7.0)
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=subs, numticks=12))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        _style_axis(ax, f"{xlabel}, log", quality_label, title=titles.get(prefix, prefix))
        ax.set_ylim(-0.03, min(1.05, max(0.3, top * 1.15)))
    _legend_outside(fig, axes[0][0], "cell")
    fig.suptitle(suptitle, fontsize=12, color=TEXT_PRIMARY, y=0.99)
    fig.text(0.0, 0.005, "point = mean over seeds, bars = +-1 sample std on both axes; failed questions count as score 0",
             ha="left", fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    _save(fig, out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results-dir", type=Path, default=Path("results/officeqa/search_agent"))
    p.add_argument("--out-dir", type=Path, default=None, help="default: <results-dir>/plots")
    p.add_argument("--prefixes", nargs="+", default=["dev"], help="run-name prefixes, one panel each (e.g. dev devq)")
    p.add_argument("--panel-titles", nargs="*", default=None, help="panel titles, one per prefix")
    p.add_argument("--quality", default="score", choices=sorted(QUALITY_LABEL))
    p.add_argument("--cost-scope", default="retrieval", choices=sorted(SCOPES))
    p.add_argument("--cells", nargs="+", default=DEFAULT_CELLS, choices=sorted(CELL_STYLE), metavar="CELL")
    p.add_argument("--num-questions", type=int, default=33, help="rows a complete run has")
    p.add_argument("--benchmark-title", default="OfficeQA dev (33 q)")
    args = p.parse_args()

    runs = load_runs(args.results_dir, args.prefixes, args.quality, args.cost_scope, args.num_questions)
    if not runs:
        sys.exit(f"no complete runs with prefix {args.prefixes} under {args.results_dir}")
    agg = aggregate(runs, args.prefixes, args.cells)
    for (prefix, cell), a in sorted(agg.items()):
        print(f"  {prefix:5s} {cell:14s} seeds={a['seeds']} failed_q={a['failed']:2d} {args.quality}={a['quality'][0]:.3f}+-{a['quality'][1]:.3f} "
              f"cost=${a['cost'][0]:.3f}+-{a['cost'][1]:.3f} latency={a['latency'][0]:.0f}+-{a['latency'][1]:.0f}s")

    titles = dict(zip(args.prefixes, args.panel_titles or args.prefixes, strict=False))
    out_dir = args.out_dir or args.results_dir / "plots"
    scope = SCOPES[args.cost_scope]
    tag = "_".join(args.prefixes) + ("" if args.cost_scope == "total" else f"_{scope['tag']}")
    scope_txt = f"cost & latency = {scope['label']}"
    qlabel = QUALITY_LABEL[args.quality]
    prefix_txt = "" if args.cost_scope == "total" else f"{args.cost_scope} "
    figure(agg, args.prefixes, titles, args.cells, "cost", f"{prefix_txt}cost per question ($)", qlabel,
           f"{args.benchmark_title}: {qlabel} vs cost ({scope_txt})", out_dir / f"dev_{args.quality}_vs_cost_{tag}.pdf")
    figure(agg, args.prefixes, titles, args.cells, "latency", f"{prefix_txt}wall-clock per question (s)", qlabel,
           f"{args.benchmark_title}: {qlabel} vs latency ({scope_txt})", out_dir / f"dev_{args.quality}_vs_latency_{tag}.pdf")


if __name__ == "__main__":
    main()
