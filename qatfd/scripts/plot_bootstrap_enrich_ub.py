"""Plots for the bootstrap / enrich upper-bound sweep (scripts/bootstrap_enrich_upper_bound.py, driven on the
cluster by `python -m qatfd.k8s submit ... --sweep bootstrap_enrich_ub`).

One run = one (X, cell, seed): X dev questions answered `passes` times in one long-lived system. This script
aggregates the seeds of each (X, cell) into mean +- std and emits three figures:

  1. ub_score_vs_cost_<pass>.pdf     1x3 panels (X = 1, 5, 10): quality vs cost per question, one point per cell
  2. ub_score_vs_latency_<pass>.pdf  same, quality vs wall-clock per question
  3. ub_trends_vs_x_<pass>.pdf       1x3 panels (quality, cost, latency) vs X, one line per cell

Metrics come from each run's report.csv (one row per (pass, question)); `--pass` selects which pass's rows
are averaged (default 2: the warm pass that sees what pass 1 left on the server; `all` pools both). Per-row
`cost` / `wall_s` already include the Bootstrap / Enrich agents' spend (precompute_* / enrich_*) on the row
they landed on, so per-question means amortize them over the X questions of that pass. Latency caveat: the
bootstrap can add thousands of seconds to pass 1 (it lands on the first question), so pass-1 latency is
dominated by it at small X.

Runs are keyed by their dir name `<prefix>_x<X>_<cell>_s<seed>_<stamp>`; the latest complete run (rows ==
X * passes) per (X, cell, seed) wins, so old local runs are superseded by the cluster's. Std is the sample
std (ddof=1) across seeds; with fewer than two seeds no bar is drawn.

Usage (from qatfd/, any interpreter with matplotlib + numpy):
    python3 scripts/plot_bootstrap_enrich_ub.py [--pass 1|2|all] [--quality score|doc_recall|page_recall]
                                                [--cost-scope total|retrieval+agents|retrieval]
                                                [--cells exp1_baseline ...] [--results-dir ...] [--out-dir ...]
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

RUN_RE = re.compile(r"^(?P<prefix>[A-Za-z0-9]+)_x(?P<x>\d+)_(?P<cell>exp\d+_[a-z_]+?)_s(?P<seed>\d+)_(?P<stamp>\d{8}_\d{6})$")

# Fixed categorical assignment (validated all-pairs on the light surface, CVD in the warn band so marker
# shape doubles as the identity encoding). Order = the sweep's DEFAULT_CELLS.
CELL_STYLE: dict[str, tuple[str, str, str]] = {
    # cell -> (legend label, color, marker)
    "exp1_baseline": ("baseline", "#2a78d6", "o"),
    "exp2_ws": ("trajectory working sets", "#eda100", "^"),
    "exp3_bs": ("bootstrap", "#1baf7a", "s"),
    "exp5_en": ("enrich", "#4a3aa7", "D"),
    "exp7_bs_en": ("bootstrap + enrich", "#e34948", "v"),
    "exp4_bs_ws": ("bootstrap + ws", "#eb6834", "P"),
    "exp6_en_ws": ("enrich + ws", "#e87ba4", "X"),
    "exp8_bs_en_ws": ("bootstrap + enrich + ws", "#008300", "*"),
}
DEFAULT_CELLS = ["exp1_baseline", "exp2_ws", "exp3_bs", "exp5_en", "exp7_bs_en"]
QUALITY_LABEL = {"score": "mean score", "doc_recall": "mean doc recall", "page_recall": "mean page recall"}

TEXT_PRIMARY, TEXT_SECONDARY, SURFACE, GRID, SPINE = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df", "#d4d3cf"


def _mean(rows: list[dict], col: str) -> float:
    vals = [float(r[col]) for r in rows if r.get(col) not in (None, "", "None")]
    return sum(vals) / len(vals) if vals else 0.0


# Which report.csv columns make up the plotted cost / latency. Per the harness schema, `cost` (`wall_s`) is the
# sum of the four phase columns; the collection agents' spend lands on the row of the question it ran before
# (precompute: first question of pass 1) or after (enrich), so per-question means amortize it over the pass.
SCOPES: dict[str, dict] = {
    "total": {"cost": ["cost"], "latency": ["wall_s"], "label": "total (retrieval + answer + bootstrap + enrich)",
              "tag": "total"},
    "retrieval+agents": {"cost": ["retrieve_cost", "precompute_cost", "enrich_cost"],
                         "latency": ["retrieve_wall_s", "precompute_wall_s", "enrich_wall_s"],
                         "label": "retrieval + collection agents (no answer agent)", "tag": "retragents"},
    "retrieval": {"cost": ["retrieve_cost"], "latency": ["retrieve_wall_s"],
                  "label": "search agent only", "tag": "retrieval"},
}


def load_runs(results_dir: Path, which_pass: str, quality: str, scope: str = "total") -> dict[tuple[int, str, int], dict]:
    """(X, cell, seed) -> per-question means of the latest complete run."""
    runs: dict[tuple[int, str, int], tuple[str, dict]] = {}
    for report in sorted(results_dir.glob("*/report.csv")):
        m = RUN_RE.match(report.parent.name)
        if not m:
            continue
        x, cell, seed, stamp = int(m["x"]), m["cell"], int(m["seed"]), m["stamp"]
        rows = list(csv.DictReader(report.open()))
        passes = sorted({int(r["pass"]) for r in rows}) if rows else []
        if not rows or len(rows) != x * len(passes) or sum(float(r["cost"] or 0) for r in rows) == 0:
            continue  # incomplete or crashed-at-startup
        sel = rows if which_pass == "all" else [r for r in rows if int(r["pass"]) == int(which_pass)]
        if not sel:
            continue
        metrics = {
            "quality": _mean(sel, quality),
            "cost": sum(_mean(sel, c) for c in SCOPES[scope]["cost"]),
            "latency": sum(_mean(sel, c) for c in SCOPES[scope]["latency"]),
            "n": len(sel),
        }
        key = (x, cell, seed)
        if key not in runs or stamp > runs[key][0]:
            runs[key] = (stamp, metrics)
    return {k: v[1] for k, v in runs.items()}


def aggregate(runs: dict[tuple[int, str, int], dict], xs: list[int], cells: list[str]) -> dict[tuple[int, str], dict]:
    """(X, cell) -> {metric: (mean, std), 'seeds': n} across seeds."""
    out: dict[tuple[int, str], dict] = {}
    for x in xs:
        for cell in cells:
            pts = [m for (rx, rc, _), m in runs.items() if rx == x and rc == cell]
            if not pts:
                continue
            agg: dict = {"seeds": len(pts)}
            for k in ("quality", "cost", "latency"):
                vals = [p[k] for p in pts]
                agg[k] = (statistics.fmean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0)
            out[(x, cell)] = agg
    return out


def _style_axis(ax: plt.Axes, xlabel: str, ylabel: str, title: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    if title:
        ax.set_title(title, fontsize=10.5, color=TEXT_PRIMARY, pad=8)
    ax.set_xlabel(xlabel, fontsize=8.5, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=8.5, color=TEXT_SECONDARY)
    ax.grid(True, which="major", linewidth=0.4, color=GRID, zorder=0)
    ax.tick_params(labelsize=8, colors=TEXT_SECONDARY)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.6)


def _legend_outside(fig: plt.Figure, ax: plt.Axes, title: str) -> None:
    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, object] = {}
    for h, lab in zip(handles, labels, strict=False):
        seen.setdefault(lab, h)
    fig.legend(list(seen.values()), list(seen), loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False,
               fontsize=9, labelcolor=TEXT_PRIMARY, title=title, title_fontsize=9.5, borderaxespad=0.5)


def _quality_ylim(agg: dict) -> tuple[float, float]:
    """0 to just above the highest mean + std, so the points are not squashed into the bottom of a 0-1 axis."""
    top = max((a["quality"][0] + a["quality"][1] for a in agg.values()), default=1.0)
    return -0.03, min(1.05, max(0.3, top * 1.15))


def scatter_figure(agg: dict, xs: list[int], cells: list[str], metric: str, xlabel: str, quality_label: str,
                   suptitle: str, out: Path, xscale: str = "log") -> None:
    fig, axes = plt.subplots(1, len(xs), figsize=(4.2 * len(xs), 4.0), facecolor=SURFACE, sharey=True)
    axes = np.atleast_1d(axes)
    ylim = _quality_ylim(agg)
    for ax, x in zip(axes, xs, strict=True):
        for cell in cells:
            a = agg.get((x, cell))
            if a is None:
                continue
            label, color, marker = CELL_STYLE[cell]
            (mx, sx), (my, sy) = a[metric], a["quality"]
            ax.errorbar(mx, my, xerr=sx or None, yerr=sy or None, fmt="none", ecolor=color, elinewidth=1.0,
                        capsize=3, capthick=1.0, alpha=0.55, zorder=2)
            ax.scatter(mx, my, s=80, color=color, marker=marker, label=label, edgecolors=SURFACE,
                       linewidths=1.2, zorder=3)
        ax.set_xscale(xscale)
        if xscale == "log":
            # cells differ by 10x+ in cost/latency; a log axis keeps the cheap cluster legible. Plain
            # number ticks at 1-2-3-5 steps rather than 10^k notation.
            lo, hi = ax.get_xlim()
            subs = (1.0, 2.0, 5.0) if np.log10(hi / lo) > 1.4 else (1.0, 2.0, 3.0, 5.0)  # fewer ticks on wide axes
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=subs, numticks=12))
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
        _style_axis(ax, f"{xlabel}, log", quality_label, title=f"X = {x}")
        ax.set_ylim(*ylim)
    _legend_outside(fig, axes[0], "cell")
    fig.suptitle(suptitle, fontsize=12, color=TEXT_PRIMARY, y=0.99)
    fig.text(0.0, 0.005, "point = mean over seeds, bars = +-1 sample std (seeds with a single run have no bar)",
             ha="left", fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    _save(fig, out)


def trend_figure(agg: dict, xs: list[int], cells: list[str], quality_label: str, suptitle: str, out: Path,
                 cost_label: str = "cost per question ($)", latency_label: str = "wall-clock per question (s)") -> None:
    panels = [("quality", quality_label), ("cost", cost_label), ("latency", latency_label)]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0), facecolor=SURFACE)
    # dodge the cells a little around each X: means often coincide exactly (e.g. every cell at 0.2), and
    # without an offset the last-drawn marker hides the others
    dodge = dict(zip(cells, np.linspace(-0.3, 0.3, len(cells)) if len(cells) > 1 else [0.0], strict=True))
    for ax, (metric, ylabel) in zip(axes, panels, strict=True):
        for cell in cells:
            pts = [(x, agg[(x, cell)][metric]) for x in xs if (x, cell) in agg]
            if not pts:
                continue
            label, color, marker = CELL_STYLE[cell]
            px = [p[0] + dodge[cell] for p in pts]
            py = [p[1][0] for p in pts]
            pe = [p[1][1] for p in pts]
            ax.errorbar(px, py, yerr=pe, color=color, ecolor=color, elinewidth=1.0, capsize=3, capthick=1.0,
                        linewidth=1.8, marker=marker, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.0,
                        label=label, alpha=0.95, zorder=3)
        ax.set_xticks(xs)
        ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
        _style_axis(ax, "X (questions per pass)", ylabel)
        if metric == "quality":
            ax.set_ylim(*_quality_ylim(agg))
        else:
            ax.set_ylim(bottom=0)
    _legend_outside(fig, axes[0], "cell")
    fig.suptitle(suptitle, fontsize=12, color=TEXT_PRIMARY, y=0.99)
    fig.text(0.0, 0.005, "point = mean over seeds, bars = +-1 sample std; cells are offset slightly around each X so "
             "coincident means stay visible", ha="left", fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    _save(fig, out)


def _save(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf", facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), format="png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} (+ .png)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results-dir", type=Path, default=Path("results/officeqa/search_agent_ub3"))
    p.add_argument("--out-dir", type=Path, default=None, help="default: <results-dir>/plots")
    p.add_argument("--pass", dest="which_pass", default="2", choices=["1", "2", "all"])
    p.add_argument("--quality", default="score", choices=sorted(QUALITY_LABEL))
    p.add_argument("--xs", type=int, nargs="+", default=[1, 5, 10])
    p.add_argument("--cells", nargs="+", default=DEFAULT_CELLS, choices=sorted(CELL_STYLE), metavar="CELL")
    p.add_argument("--cost-scope", default="total", choices=sorted(SCOPES),
                   help="which phases the cost/latency axes include (default: total, i.e. the report's cost / wall_s)")
    p.add_argument("--benchmark-title", default="OfficeQA dev")
    args = p.parse_args()

    runs = load_runs(args.results_dir, args.which_pass, args.quality, args.cost_scope)
    if not runs:
        sys.exit(f"no complete runs under {args.results_dir}")
    agg = aggregate(runs, args.xs, args.cells)
    for (x, cell), a in sorted(agg.items()):
        print(f"  X={x:<3} {cell:14s} seeds={a['seeds']} {args.quality}={a['quality'][0]:.3f}+-{a['quality'][1]:.3f} "
              f"cost=${a['cost'][0]:.3f}+-{a['cost'][1]:.3f} latency={a['latency'][0]:.0f}+-{a['latency'][1]:.0f}s")

    out_dir = args.out_dir or args.results_dir / "plots"
    scope = SCOPES[args.cost_scope]
    tag = f"pass{args.which_pass}" if args.which_pass != "all" else "allpasses"
    if args.cost_scope != "total":
        tag = f"{scope['tag']}_{tag}"   # the default (total) keeps the original file names
    pass_txt = f"pass {args.which_pass}" if args.which_pass != "all" else "all passes"
    scope_txt = f"{pass_txt}; cost & latency = {scope["label"]}"
    qlabel = QUALITY_LABEL[args.quality]
    cost_axis = "cost per question ($)" if args.cost_scope == "total" else f"{args.cost_scope} cost per question ($)"
    lat_axis = "wall-clock per question (s)" if args.cost_scope == "total" else f"{args.cost_scope} wall-clock per question (s)"
    scatter_figure(agg, args.xs, args.cells, "cost", cost_axis, qlabel,
                   f"{args.benchmark_title}: {qlabel} vs cost by X ({scope_txt})", out_dir / f"ub_{args.quality}_vs_cost_{tag}.pdf")
    scatter_figure(agg, args.xs, args.cells, "latency", lat_axis, qlabel,
                   f"{args.benchmark_title}: {qlabel} vs latency by X ({scope_txt})",
                   out_dir / f"ub_{args.quality}_vs_latency_{tag}.pdf")
    trend_figure(agg, args.xs, args.cells, qlabel, f"{args.benchmark_title}: {qlabel}, cost and latency vs X ({scope_txt})",
                 out_dir / f"ub_{args.quality}_trends_vs_x_{tag}.pdf", cost_axis, lat_axis)


if __name__ == "__main__":
    main()
