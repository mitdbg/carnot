"""Scatterplots for the OfficeQA working-set ablation: one panel per retrieval tool set,
one point per (agent model, working set on/off) run, plotting answer quality against the
SEARCH AGENT's cost and latency. Emits two PDFs — one for cost, one for latency.

Both axes isolate the search agent (the retrieval phase the working set actually affects):
`retrieve_cost` and `retrieve_wall_s`, not the whole-pipeline `cost` / `wall_s` (which also
fold in the answer/compute agent). Per the harness schema, cost = retrieve_cost + compute_cost
and wall_s = retrieve_wall_s + compute_wall_s. Latency caveat: `retrieve_wall_s` is measured
under each run's configured worker concurrency, so — unlike cost — it is not concurrency-invariant.

Each run dir under results/officeqa/search_agent/ is one point: the agent model and the
working-set state are read from the run's config.yaml snapshot (`systems.working_set_off`),
falling back to the `_ws_off` run-name marker for runs written before that flag existed.
If a (tool set, model, ws) combo has several completed runs, the latest one wins; runs with
zero cost (crashed-at-startup sweeps) or missing report.csv are skipped. All fields come from
each run's report.csv (the harness's canonical per-question output).

Usage (from qatfd/, any interpreter with matplotlib + pyyaml):
    python3 scripts/plot_working_set_ablation.py [results_dir] [out_dir]
Defaults: results/officeqa/search_agent, writing both PDFs into that dir:
    working_set_ablation_scatter_retrieve_cost.pdf
    working_set_ablation_scatter_retrieve_latency.pdf
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

ARMS = ["grep_read", "vector_read", "grep_vector_read", "sem_read", "all_tools"]
ARM_TITLES = {
    "grep_read": "grep + read",
    "vector_read": "vector + read",
    "grep_vector_read": "grep + vector + read",
    "sem_read": "sem filter + read",
    "all_tools": "grep + vector + sem + read",
}

# Fixed categorical assignment (validated 4-slot palette); marker shape doubles the
# working-set encoding so identity never rides on color alone.
CONFIGS = [
    # (model substring, ws, label, color, marker)
    ("qwen3.6-35b-a3b", "on", "qwen3.6-35b · working set ON", "#2a78d6", "o"),
    ("qwen3.6-35b-a3b", "off", "qwen3.6-35b · working set OFF", "#eb6834", "^"),
    ("gemini-3.5-flash", "on", "gemini-3.5-flash · working set ON", "#1baf7a", "o"),
    ("gemini-3.5-flash", "off", "gemini-3.5-flash · working set OFF", "#eda100", "^"),
]

TEXT_PRIMARY, TEXT_SECONDARY, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

# One figure per metric. Each isolates the search agent (retrieval phase). `key` indexes the
# per-run metrics dict built in load_runs(); the rest is pure axis/label/output styling.
METRICS = [
    {
        "key": "retrieve_cost_per_q",
        "out": "working_set_ablation_scatter_retrieve_cost.pdf",
        "suptitle": "OfficeQA dev (33 q): score vs search-agent cost by retrieval tool set",
        "xlabel": "search-agent cost per question ($, log)",
        "xscale": "log",
        "xlim": (0.03, 1.6),
        "xticks": [0.05, 0.1, 0.2, 0.5, 1.0],
        "xticklabels": ["$0.05", "$0.10", "$0.20", "$0.50", "$1"],
        "caption": None,
    },
    {
        "key": "retrieve_wall_s",
        "out": "working_set_ablation_scatter_retrieve_latency.pdf",
        "suptitle": "OfficeQA dev (33 q): score vs search-agent wall-clock by retrieval tool set",
        "xlabel": "search-agent wall-clock per question (s)",
        "xscale": "linear",
        "xlim": (80, 820),
        "xticks": [200, 400, 600, 800],
        "xticklabels": ["200 s", "400 s", "600 s", "800 s"],
        "caption": ("Wall-clock measured under each run's configured worker concurrency; "
                    "unlike cost, latency is not concurrency-invariant."),
    },
]


def load_runs(results_dir: Path) -> dict[tuple[str, str, str], dict]:
    """(arm, model, ws) -> metrics of the latest completed run of that combo."""
    runs: dict[tuple[str, str, str], tuple[str, dict]] = {}
    for report in sorted(results_dir.glob("*/report.csv")):
        run = report.parent.name
        m = re.match(r"(.+?)(_ws_off)?_(\d{8}_\d{6})$", run)
        if not m or m.group(1) not in ARMS:
            continue
        arm, stamp = m.group(1), m.group(3)
        cfg = yaml.safe_load((report.parent / "config.yaml").open())
        model = cfg["inference"]["llm_model"].split("/")[-1]
        # Every tool set now writes under the one `search_agent` dir, so take the working-set
        # state from the config rather than trusting the run-name marker to have been set.
        ws_off = (cfg.get("systems") or {}).get("working_set_off")
        ws = ("off" if ws_off else "on") if ws_off is not None else ("off" if m.group(2) else "on")
        rows = list(csv.DictReader(report.open()))
        cost = sum(float(r["cost"]) for r in rows if r["cost"])
        if not rows or cost == 0:  # crashed-at-startup sweep; not a real datapoint
            continue
        scores = [float(r["score"]) for r in rows if r["score"] not in ("", "None")]

        def _mean(col: str) -> float:
            vals = [float(r[col]) for r in rows if r[col] not in ("", "None")]
            return sum(vals) / len(vals) if vals else 0.0

        metrics = {
            "n": len(rows),
            "score": sum(scores) / len(scores),
            # isolate the search agent: retrieval-phase cost/latency, not whole-pipeline
            "retrieve_cost_per_q": _mean("retrieve_cost"),
            "retrieve_wall_s": _mean("retrieve_wall_s"),
        }
        key = (arm, model, ws)
        if key not in runs or stamp > runs[key][0]:
            runs[key] = (stamp, metrics)
    return {k: v[1] for k, v in runs.items()}


def make_figure(runs: dict[tuple[str, str, str], dict], spec: dict, out_pdf: Path) -> None:
    """Render one score-vs-`spec['key']` scatter (6 panels + legend) to `out_pdf`."""
    key = spec["key"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), facecolor=SURFACE)
    for ax, arm in zip(axes.flat, ARMS, strict=False):
        ax.set_facecolor(SURFACE)
        for model, ws, label, color, marker in CONFIGS:
            pt = runs.get((arm, model, ws))
            if pt is None:
                continue
            ax.scatter(pt[key], pt["score"], s=90, color=color, marker=marker,
                       label=label, edgecolors=SURFACE, linewidths=1.5, zorder=3)
            # ws-on labels sit above their point, ws-off below — paired runs often nearly
            # coincide (same tool set + model), so a fixed single offset collides.
            dy = 10 if ws == "on" else -15
            ax.annotate(f"{pt['score']:.2f}", (pt[key], pt["score"]),
                        xytext=(0, dy), textcoords="offset points",
                        ha="center", fontsize=7.5, color=TEXT_SECONDARY, zorder=4)
        ax.set_xscale(spec["xscale"])
        ax.set_xlim(*spec["xlim"])
        ax.set_xticks(spec["xticks"])
        ax.set_xticklabels(spec["xticklabels"])
        ax.minorticks_off()
        ax.set_title(ARM_TITLES[arm], fontsize=10.5, color=TEXT_PRIMARY, pad=8)
        ax.set_xlabel(spec["xlabel"], fontsize=8.5, color=TEXT_SECONDARY)
        ax.set_ylabel("mean score", fontsize=8.5, color=TEXT_SECONDARY)
        ax.set_ylim(0, 0.75)
        ax.grid(True, which="major", linewidth=0.4, color="#e4e3df", zorder=0)
        ax.tick_params(labelsize=8, colors=TEXT_SECONDARY)
        for spine in ax.spines.values():
            spine.set_color("#d4d3cf")
            spine.set_linewidth(0.6)

    # Last grid cell hosts the shared legend instead of a sixth panel.
    legend_ax = axes.flat[len(ARMS)]
    legend_ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="center", frameon=False, fontsize=9.5,
                     labelcolor=TEXT_PRIMARY, title="agent model · working set",
                     title_fontsize=10)

    fig.suptitle(spec["suptitle"], fontsize=13, color=TEXT_PRIMARY, y=0.98)
    bottom = 0.035 if spec["caption"] else 0.0
    fig.tight_layout(rect=(0, bottom, 1, 0.96))
    if spec["caption"]:
        fig.text(0.5, 0.008, spec["caption"], ha="center", fontsize=7.5, color=TEXT_SECONDARY)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out_pdf}")


def main() -> None:
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/officeqa/search_agent")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else results_dir
    runs = load_runs(results_dir)
    if not runs:
        sys.exit(f"no completed runs found under {results_dir}")
    for spec in METRICS:
        make_figure(runs, spec, out_dir / spec["out"])


if __name__ == "__main__":
    main()
