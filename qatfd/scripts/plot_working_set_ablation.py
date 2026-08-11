"""Scatterplots for the OfficeQA working-set ablation: one panel per retrieval tool set,
one point per (agent model, working set on/off) run, plotting answer quality against cost.

Each run dir under results/officeqa/ablation_search_agent/ is one point: the agent model is
read from the run's config.yaml snapshot, working-set state from the `_ws_off` run-name marker.
If a (tool set, model, ws) combo has several completed runs, the latest one wins; runs with
zero cost (crashed-at-startup sweeps) or missing report.csv are skipped.

Usage (from qatfd/, any interpreter with matplotlib + pyyaml):
    python3 scripts/plot_working_set_ablation.py [results_dir] [out_pdf]
Defaults: results/officeqa/ablation_search_agent -> working_set_ablation_scatter.pdf therein.
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


def load_runs(results_dir: Path) -> dict[tuple[str, str, str], dict]:
    """(arm, model, ws) -> metrics of the latest completed run of that combo."""
    runs: dict[tuple[str, str, str], tuple[str, dict]] = {}
    for report in sorted(results_dir.glob("*/report.csv")):
        run = report.parent.name
        m = re.match(r"(.+?)(_ws_off)?_(\d{8}_\d{6})$", run)
        if not m or m.group(1) not in ARMS:
            continue
        arm, ws, stamp = m.group(1), ("off" if m.group(2) else "on"), m.group(3)
        cfg = yaml.safe_load((report.parent / "config.yaml").open())
        model = cfg["inference"]["llm_model"].split("/")[-1]
        rows = list(csv.DictReader(report.open()))
        cost = sum(float(r["cost"]) for r in rows if r["cost"])
        if not rows or cost == 0:  # crashed-at-startup sweep; not a real datapoint
            continue
        scores = [float(r["score"]) for r in rows if r["score"] not in ("", "None")]
        metrics = {
            "n": len(rows),
            "score": sum(scores) / len(scores),
            "cost_per_q": cost / len(rows),
        }
        key = (arm, model, ws)
        if key not in runs or stamp > runs[key][0]:
            runs[key] = (stamp, metrics)
    return {k: v[1] for k, v in runs.items()}


def main() -> None:
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/officeqa/ablation_search_agent")
    out_pdf = Path(sys.argv[2]) if len(sys.argv) > 2 else results_dir / "working_set_ablation_scatter.pdf"
    runs = load_runs(results_dir)
    if not runs:
        sys.exit(f"no completed runs found under {results_dir}")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), facecolor=SURFACE)
    for ax, arm in zip(axes.flat, ARMS, strict=False):
        ax.set_facecolor(SURFACE)
        for model, ws, label, color, marker in CONFIGS:
            pt = runs.get((arm, model, ws))
            if pt is None:
                continue
            ax.scatter(pt["cost_per_q"], pt["score"], s=90, color=color, marker=marker,
                       label=label, edgecolors=SURFACE, linewidths=1.5, zorder=3)
            # ws-on labels sit above their point, ws-off below — paired runs often nearly
            # coincide (same tool set + model), so a fixed single offset collides.
            dy = 10 if ws == "on" else -15
            ax.annotate(f"{pt['score']:.2f}", (pt["cost_per_q"], pt["score"]),
                        xytext=(0, dy), textcoords="offset points",
                        ha="center", fontsize=7.5, color=TEXT_SECONDARY, zorder=4)
        ax.set_xscale("log")
        ax.set_xlim(0.03, 1.6)
        ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1.0])
        ax.set_xticklabels(["$0.05", "$0.10", "$0.20", "$0.50", "$1"])
        ax.minorticks_off()
        ax.set_title(ARM_TITLES[arm], fontsize=10.5, color=TEXT_PRIMARY, pad=8)
        ax.set_xlabel("cost per question ($, log)", fontsize=8.5, color=TEXT_SECONDARY)
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

    fig.suptitle("OfficeQA dev (33 q): score vs cost by retrieval tool set",
                 fontsize=13, color=TEXT_PRIMARY, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", facecolor=SURFACE)
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
