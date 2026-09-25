"""Plot the Codex session-resume / shell ablation: mean +- std (across shuffle seeds) of run-level
quality vs cost and quality vs latency, one point per scenario (and per tagged variant of a scenario).

Per run: quality = mean score over ALL rows (a failed row scores 0), cost = total $ across rows,
latency = total answer wall time (s) across rows. Each scenario's point is the mean over its seeds
with +-1 sample std error bars on both axes. Labels carry `ok/n` (rows that produced an answer,
averaged over seeds) so a run poisoned by failures is visible on the chart itself.

Run dirs are named `codex_<scenario>[_<variant>]_<split>_s<seed>_<YYYYmmdd>_<HHMMSS>` (see
scripts/run_codex_ablation.sh, VARIANT=...). A variant (e.g. `c40` = auto-compact at 40% of the context
window) is a separate series drawn in the scenario's hue with a different marker / line style, so it
sits next to the untagged baseline instead of replacing it. Newest run dir wins per (scenario, variant, seed).
Runs still in progress are dropped by default (a run counts as complete when it has as many rows as the
largest run found; override with --min-rows), so a partial seed never drags a scenario's mean around.
Plots are per benchmark: the benchmark is read from the results root (`results/<benchmark>/codex`, or
--benchmark) and goes into every title and output file name, so officeqa and officeqa_synth never share
a figure or overwrite each other's PNGs.

Usage (from qatfd/):
    python3 scripts/plot_codex_ablation.py [--results-root results/officeqa/codex] [--out-dir <dir>]
    python3 scripts/plot_codex_ablation.py --results-root results/officeqa_synth/codex
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fixed scenario order + fixed categorical hue per scenario (never cycled)
SCENARIOS = [
    ("par", "isolation (parallel)", "#2a78d6"),
    ("seq_resume", "sequential + resume", "#1baf7a"),
    ("seq_shell", "sequential + shell", "#9b59d0"),
    ("seq_shell_resume", "sequential + shell + resume", "#d63d7c"),
    # SearchAgent baseline (scripts/run_search_agent_baseline.sh): full working set, same model / seeds
    ("sa_par", "search agent (parallel)", "#4a4a4a"),
    ("sa_seq", "search agent (sequential + ws reuse)", "#a0a0a0"),
]
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
# codex_<scenario>... from run_codex_ablation.sh, sa_<par|seq>... from run_search_agent_baseline.sh; the
# search-agent scenario keys carry the `sa_` prefix so both systems share one SCENARIOS table.
RUN_RE = re.compile(
    r"^(?:codex_(?P<scenario>seq_shell_resume|seq_shell|seq_resume|par)"
    r"|(?P<sa_scenario>sa_(?:par|seq)))"
    r"(?:_(?P<variant>[a-z0-9]+))?_(?P<split>[a-z]+)_s(?P<seed>\d+)_\d{8}_\d{6}$"
)
SIBLING_SYSTEM_DIRS = ("search_agent",)  # results/<bench>/<dir> scanned alongside results/<bench>/codex
# baseline (untagged) first, then variants in the order they are first seen
VARIANT_MARKERS = ["o", "s", "D", "^", "v"]
VARIANT_LINESTYLES = ["-", "--", ":", "-."]

# Questions whose wall time is dropped from LATENCY aggregates only (quality and cost still count them).
# Keyed by (run label without the timestamp suffix, qid) -> reason. Use for provider outages, not slow agents.
LATENCY_EXCLUDE: dict[tuple[str, str], str] = {
    ("codex_seq_shell_dev_s0", "UID0053"):
        "6085 s: one 100-min stall waiting for the model response (provider), no agent activity",
    # v1-corpus sweep (2026-09-22): the same ~100-min OpenRouter stall signature (wall 6067-6234 s on
    # questions with an ordinary number of agent events); the three par_s2 questions overlapped in time
    ("codex_par_v1_dev_s2", "UID0055"): "6067 s: provider stall (07:26-09:07 UTC outage window)",
    ("codex_par_v1_dev_s2", "UID0035"): "6067 s: provider stall (same outage window)",
    ("codex_par_v1_dev_s2", "UID0056"): "6102 s: provider stall (same outage window)",
    ("codex_seq_resume_v1c40_dev_s1", "UID0025"): "6198 s: provider stall (15:08-16:51 UTC)",
    ("codex_seq_shell_v1_dev_s2", "UID0003"): "6234 s: provider stall (10:46-12:30 UTC)",
}

# Single questions dropped from EVERY aggregate (quality, cost, latency, by-index) for one run, keyed by
# (run label, qid) -> reason. For non-resume scenarios, where questions are independent, a question whose
# answer came from reading gold / prior results on disk is dropped on its own instead of the whole run.
CONTAMINATED_QUESTIONS: dict[tuple[str, str], str] = {
    ("codex_seq_shell_v1_dev_s0", "UID0030"):
        "contaminated: read prior runs' traces for UID0030 and the answer row in skunk-ui/skunk/officeqa_full.csv",
    ("codex_seq_shell_v1_dev_s2", "UID0030"):
        "contaminated: read prior runs' results.jsonl rows for UID0030 under v1_small_models_results/",
}

# Whole runs dropped from EVERY aggregate (quality, cost, latency, by-index, steps), keyed by run label -> reason.
# Reserved for runs whose answers are not the agent's own work, e.g. a shell-enabled agent that read the answer key.
EXCLUDED_RUNS: dict[str, str] = {
    "codex_seq_shell_resume_dev_s2":
        "contaminated: from question 3 on it grepped benchmarks/officeqa/officeqa_pro.csv (the gold answers) "
        "for each question's text; 31/33 correct",
    "codex_seq_shell_resume_dev_s1":
        "contaminated: grepped earlier runs' report.csv under v1_small_models_results/ for 'UID0010.*answer|FINAL_ANSWER' "
        "(gold + prior predictions)",
    # v1-corpus sweep (2026-09-22); resume shares one thread across questions, so any hit taints the run
    "codex_seq_shell_resume_v1_dev_s0":
        "contaminated: for 28/33 questions grepped each question's text in v1_small_models_results/.../report.csv "
        "(gold answer column) via a relative path -- the resumed session ran in the qatfd checkout, not its workspace",
    "codex_seq_shell_resume_v1_dev_s1":
        "contaminated: 7 questions grepped v1_small_models_results/ report.csv / results.jsonl (gold + prior predictions)",
    "codex_seq_shell_resume_v1_dev_s2":
        "contaminated: ~20 questions read benchmarks/officeqa/officeqa_pro.csv and synth_dev_qa_pairs.json (gold answers)",
}


def run_label(run_dir_name: str) -> str:
    """Run dir name minus its `_YYYYmmdd_HHMMSS` suffix, i.e. the experiments.run_name."""
    return re.sub(r"_\d{8}_\d{6}$", "", run_dir_name)


def latency_excluded(run_dir_name: str, qid: str) -> bool:
    return (run_label(run_dir_name), qid) in LATENCY_EXCLUDE


def contaminated(run_dir_name: str, qid: str) -> bool:
    return (run_label(run_dir_name), qid) in CONTAMINATED_QUESTIONS


def parse_run_name(name: str) -> dict | None:
    m = RUN_RE.match(name)
    if not m:
        return None
    scenario = m["scenario"] or m["sa_scenario"]
    return {"scenario": scenario, "variant": m["variant"] or "", "split": m["split"], "seed": int(m["seed"])}


def run_dirs(results_root: Path):
    """Run dirs under the codex results root plus its sibling system roots (results/<bench>/search_agent)."""
    roots = [results_root] + [results_root.parent / d for d in SIBLING_SYSTEM_DIRS if results_root.name == "codex"]
    for root in roots:
        if root.is_dir():
            yield from sorted(p for p in root.iterdir() if p.is_dir())


def benchmark_label(results_root: Path, override: str | None = None) -> str:
    """The benchmark a results root belongs to: `results/<benchmark>/codex` -> `<benchmark>`. A sweep kept
    under its own root (`results/<tag>/<benchmark>/codex`, e.g. the v3 sample) is labelled `<benchmark>_<tag>`
    so its figures never read as, or overwrite, the default root's."""
    if override:
        return override
    root = results_root.resolve()
    if root.name != "codex":
        return root.name
    bench, tag = root.parent.name, root.parent.parent.name
    return bench if tag in ("results", "") else f"{bench}_{tag}"


def count_rows(results_path: Path) -> int:
    with results_path.open() as f:
        return sum(1 for line in f if line.strip())


def find_runs(results_root: Path, min_rows: int | None = None) -> dict[tuple[str, str], dict[int, Path]]:
    """{(scenario, variant): {seed: run_dir}}, newest COMPLETE run dir per key (dir names sort by timestamp).
    A run is complete when its results.jsonl has >= `min_rows` rows; None => the largest row count seen,
    i.e. only finished runs (an in-progress run has fewer rows than a finished one on the same benchmark)."""
    found = []
    for d in run_dirs(results_root):
        info = parse_run_name(d.name)
        if info and (d / "results.jsonl").exists():
            if run_label(d.name) in EXCLUDED_RUNS:
                print(f"[excluded] {d.name}: {EXCLUDED_RUNS[run_label(d.name)]}")
                continue
            found.append((info, d, count_rows(d / "results.jsonl")))
    if min_rows is None:
        min_rows = max((n for _, _, n in found), default=0)
    runs: dict[tuple[str, str], dict[int, Path]] = defaultdict(dict)
    for info, d, n in found:
        if n < min_rows:
            print(f"[skip] {d.name}: {n}/{min_rows} rows (in progress or incomplete)")
            continue
        runs[(info["scenario"], info["variant"])][info["seed"]] = d
    return runs


def quality_label(results_root: Path) -> str:
    """Y-axis wording from the scorer the runs actually used (nugget recall vs exact-answer score)."""
    scorers: set[str] = set()
    for d in run_dirs(results_root):
        if parse_run_name(d.name) and (d / "results.jsonl").exists():
            with (d / "results.jsonl").open() as f:
                for line in f:
                    if line.strip():
                        scorers.add(json.loads(line).get("scorer", ""))
    if scorers and all(s.startswith("karl.nugget") for s in scorers):
        return "quality: mean nugget recall (LLM judge) over questions"
    return "quality: mean score over questions"


def iter_series(keys) -> list[dict]:
    """Series in display order: SCENARIOS order, baseline variant before tagged variants.
    Each entry: scenario, variant, key, name, color, marker, linestyle."""
    keys = set(keys)
    variants = [""] + sorted({v for _, v in keys if v})
    out = []
    for scenario, name, color in SCENARIOS:
        for variant in variants:
            if (scenario, variant) not in keys:
                continue
            vi = variants.index(variant)
            out.append({
                "scenario": scenario, "variant": variant, "key": (scenario, variant),
                "name": name if not variant else f"{name} [{variant}]", "color": color,
                "marker": VARIANT_MARKERS[vi % len(VARIANT_MARKERS)],
                "linestyle": VARIANT_LINESTYLES[vi % len(VARIANT_LINESTYLES)],
            })
    return out


def load_runs(results_root: Path, min_rows: int | None = None) -> dict[tuple[str, str], dict[int, dict]]:
    """{(scenario, variant): {seed: run-level metrics}}."""
    runs: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    for key, per_seed in find_runs(results_root, min_rows).items():
        for seed, d in per_seed.items():
            rows = [json.loads(line) for line in (d / "results.jsonl").open() if line.strip()]
            dropped = [r["qid"] for r in rows if contaminated(d.name, r["qid"])]
            for qid in dropped:
                print(f"[contaminated] {d.name} {qid}: {CONTAMINATED_QUESTIONS[(run_label(d.name), qid)]}")
            rows = [r for r in rows if not contaminated(d.name, r["qid"])]
            if not rows:
                continue
            runs[key][seed] = {
                "dir": d.name,
                "n": len(rows),
                "n_ok": sum(1 for r in rows if not r["failed"]),
                "quality": statistics.mean(r["score"] for r in rows),
                # all-in $ per run; `cost` already includes the Bootstrap / Enrich collection agents'
                # spend (they run on the question's LLM client), broken out here for the table
                "cost": sum(r["cost"] for r in rows),
                "collection_agent_cost": sum(r.get("precompute_cost", 0.0) + r.get("enrich_cost", 0.0) for r in rows),
                "latency": sum(r["wall_s"] for r in rows if not latency_excluded(d.name, r["qid"])),
                "n_latency_excluded": sum(1 for r in rows if latency_excluded(d.name, r["qid"])),
            }
    return runs


def mean_std(vals: list[float]) -> tuple[float, float]:
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def summarize(runs: dict[tuple[str, str], dict[int, dict]]) -> list[dict]:
    out = []
    for s in iter_series(runs):
        per_seed = list(runs[s["key"]].values())
        q, c, lat = (mean_std([r[k] for r in per_seed]) for k in ("quality", "cost", "latency"))
        out.append({
            **s, "seeds": len(per_seed),
            "quality": q, "cost": c, "latency": lat,
            "collection_agent_cost": mean_std([r.get("collection_agent_cost", 0.0) for r in per_seed]),
            "ok": statistics.mean(r["n_ok"] for r in per_seed), "n": per_seed[0]["n"],
            "n_latency_excluded": sum(r["n_latency_excluded"] for r in per_seed),
        })
    return out


def plot(summary: list[dict], x_key: str, x_label: str, y_label: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for s in summary:
        (x, xerr), (y, yerr) = s[x_key], s["quality"]
        ax.errorbar(
            x, y, xerr=xerr, yerr=yerr, fmt=s["marker"], color=s["color"], ecolor=s["color"],
            markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5, elinewidth=1.2, capsize=3,
            label=f"{s['name']}  (n={s['seeds']})",
        )
        # the legend names the series; only flag points where some questions produced no answer
        if s["ok"] < s["n"]:
            ax.annotate(
                f"{s['ok']:.0f}/{s['n']} answered", (x, y), xytext=(9, 6),
                textcoords="offset points", fontsize=8, color=INK_2, ha="left", va="bottom",
            )
    n_excl = sum(s["n_latency_excluded"] for s in summary)
    if x_key == "latency" and n_excl:
        x_label += f"\n({n_excl} question(s) with provider outages excluded from latency; see LATENCY_EXCLUDE)"
    ax.set_xlabel(x_label, color=INK_2, fontsize=10)
    ax.set_ylabel(y_label, color=INK_2, fontsize=10)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(left=0)
    ax.grid(True, color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    # legend outside the axes (right), so it never covers the points; bbox_inches="tight" keeps it in the file
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results/officeqa/codex")
    ap.add_argument("--out-dir", default=None, help="defaults to <results-root>/plots")
    ap.add_argument("--benchmark", default=None, help="label for titles/file names; defaults to results/<benchmark>/codex")
    ap.add_argument("--min-rows", type=int, default=None, help="rows a run needs to count as complete (default: the largest run's)")
    args = ap.parse_args()
    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir) if args.out_dir else results_root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    bench = benchmark_label(results_root, args.benchmark)

    runs = load_runs(results_root, args.min_rows)
    summary = summarize(runs)
    if not summary:
        raise SystemExit(f"no codex ablation runs found under {results_root}")

    print(f"{'scenario':36s} {'seeds':>5s} {'answered':>9s} {'quality':>15s} {'cost $':>17s} {'(coll. agents $)':>16s} {'latency s':>19s} {'excl':>4s}")
    for s in summary:
        print(
            f"{s['name']:36s} {s['seeds']:5d} {s['ok']:5.1f}/{s['n']:<3d} "
            f"{s['quality'][0]:6.3f} +- {s['quality'][1]:5.3f} "
            f"{s['cost'][0]:8.3f} +- {s['cost'][1]:6.3f} "
            f"{s['collection_agent_cost'][0]:16.3f} "
            f"{s['latency'][0]:9.0f} +- {s['latency'][1]:7.0f} {s['n_latency_excluded']:4d}"
        )
    present = {run_label(r["dir"]) for per_seed in runs.values() for r in per_seed.values()}
    for (label, qid), why in LATENCY_EXCLUDE.items():
        if label in present:
            print(f"latency excluded: {label} {qid}: {why}")

    y_label = quality_label(results_root)
    cost_png = out_dir / f"codex_ablation_quality_vs_cost_{bench}.png"
    lat_png = out_dir / f"codex_ablation_quality_vs_latency_{bench}.png"
    plot(summary, "cost", "cost: total $ per run (OpenRouter, incl. embeddings)", y_label, f"Codex ablation on {bench}: quality vs cost", cost_png)
    plot(summary, "latency", "latency: total answer wall time per run (s)", y_label, f"Codex ablation on {bench}: quality vs latency", lat_png)
    print(f"wrote {cost_png} and {lat_png}")


if __name__ == "__main__":
    main()
