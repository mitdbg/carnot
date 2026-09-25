"""Plot the Codex ablation as a function of question index (position in the shuffled answering order).

Top panel: cumulative quality score (running sum of per-question scores; a failed row scores 0) at each
index, averaged across shuffle seeds, one line per scenario (shaded band = +-1 sample std across seeds).
Bottom panel: the largest single-request context (tokens) seen while answering the question at that index,
again averaged across seeds. Context comes from Codex's own session rollout files
(`<run>/codex_home/sessions/**/rollout-*.jsonl`, `token_count` events); the parallel scenario runs
`codex exec --ephemeral` and leaves no rollouts, so it has no context line.

The question at index i is the same across scenarios for a given seed (same shuffle), so the lines are
directly comparable. Sequential runs are ordered by `started_at`; a parallel run borrows the order of the
sequential run with the same seed (falling back to its own `started_at` order).

Usage (from qatfd/):
    python3 scripts/plot_codex_by_index.py [--results-root results/officeqa/codex] [--out-dir <dir>]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from plot_codex_ablation import INK, INK_2, MUTED, SURFACE, benchmark_label, contaminated, find_runs, iter_series, quality_label

SEQUENTIAL = ("seq_resume", "seq_shell", "seq_shell_resume", "sa_seq")


def _jsonl(path: Path):
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def load_rows(run_dir: Path) -> dict[str, dict]:
    """Rows minus the questions listed in plot_codex_ablation.CONTAMINATED_QUESTIONS for this run (that
    seed's curve is one index shorter; mean_band averages whatever seeds have a value at each index)."""
    return {r["qid"]: r for r in _jsonl(run_dir / "results.jsonl") if not contaminated(run_dir.name, r["qid"])}


def started_order(rows: dict[str, dict]) -> list[str]:
    return [r["qid"] for r in sorted(rows.values(), key=lambda r: r["started_at"])]


def rollout_files(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "codex_home").rglob("rollout-*.jsonl"))


def peak_context_by_turn(rollout: Path) -> list[int]:
    """Largest single-request context (last_token_usage.total_tokens) per turn, in turn order."""
    peaks: list[int] = []
    for e in _jsonl(rollout):
        p = e.get("payload") if isinstance(e.get("payload"), dict) else {}
        if e.get("type") != "event_msg":
            continue
        if p.get("type") == "task_started":
            peaks.append(0)
        elif p.get("type") == "token_count" and peaks:
            last = (p.get("info") or {}).get("last_token_usage") or {}
            peaks[-1] = max(peaks[-1], int(last.get("total_tokens", 0)))
    return peaks


def codex_stdout(run_dir: Path, qid: str) -> Path | None:
    """Codex's raw `--json` stdout for `qid`: `traces/<qid>.codex.jsonl` on current runs, or the
    trace file itself on runs made before the dump was split out of it."""
    for name in (f"{qid}.codex.jsonl", f"{qid}.jsonl"):
        p = run_dir / "traces" / name
        if p.exists():
            return p
    return None


def context_by_qid(scenario: str, run_dir: Path, order: list[str]) -> dict[str, int]:
    """{qid: peak request tokens}. Resume runs: one rollout, turn k <-> order[k]. Non-resume
    sequential runs: one rollout per question, matched via the thread id in the codex stdout dump."""
    files = rollout_files(run_dir)
    if not files:
        return {}
    if scenario.endswith("resume"):
        peaks = peak_context_by_turn(files[0])
        return {qid: peaks[i] for i, qid in enumerate(order) if i < len(peaks)}
    # rollout-<timestamp>-<36-char thread uuid>.jsonl
    by_thread = {f.name[-42:-6]: f for f in files}
    out: dict[str, int] = {}
    for qid in order:
        trace = codex_stdout(run_dir, qid)
        if trace is None:
            continue
        tid = next((e.get("thread_id") for e in _jsonl(trace) if e.get("type") == "thread.started"), None)
        f = by_thread.get(tid)
        if f is None:
            continue
        peaks = peak_context_by_turn(f)
        out[qid] = max(peaks) if peaks else 0
    return out


def collect(results_root: Path, min_rows: int | None = None) -> dict[tuple[str, str], dict[int, dict]]:
    """{(scenario, variant): {seed: {"cum": [..], "ctx": [..|None], "order": [...]}}}."""
    runs = find_runs(results_root, min_rows)
    # answering order per seed, taken from any sequential run with that seed (same shuffle for all)
    seq_order: dict[int, list[str]] = {}
    for (scenario, _), per_seed in runs.items():
        if scenario in SEQUENTIAL:
            for seed, d in per_seed.items():
                seq_order.setdefault(seed, started_order(load_rows(d)))

    data: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    for key, per_seed in runs.items():
        scenario = key[0]
        for seed, d in per_seed.items():
            rows = load_rows(d)
            order = seq_order.get(seed) if scenario.endswith("par") else started_order(rows)
            if order is None or set(order) != set(rows):
                order = started_order(rows)
            scores = [0.0 if rows[q]["failed"] else float(rows[q]["score"]) for q in order]
            cum, acc = [], 0.0
            for s in scores:
                acc += s
                cum.append(acc)
            ctx = context_by_qid(scenario, d, order)
            data[key][seed] = {
                "dir": d.name,
                "order": order,
                "cum": cum,
                "ctx": [ctx.get(q) for q in order],
            }
    return data


def mean_band(series: list[list[float | None]]) -> tuple[list[float], list[float], list[float]]:
    """Index-wise mean and +-1 std over the seeds that have a value at that index."""
    n = max(len(s) for s in series)
    means, los, his = [], [], []
    for i in range(n):
        vals = [s[i] for s in series if i < len(s) and s[i] is not None]
        if not vals:
            means.append(float("nan")), los.append(float("nan")), his.append(float("nan"))
            continue
        m = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        means.append(m), los.append(m - sd), his.append(m + sd)
    return means, los, his


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color="#e6e4df", linewidth=0.6)
    ax.set_axisbelow(True)


def plot(data: dict[tuple[str, str], dict[int, dict]], out_path: Path, bench: str, y_label: str) -> None:
    fig, (ax_q, ax_c) = plt.subplots(2, 1, figsize=(8.5, 8), dpi=160, sharex=True,
                                     gridspec_kw={"height_ratios": [1.15, 1], "hspace": 0.12})
    fig.patch.set_facecolor(SURFACE)
    style(ax_q), style(ax_c)

    for s in iter_series(data):
        per_seed, name, color, ls = data[s["key"]], s["name"], s["color"], s["linestyle"]
        seeds = sorted(per_seed)
        x = list(range(1, max(len(per_seed[sd]["cum"]) for sd in seeds) + 1))

        m, lo, hi = mean_band([per_seed[sd]["cum"] for sd in seeds])
        ax_q.plot(x, m, color=color, linewidth=2, linestyle=ls, label=f"{name}  (n={len(seeds)} seeds)")
        ax_q.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)

        ctx_series = [per_seed[sd]["ctx"] for sd in seeds if any(v is not None for v in per_seed[sd]["ctx"])]
        if ctx_series:
            m, lo, hi = mean_band(ctx_series)
            ax_c.plot(x, m, color=color, linewidth=2, linestyle=ls, label=name)
            ax_c.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)
        elif not s["scenario"].startswith("sa_"):
            # codex runs without rollouts (parallel is --ephemeral); the search agent has no codex
            # rollouts at all, so it simply has no entry in the context panel
            ax_c.plot([], [], color=color, linewidth=2, linestyle=ls, alpha=0.4,
                      label=f"{name}  (ephemeral: no rollouts)")

    ax_q.set_ylabel(y_label, color=INK, fontsize=9.5)
    ax_q.set_title(f"Codex on {bench}: cumulative quality and peak request context by question index",
                   color=INK, fontsize=11, loc="left", pad=10)
    ax_q.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    ax_c.axhline(945_000, color=MUTED, linewidth=0.8, linestyle="--")
    ax_c.text(ax_c.get_xlim()[1], 945_000, "auto-compact threshold (90% of 1.05M) ", color=MUTED, fontsize=7.5,
              va="bottom", ha="right")
    ax_c.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax_c.set_ylabel("peak single-request context (tokens)", color=INK, fontsize=9.5)
    ax_c.set_xlabel("question index in answering order (1 = first)", color=INK, fontsize=9.5)
    ax_c.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax_c.set_xlim(0.5, None)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out_path}")


def print_table(data: dict[tuple[str, str], dict[int, dict]]) -> None:
    keys = [s["key"] for s in iter_series(data)]
    n = max(len(v["cum"]) for k in keys for v in data[k].values())
    print("\nmean cumulative score by index (then mean peak context, k tokens)")
    print("idx  " + "  ".join(f"{'_'.join(filter(None, k)):>17}" for k in keys))
    for i in range(n):
        cells = []
        for k in keys:
            cum = [v["cum"][i] for v in data[k].values() if i < len(v["cum"])]
            ctx = [v["ctx"][i] for v in data[k].values() if i < len(v["ctx"]) and v["ctx"][i] is not None]
            c = f"{statistics.mean(cum):5.2f}" if cum else "    -"
            t = f"{statistics.mean(ctx) / 1000:6.0f}k" if ctx else "      -"
            cells.append(f"{c} {t}".rjust(17))
        print(f"{i + 1:>3}  " + "  ".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results/officeqa/codex")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--benchmark", default=None, help="label for the title/file name; defaults to results/<benchmark>/codex")
    ap.add_argument("--min-rows", type=int, default=None, help="rows a run needs to count as complete (default: the largest run's)")
    args = ap.parse_args()
    root = Path(args.results_root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "plots"
    bench = benchmark_label(root, args.benchmark)

    data = collect(root, args.min_rows)
    if not data:
        raise SystemExit(f"no codex runs found under {root}")
    for s in iter_series(data):
        for seed, v in sorted(data[s["key"]].items()):
            n_ctx = sum(1 for c in v["ctx"] if c is not None)
            print(f"{s['name']:>36} s{seed}: {v['dir']}  rows={len(v['cum'])}  ctx_rows={n_ctx}")
    nugget = quality_label(root).startswith("quality: mean nugget")
    y_label = "cumulative nugget recall (sum of per-question recall)" if nugget else "cumulative score (sum of per-question scores)"
    plot(data, out_dir / f"codex_ablation_by_index_{bench}.png", bench, y_label)
    print_table(data)


if __name__ == "__main__":
    main()
