"""Decompose Codex per-question latency into steps (LLM calls) from the session rollout files.

A "step" is one model response: it starts when the previous step's tool outputs were returned
(the previous `token_count` event, or `task_started` for the first step) and its LLM phase ends at
the last `function_call` / `message` response item of that response. The tool phase runs from there
to the step's own `token_count` event (which Codex emits after the tool outputs are appended).
Only runs with rollouts are covered: `codex exec --ephemeral` (the parallel scenario) leaves none.

Usage (from qatfd/):
    python3 scripts/codex_step_latency.py [--results-root results/officeqa/codex]
Prints a per-scenario table (mean over seeds) and a per-run table, and writes
<results-root>/plots/codex_steps.jsonl with one row per step for further analysis.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from plot_codex_ablation import LATENCY_EXCLUDE, benchmark_label, find_runs, iter_series, latency_excluded


def _jsonl(path: Path):
    for line in path.open():
        line = line.strip()
        if line:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _ts(e: dict) -> float:
    return datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00")).timestamp()


def steps_from_rollout(rollout: Path) -> list[list[dict]]:
    """[[step, ...] per turn]. step = {llm_s, tool_s, ctx_tokens, n_tools, tools}."""
    turns: list[list[dict]] = []
    step_start = None
    llm_end = None
    tools: list[str] = []
    for e in _jsonl(rollout):
        p = e.get("payload") if isinstance(e.get("payload"), dict) else {}
        t, pt = e.get("type"), p.get("type")
        if t == "event_msg" and pt == "task_started":
            turns.append([])
            step_start, llm_end, tools = _ts(e), None, []
        elif t == "response_item" and pt in ("function_call", "message") and step_start is not None:
            if pt == "message" and p.get("role") == "user":
                continue
            if pt == "function_call":
                tools.append(p.get("name"))
            llm_end = _ts(e)
        elif t == "event_msg" and pt == "token_count" and step_start is not None and turns:
            end = _ts(e)
            if llm_end is None:
                llm_end = end
            last = (p.get("info") or {}).get("last_token_usage") or {}
            turns[-1].append({
                "llm_s": llm_end - step_start,
                "tool_s": end - llm_end,
                "ctx_tokens": int(last.get("total_tokens", 0)),
                "output_tokens": int(last.get("output_tokens", 0)),
                "n_tools": len(tools),
                "tools": tools,
            })
            step_start, llm_end, tools = end, None, []
    return turns


def codex_stdout(run_dir: Path, qid: str) -> Path | None:
    """Codex's raw `--json` stdout for `qid`: `traces/<qid>.codex.jsonl` on current runs, or the
    trace file itself on runs made before the dump was split out of it."""
    for name in (f"{qid}.codex.jsonl", f"{qid}.jsonl"):
        p = run_dir / "traces" / name
        if p.exists():
            return p
    return None


def load_run(scenario: str, run_dir: Path) -> list[dict]:
    """One row per question: wall_s from results.jsonl plus the step decomposition."""
    rows = {r["qid"]: r for r in _jsonl(run_dir / "results.jsonl")}
    order = [r["qid"] for r in sorted(rows.values(), key=lambda r: r["started_at"])]
    files = sorted((run_dir / "codex_home").rglob("rollout-*.jsonl"))
    per_q: dict[str, list[dict]] = {}
    if scenario.endswith("resume") and files:
        turns = steps_from_rollout(files[0])
        per_q = {q: turns[i] for i, q in enumerate(order) if i < len(turns)}
    else:
        by_thread = {f.name[-42:-6]: f for f in files}
        for q in order:
            trace = codex_stdout(run_dir, q)
            if trace is None:
                continue
            tid = next((e.get("thread_id") for e in _jsonl(trace) if e.get("type") == "thread.started"), None)
            if tid in by_thread:
                turns = steps_from_rollout(by_thread[tid])
                per_q[q] = [s for t in turns for s in t]
    out = []
    for i, q in enumerate(order):
        steps = per_q.get(q)
        if not steps:
            continue
        if latency_excluded(run_dir.name, q):
            print(f"[exclude] {run_dir.name} {q}: {LATENCY_EXCLUDE[(run_dir.name[:-16], q)]}")
            continue
        out.append({
            "scenario": scenario, "run": run_dir.name, "index": i + 1, "qid": q,
            "wall_s": rows[q]["wall_s"], "n_steps": len(steps),
            "llm_s": sum(s["llm_s"] for s in steps), "tool_s": sum(s["tool_s"] for s in steps),
            "n_tool_calls": sum(s["n_tools"] for s in steps),
            "peak_ctx": max(s["ctx_tokens"] for s in steps),
            "steps": steps,
        })
    return out


def fmt_row(label: str, qs: list[dict]) -> str:
    steps = [s for q in qs for s in q["steps"]]
    n = len(qs)

    def mean(xs):
        return statistics.mean(xs) if xs else float("nan")

    return (f"{label:<36} {n:>3}  {mean([q['wall_s'] for q in qs]):7.1f}  {mean([q['n_steps'] for q in qs]):6.1f}"
            f"  {mean([s['llm_s'] for s in steps]):7.2f}  {mean([s['tool_s'] for s in steps]):7.2f}"
            f"  {mean([q['llm_s'] for q in qs]):7.1f}  {mean([q['tool_s'] for q in qs]):7.1f}"
            f"  {mean([q['wall_s'] - q['llm_s'] - q['tool_s'] for q in qs]):7.1f}"
            f"  {mean([s['ctx_tokens'] for s in steps]) / 1000:6.0f}k")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results/officeqa/codex")
    ap.add_argument("--benchmark", default=None, help="label for the output file name; defaults to results/<benchmark>/codex")
    ap.add_argument("--min-rows", type=int, default=None, help="rows a run needs to count as complete (default: the largest run's)")
    args = ap.parse_args()
    root = Path(args.results_root)
    bench = benchmark_label(root, args.benchmark)

    runs = find_runs(root, args.min_rows)
    series = iter_series(runs)

    all_rows: list[dict] = []
    by_series: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_run: dict[str, list[dict]] = {}
    for s in series:
        for seed, d in sorted(runs[s["key"]].items()):
            rows = load_run(s["scenario"], d)
            if rows:
                by_run[d.name] = rows
                by_series[s["key"]].extend(rows)
                all_rows.extend(rows)

    hdr = (f"{'':<36} {'q':>3}  {'wall_s':>7}  {'steps':>6}  {'llm/step':>7}  {'tool/step':>7}"
           f"  {'llm_s/q':>7}  {'tool_s/q':>7}  {'other/q':>7}  {'ctx/step':>7}")
    print("per scenario (means over questions, all seeds pooled)")
    print(hdr)
    for s in series:
        if s["key"] in by_series:
            print(fmt_row(s["name"], by_series[s["key"]]))
    print("\nper run")
    print(hdr)
    for name, qs in by_run.items():
        print(fmt_row(name[6:-16], qs))

    # LLM latency vs context size, pooled over all steps
    steps = [s for q in all_rows for s in q["steps"]]
    print("\nLLM latency per step by context-size bucket (all rollout runs pooled)")
    edges = [0, 50_000, 100_000, 200_000, 400_000, 600_000, 800_000, 10**9]
    for lo, hi in zip(edges, edges[1:]):
        b = [s for s in steps if lo <= s["ctx_tokens"] < hi]
        if b:
            print(f"  {lo / 1000:>4.0f}k-{hi / 1000 if hi < 10**9 else 9999:>4.0f}k  n={len(b):5d}"
                  f"  llm_s mean={statistics.mean(s['llm_s'] for s in b):6.2f}"
                  f"  median={statistics.median(s['llm_s'] for s in b):6.2f}"
                  f"  out_tok mean={statistics.mean(s['output_tokens'] for s in b):6.0f}")

    out = root / "plots" / f"codex_steps_{bench}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for q in all_rows:
            for k, s in enumerate(q["steps"]):
                f.write(json.dumps({**{kk: vv for kk, vv in q.items() if kk != "steps"}, "step": k + 1, **s}) + "\n")
    print(f"\nwrote {out} ({len(steps)} steps)")


if __name__ == "__main__":
    main()
