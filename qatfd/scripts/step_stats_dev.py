"""Per-step statistics of the RETRIEVAL (search) agent for a dev-split sweep (results/<benchmark>/search_agent/
<prefix>_<cell>_s<seed>_<ts>/traces/<qid>.jsonl): steps per query, latency per step (split into LLM generation and
tool execution), cost per step (split into input-token and output-token spend).

Attribution: a question's trace interleaves several agents (search, compute-answer, bootstrap, enrich); every
`agent_step` event is credited to the agent of the most recent generation `call` event before it (`call_site`),
so only `search_agent` steps and calls are counted here. Per step, LLM latency is the summed `latency_s` of the
step's search-agent calls and tool latency is the step's `tool_latency_s`; the remainder of `step_latency_s` is
client-side work (parsing, observation formatting) and is shown as its own column. Cost per call is the trace's
`cost` (priced as the report is: uncached input at `in`, cached input at `cached`, output + thinking at `out`);
the output share is (out_tok + think_tok) x the model's `out` price and the input share is the remainder, so the
input share already reflects prompt-cache discounts. Failed questions are included (their steps are real steps).

Usage (from qatfd/):
    python3 scripts/step_stats_dev.py --prefix dev [--results-dir results/officeqa/search_agent] [--out tables.md]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

RUN_RE = re.compile(r"^(?P<prefix>[A-Za-z0-9]+)_(?P<cell>exp\d+_[a-z_]+?)_s(?P<seed>\d+)_(?P<stamp>\d{8}_\d{6})$")
CELL_NAMES = {"exp1_baseline": "baseline", "exp2_ws": "ws", "exp3_bs": "bs", "exp5_en": "en", "exp7_bs_en": "bs+en"}
SEARCH_SITE = "search_agent"
KEYS = ("questions", "steps", "step_s", "llm_s", "tool_s", "cost", "out_cost", "in_tok", "out_tok", "calls")


def load_prices(path: Path) -> dict[str, dict[str, float]]:
    return yaml.safe_load(path.read_text())["llm_prices"]


def out_price(prices: dict, model: str) -> float:
    if model in prices:
        return prices[model]["out"] / 1e6
    for k, p in prices.items():  # substring match, like the client
        if k in model:
            return p["out"] / 1e6
    raise KeyError(f"no price for {model}")


def question_stats(trace: Path, prices: dict) -> dict[str, float]:
    """Totals over the search agent's steps in one question trace."""
    tot = dict.fromkeys(KEYS, 0.0)
    tot["questions"] = 1
    site = None            # agent of the most recent generation call
    pending: list[dict] = []   # search-agent calls since the last agent_step
    for line in trace.open():
        e = json.loads(line)
        d = e.get("data") or {}
        if e["id"] == "call":
            if d.get("call_site") == "embed":
                continue
            site = d.get("call_site")
            if site == SEARCH_SITE:
                pending.append(d)
        elif e["id"] == "agent_step":
            if site == SEARCH_SITE:
                tot["steps"] += 1
                tot["step_s"] += d.get("step_latency_s") or 0.0
                tot["tool_s"] += d.get("tool_latency_s") or 0.0
                for c in pending:
                    tot["calls"] += 1
                    tot["llm_s"] += c.get("latency_s") or 0.0
                    tot["cost"] += c.get("cost") or 0.0
                    tot["in_tok"] += c.get("in_tok") or 0
                    tot["out_tok"] += (c.get("out_tok") or 0) + (c.get("think_tok") or 0)
                    tot["out_cost"] += ((c.get("out_tok") or 0) + (c.get("think_tok") or 0)) * out_price(prices, c["model"])
            pending = []
    return tot


def load(results_dir: Path, prefix: str, prices: dict, n_questions: int) -> dict[tuple[str, int], dict]:
    """(cell, seed) -> totals of the latest complete run."""
    runs: dict[tuple[str, int], tuple[str, Path]] = {}
    for run in sorted(results_dir.iterdir()):
        m = RUN_RE.match(run.name)
        if not m or m["prefix"] != prefix or not (run / "report.csv").exists():
            continue
        traces = list((run / "traces").glob("*.jsonl"))
        if len(traces) < n_questions:
            continue
        key = (m["cell"], int(m["seed"]))
        if key not in runs or m["stamp"] > runs[key][0]:
            runs[key] = (m["stamp"], run)
    out = {}
    for key, (_, run) in runs.items():
        tot = dict.fromkeys(KEYS, 0.0)
        for tr in sorted((run / "traces").glob("*.jsonl")):
            q = question_stats(tr, prices)
            for k in KEYS:
                tot[k] += q[k]
        tot["run"] = run.name
        out[key] = tot
    return out


def row(name: str, t: dict) -> str:
    steps = t["steps"] or 1
    other = t["step_s"] - t["llm_s"] - t["tool_s"]
    in_cost = t["cost"] - t["out_cost"]
    return (f"| {name} | {t['steps'] / t['questions']:.2f} | {t['step_s'] / steps:.2f} | {t['llm_s'] / steps:.2f} | "
            f"{t['tool_s'] / steps:.2f} | {other / steps:.2f} | {t['cost'] / steps * 100:.3f} | "
            f"{in_cost / steps * 100:.3f} | {t['out_cost'] / steps * 100:.3f} | "
            f"{t['in_tok'] / steps / 1000:.1f}k | {t['out_tok'] / steps:.0f} |")


HEADER = ("| config | steps / query | s / step | LLM s / step | tool s / step | other s / step | ¢ / step | "
          "input ¢ / step | output ¢ / step | input tok / step | output tok / step |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|")


def table(title: str, cells: dict[str, dict]) -> str:
    lines = [f"**{title}**", "", HEADER]
    for cell, name in CELL_NAMES.items():
        if cell in cells:
            lines.append(row(name, cells[cell]))
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results-dir", type=Path, default=Path("results/officeqa/search_agent"))
    p.add_argument("--prefix", default="dev")
    p.add_argument("--num-questions", type=int, default=33)
    p.add_argument("--prices", type=Path, default=Path("configs/inference/base.yaml"))
    p.add_argument("--out", type=Path, default=None, help="also write the markdown here")
    args = p.parse_args()

    prices = load_prices(args.prices)
    runs = load(args.results_dir, args.prefix, prices, args.num_questions)
    if not runs:
        sys.exit(f"no complete runs with prefix {args.prefix!r} under {args.results_dir}")
    seeds = sorted({s for _, s in runs})
    blocks = []
    for seed in seeds:
        blocks.append(table(f"seed {seed}", {c: t for (c, s), t in runs.items() if s == seed}))
    pooled: dict[str, dict] = defaultdict(lambda: dict.fromkeys(KEYS, 0.0))
    for (c, _), t in runs.items():
        for k in KEYS:
            pooled[c][k] += t[k]
    blocks.append(table(f"all seeds pooled ({', '.join(map(str, seeds))}; per-step means weighted by steps)", pooled))
    text = "\n\n".join(blocks) + "\n"
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
