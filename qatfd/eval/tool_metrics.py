"""Per-system tool-call breakdown table from ``results/*/traces/<qid>.jsonl``.

This produces the second main table: **one row-group per benchmark**, with the
three systems (RAG-LLM, SearchAgent, QATFD) repeated within each group, reporting
where time goes inside each system's *retrieval* stage:

* avg. # search steps (agent steps; 1.0 for RAG-LLM),
* avg. # vector-search / grep / read-document / semantic-filter tool calls per question,
* avg. execution latency of each of those tool types.

Counts include every *attempted* tool call, including errored / hallucinated ones
(a step where the agent invokes a tool it doesn't actually have raises "Forbidden
function evaluation"). This is intentional: a nonzero count for such a tool is a
useful signal of a prompting bug (e.g. QATFD attempting ``search_corpus`` on QAMPARI).
Because errored calls return in ~0s, they pull that tool's latency average toward 0.
Latency is wall-clock (``observation.t - tool_code.t``) under the runner's concurrent
execution, so it reflects real per-call wall time (including shared ChromaDB /
thread-pool contention), not isolated tool CPU time.

Where the numbers come from
---------------------------
The runner writes one structured event stream per question at
``<run>/traces/<qid>.jsonl`` (the same streams the trace viewer renders). Each
event has ``kind`` (system/user/call/assistant/note/observation/...), a float
``t`` (seconds since the question started), a ``message``, and optional
``data``. We reconstruct, per question, exactly what the viewer shows:

* **Stage split** — the answer stage begins at the *second* ``system`` event
  (mirrors the viewer's ``answerStartIdx``); everything before it is the
  retrieval stage, which is all this table cares about.
* **Turns/steps** — the retrieval stage is grouped into turns (an ``assistant``
  reply, the LLM ``call``\\ s that preceded it, the ``tool_code`` note it emitted,
  and the resulting ``observation``\\ s), mirroring the viewer's ``groupAgentTurns``.
* **Which tool a step ran** — the first non-comment line's ``funcname(`` in the
  emitted code (mirrors ``actionLabel``); this ignores tool names that appear
  only in the model's ``# comments``. A step is additionally counted as
  ``semantic_filter`` if it produced a structured ``semantic_filter n_in=...``
  observation (emitted by ``qatfd.systems.semfilter``).
* **Tool-execution latency** — ``observation.t - tool_code.t`` for that step,
  i.e. pure tool time with the LLM reasoning of the step excluded.
* **RAG-LLM** has no agent loop: its retrieval is a single ``search_corpus`` under
  an ``op == "retrieve"`` step (one embed ``call`` + a ``retrieved`` observation),
  so it contributes exactly one vector-search call, and latency = embed latency +
  the Chroma query time.

Usage
-----
    python qatfd/eval/tool_metrics.py                 # table -> stdout
    python qatfd/eval/tool_metrics.py --out tools.tex
    python qatfd/eval/tool_metrics.py --llm gemini-3.5-flash
    python qatfd/eval/tool_metrics.py --benchmarks officeqa,qampari
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Sibling module in this same folder. Relative import when imported as a package,
# plain import when run as a script (this dir is then on sys.path).
try:
    from .latex_tables import BENCHMARKS, _default_results_root
except ImportError:
    from latex_tables import BENCHMARKS, _default_results_root

# The tool types broken out (in column order). ``search_corpus`` == vector search.
# ``semantic_filter`` is QATFD-only (0 for RAG-LLM / SearchAgent).
TOOL_KEYS: list[str] = ["search_corpus", "grep_corpus", "read_document", "semantic_filter"]
TOOL_HEADER: dict[str, str] = {
    "search_corpus": "Vec.",
    "grep_corpus": "Grep",
    "read_document": "Read",
    "semantic_filter": "Sem.Filt.",
}

# Systems shown, in row order within each benchmark group. RAG-LLM is collapsed to
# one row (no k split) for this diagnostic table.
TOOL_TABLE_SYSTEMS: list[str] = ["rag_llm", "search_agent", "qatfd_search_agent"]
SYSTEM_ROW_LABEL: dict[str, str] = {
    "rag_llm": "RAG-LLM",
    "search_agent": "SearchAgent",
    "qatfd_search_agent": "QATFD",
}


# --------------------------------------------------------------------------- #
# per-question trace parsing (mirrors skunk/eval/trace_viewer/app.html)
# --------------------------------------------------------------------------- #


def _iter_question_events(traces_dir: Path) -> dict[str, list[dict]]:
    """Per-question event lists from traces/<qid>.jsonl, keyed by qid (sorted)."""
    by_qid: dict[str, list[dict]] = {}
    for path in sorted(traces_dir.glob("*.jsonl")):
        events: list[dict] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        by_qid[path.stem] = events
    return by_qid


def _answer_start_idx(events: list[dict]) -> int:
    """Index where the answer stage begins: the first ``system`` event at i>=1
    (the answer prompt). Retrieval stage == events[:idx]."""
    for i in range(1, len(events)):
        if events[i].get("kind") == "system":
            return i
    return len(events)


def _tool_code_str(message: str) -> str:
    """ "tool_code '<python repr>'" -> readable code (strip key + one quote layer)."""
    s = re.sub(r"^tool_code\s+", "", message or "")
    s = re.sub(r"^'([\s\S]*)'$", r"\1", s)
    s = re.sub(r'^"([\s\S]*)"$', r"\1", s)
    return s.replace("\\n", "\n").replace("\\t", "\t").replace("\\'", "'").replace('\\"', '"').replace("\\\\", "\\")


def _action_name(code: str) -> str:
    """First non-comment line's ``funcname(`` — the step's primary tool call."""
    for line in code.split("\n"):
        t = line.strip()
        if t and not t.startswith("#"):
            m = re.match(r"([A-Za-z_]\w*)\s*\(", t)
            return m.group(1) if m else ""
    return ""


@dataclass
class _Turn:
    lead: list[dict] = field(default_factory=list)
    tool_code: dict | None = None
    results: list[dict] = field(default_factory=list)


def _group_turns(events: list[dict]) -> list[_Turn]:
    """Reconstruct agent turns from a flat retrieval-stage event stream."""
    turns: list[_Turn] = []
    pending: list[dict] = []
    cur: _Turn | None = None
    for e in events:
        k = e.get("kind")
        msg = e.get("message") or ""
        if k == "assistant":
            cur = _Turn(lead=pending)
            pending = []
            turns.append(cur)
        elif k == "call":
            (cur.lead if (cur and not cur.results) else pending).append(e)
        elif k in ("observation", "error"):
            (cur.results if cur else pending).append(e)
        elif k == "note":
            if cur and not cur.results and msg.startswith("tool_code"):
                cur.tool_code = e
    return turns


@dataclass
class QuestionToolMetrics:
    """One question's retrieval-stage tool usage."""

    n_steps: int = 0
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    latencies: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))


def _parse_search_agent(events: list[dict]) -> QuestionToolMetrics:
    retrieval = events[: _answer_start_idx(events)]
    m = QuestionToolMetrics()
    for turn in _group_turns(retrieval):
        if turn.tool_code is None:
            continue
        m.n_steps += 1
        name = _action_name(_tool_code_str(turn.tool_code.get("message", "")))
        # Structured semantic_filter observation is authoritative when present.
        if any((r.get("message") or "").startswith("semantic_filter") for r in turn.results):
            name = "semantic_filter"
        if name not in TOOL_KEYS:
            continue
        # Count every ATTEMPTED tool call, including errored / hallucinated ones. This is
        # deliberate: a nonzero count for a tool the agent doesn't actually have (e.g. QATFD
        # calling `search_corpus`, which raises "Forbidden function evaluation") is a useful
        # signal of a prompting bug — the QATFD system prompt still references `search_corpus`
        # in its grep/prune tool docs, and QAMPARI's entity questions make the agent take the
        # bait. Errored calls return in ~0s, so they pull the latency average toward 0.
        m.counts[name] += 1
        # Latency is to the tool's OWN result — the first observation/error after the
        # tool_code — NOT max(results.t): _group_turns also attaches trailing error events
        # (e.g. a later "no fenced block" model misfire) to this turn, and those would
        # otherwise inflate the latency by tens of seconds.
        first_t = next((r["t"] for r in turn.results if r.get("t") is not None), None)
        if first_t is not None and turn.tool_code.get("t") is not None:
            m.latencies[name].append(max(0.0, first_t - turn.tool_code["t"]))
    return m


def _parse_rag_llm(events: list[dict]) -> QuestionToolMetrics:
    """RAG-LLM: exactly one vector search under an ``op == 'retrieve'`` step."""
    m = QuestionToolMetrics(n_steps=1)
    retrieve = [e for e in events if e.get("op") == "retrieve"]
    obs = [e for e in retrieve if e.get("kind") == "observation" and e.get("t") is not None]
    calls = [e for e in retrieve if e.get("kind") == "call" and e.get("t") is not None]
    m.counts["search_corpus"] = 1
    if obs and calls:
        lm = re.search(r"latency_s=([\d.]+)", calls[0].get("message") or "")
        embed_lat = float(lm.group(1)) if lm else 0.0
        query_time = max(0.0, max(o["t"] for o in obs) - calls[0]["t"])
        m.latencies["search_corpus"].append(embed_lat + query_time)
    return m


def parse_run(traces_dir: Path, system: str) -> list[QuestionToolMetrics]:
    """Per-question tool metrics for one run."""
    parser = _parse_rag_llm if system == "rag_llm" else _parse_search_agent
    return [parser(evs) for evs in _iter_question_events(traces_dir).values()]


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


@dataclass
class Cell:
    """Aggregated metrics for one (benchmark, system) row."""

    n_runs: int = 0
    n_questions: int = 0
    avg_steps: float | None = None
    avg_count: dict[str, float] = field(default_factory=dict)  # tool -> avg calls/question
    avg_latency: dict[str, float | None] = field(default_factory=dict)  # tool -> avg per-call latency


def _run_llm(config_path: Path) -> str | None:
    if not config_path.exists():
        return None
    cfg = yaml.safe_load(config_path.read_text()) or {}
    systems = cfg.get("systems", {}) or {}
    return systems.get("llm_model") or systems.get("agent_model_id")


def aggregate_cell(run_dirs: list[Path], system: str) -> Cell | None:
    """Average across runs of one (benchmark, system): per-run means of steps and
    per-tool counts are averaged equally; per-tool latency pools all calls in a run,
    then averages run-level means."""
    per_run_steps: list[float] = []
    per_run_counts: dict[str, list[float]] = defaultdict(list)
    per_run_latency: dict[str, list[float]] = defaultdict(list)
    total_q = 0

    for run_dir in run_dirs:
        metrics = parse_run(run_dir / "traces", system)
        if not metrics:
            continue
        total_q += len(metrics)
        per_run_steps.append(sum(m.n_steps for m in metrics) / len(metrics))
        for tool in TOOL_KEYS:
            per_run_counts[tool].append(sum(m.counts.get(tool, 0) for m in metrics) / len(metrics))
            calls = [lat for m in metrics for lat in m.latencies.get(tool, [])]
            if calls:
                per_run_latency[tool].append(sum(calls) / len(calls))

    if not per_run_steps:
        return None

    cell = Cell(n_runs=len(per_run_steps), n_questions=total_q)
    cell.avg_steps = sum(per_run_steps) / len(per_run_steps)
    for tool in TOOL_KEYS:
        counts = per_run_counts[tool]
        cell.avg_count[tool] = (sum(counts) / len(counts)) if counts else 0.0
        lats = per_run_latency[tool]
        cell.avg_latency[tool] = (sum(lats) / len(lats)) if lats else None
    return cell


def _run_dirs_for(results_root: Path, benchmark: str, system: str, llm: str | None) -> list[Path]:
    base = results_root / benchmark / system
    if not base.is_dir():
        return []
    dirs = []
    for run_dir in sorted(base.iterdir()):
        if not any((run_dir / "traces").glob("*.jsonl")):
            continue
        if llm is not None and _run_llm(run_dir / "config.yaml") != llm:
            continue
        dirs.append(run_dir)
    return dirs


def build_cells(
    results_root: Path,
    *,
    benchmarks: list[str],
    llm: str | None = None,
) -> dict[str, dict[str, Cell | None]]:
    """benchmark -> system -> Cell (or None if no runs)."""
    out: dict[str, dict[str, Cell | None]] = {}
    for bench in benchmarks:
        out[bench] = {}
        for system in TOOL_TABLE_SYSTEMS:
            run_dirs = _run_dirs_for(results_root, bench, system, llm)
            out[bench][system] = aggregate_cell(run_dirs, system) if run_dirs else None
    return out


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _num(v: float | None, fmt: str) -> str:
    return "--" if v is None else fmt.format(v)


def render_tool_table(
    cells: dict[str, dict[str, Cell | None]],
    *,
    benchmarks: list[str],
    label: str = "tab:tool-breakdown",
) -> str:
    # Columns: Benchmark | System | steps | Vec# Grep# Sem# | VecLat GrepLat SemLat
    n_tools = len(TOOL_KEYS)
    colspec = "ll" + "r" + "r" * n_tools + "r" * n_tools
    count_headers = [TOOL_HEADER[t] for t in TOOL_KEYS]
    lat_headers = [TOOL_HEADER[t] for t in TOOL_KEYS]

    out: list[str] = []
    out.append("% --- auto-generated by qatfd/eval/tool_metrics.py ---")
    out.append("% requires \\usepackage{booktabs}")
    out.append(r"\begin{table}[t]")
    out.append(r"  \centering")
    out.append(r"  \small")
    out.append(rf"  \begin{{tabular}}{{{colspec}}}")
    out.append(r"    \toprule")
    # grouped header
    first_count = 3  # 1-based column index of first count col (after Benchmark, System, Steps)
    out.append(
        rf"    & & & \multicolumn{{{n_tools}}}{{c}}{{Avg.\ \# tool calls}} "
        rf"& \multicolumn{{{n_tools}}}{{c}}{{Avg.\ tool latency (s)}} \\"
    )
    out.append(
        rf"    \cmidrule(lr){{{first_count + 1}-{first_count + n_tools}}}"
        rf"\cmidrule(lr){{{first_count + n_tools + 1}-{first_count + 2 * n_tools}}}"
    )
    out.append("    Benchmark & System & Steps & " + " & ".join(count_headers + lat_headers) + r" \\")
    out.append(r"    \midrule")

    for bi, bench in enumerate(benchmarks):
        systems = cells.get(bench, {})
        bench_label = BENCHMARKS.get(bench, bench)
        for si, system in enumerate(TOOL_TABLE_SYSTEMS):
            cell = systems.get(system)
            # Benchmark name on the first system row only; \midrule separates groups.
            bench_cell = bench_label if si == 0 else ""
            row = [bench_cell, SYSTEM_ROW_LABEL[system]]
            if cell is None:
                row += ["--"] * (1 + 2 * n_tools)
            else:
                row.append(_num(cell.avg_steps, "{:.1f}"))
                row += [_num(cell.avg_count.get(t, 0.0), "{:.1f}") for t in TOOL_KEYS]
                row += [_num(cell.avg_latency.get(t), "{:.2f}") for t in TOOL_KEYS]
            comment = ""
            if cell is not None:
                comment = f"  % n_runs={cell.n_runs}, n_q={cell.n_questions}"
            out.append("    " + " & ".join(row) + r" \\" + comment)
        if bi != len(benchmarks) - 1:
            out.append(r"    \midrule")

    out.append(r"    \bottomrule")
    out.append(r"  \end{tabular}")
    out.append(
        r"  \caption{Retrieval-stage tool usage per system. \emph{Steps} is the average "
        r"number of agent steps (1.0 for RAG-LLM). \emph{Vec./Grep/Read/Sem.Filt.} are the "
        r"average number of vector-search, grep, read-document, and semantic-filter tool "
        r"calls per question, and their average per-call latency in seconds. Semantic filter "
        r"is a QATFD-only tool (0 for RAG-LLM and SearchAgent). Counts include every "
        r"attempted call, so a nonzero count for a tool the agent lacks (e.g.\ QATFD's "
        r"hallucinated \texttt{search\_corpus} on QAMPARI) signals a prompting bug; such "
        r"errored calls return in ${\sim}0$s and pull that tool's latency toward zero. "
        r"Latency is wall-clock time between the tool call and its result under concurrent "
        r"execution, so it includes shared-index and thread-pool contention, not just the "
        r"tool's own CPU time. Values are averaged across repeated runs of each configuration.}"
    )
    out.append(rf"  \label{{{label}}}")
    out.append(r"\end{table}")
    return "\n".join(out)


def generate_tool_table(
    results_root: Path | str,
    *,
    benchmarks: list[str] | None = None,
    llm: str | None = None,
) -> str:
    results_root = Path(results_root)
    bench_keys = benchmarks if benchmarks is not None else list(BENCHMARKS)
    cells = build_cells(results_root, benchmarks=bench_keys, llm=llm)
    return render_tool_table(cells, benchmarks=bench_keys, label="tab:tool-breakdown")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", type=Path, default=_default_results_root())
    parser.add_argument("--out", type=str, default="-", help="output .tex file, or '-' for stdout")
    parser.add_argument("--benchmarks", type=str, default=None, help="comma-separated benchmark keys")
    parser.add_argument("--llm", type=str, default=None, help="only include runs for this llm_model")
    args = parser.parse_args(argv)

    if not args.results_root.exists():
        print(f"[tool_metrics] results root not found: {args.results_root}", file=sys.stderr)
        return 1

    bench_list = [b.strip() for b in args.benchmarks.split(",")] if args.benchmarks else None
    table = generate_tool_table(args.results_root, benchmarks=bench_list, llm=args.llm)
    if args.out == "-":
        print(table)
    else:
        Path(args.out).write_text(table + "\n", encoding="utf-8")
        print(f"[tool_metrics] wrote table to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
