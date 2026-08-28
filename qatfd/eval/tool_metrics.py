"""Per-configuration tool-call breakdown table from ``results/*/traces/<qid>.jsonl``.

This produces the second main table: **one row-group per benchmark**, with one row per
retrieval configuration found in the results — RAG-LLM plus each SearchAgent tool set
(``SearchAgent`` = grep+vector, ``QATFD`` = grep+sem, and any other point in the lattice).
The tool sets are one system configured by flags, so rows are identified from each run's
``config.yaml`` via ``variants.py``, not from its results/ directory name. Each row reports
where time goes inside that configuration's *retrieval* stage:

* avg. # search steps (agent steps; 1.0 for RAG-LLM),
* avg. # vector-search / grep / read-document / semantic-filter tool calls per question,
* avg. execution latency of each of those tool types.

Counts include every *attempted* tool call, including errored / hallucinated ones
(a step where the agent invokes a tool it doesn't actually have raises "Forbidden
function evaluation"). This is intentional: a nonzero count for such a tool is a
useful signal of a prompting bug (e.g. a sem-only configuration attempting
``search_corpus`` on QAMPARI). Errored calls return in ~0s, so they pull that
tool's latency average toward 0.
Latency is wall-clock under the runner's concurrent execution, so it reflects real
per-call wall time (including shared ChromaDB / thread-pool contention), not
isolated tool CPU time.

Where the numbers come from
---------------------------
The runner writes one structured event stream per question at
``<run>/traces/<qid>.jsonl`` (``skunk.trace.TraceEvent`` lines). Each event has an
``id`` (the emitting call site), a ``kind`` (system/user/call/assistant/tool_call/
observation/lifecycle/...), a float ``t`` (seconds since the question started), and
a structured ``data`` payload. Per question:

* **Stage split** — the answer stage begins at the *second* ``system`` event (the
  compute agent's system prompt); everything before it is the retrieval stage,
  which is all this table cares about.
* **Steps** — each executed agent step closes with an ``agent_step`` event
  (``kind="assistant"``) whose ``data`` carries the step's parsed tool ``code``,
  ``is_final`` flag, and any ``error``. Steps with code that is not a final answer
  are counted (see ``_iter_agent_steps``).
* **Which tool a step ran** — the ``data.tool`` of the step's ``kind="tool_call"``
  result event when the call executed (authoritative — e.g. ``semantic_filter``);
  for errored / hallucinated calls, the first non-comment line's ``funcname(`` in
  ``data.code``, so attempted calls are still counted.
* **Tool-execution latency** — the ``agent_step`` event's ``data.tool_latency_s``,
  measured by the agent around the sandboxed tool execution (LLM reasoning
  excluded); errored calls are timed too, since the sandbox captures their
  exception.
* **RAG-LLM** has no agent loop: its retrieval is a single direct vector search
  that emits an ``embed`` ``call`` event and a ``retrieved`` observation, so it
  contributes exactly one vector-search call, and latency = the ``retrieved``
  event's ``data.latency_s`` (falling back, on traces predating that field, to
  embed latency + the embed-to-``retrieved`` event gap).

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
    from .latex_tables import BENCHMARKS, _default_results_root, _tex_escape
    from .variants import Variant, judge_of, llm_of, model_label, variant_of
except ImportError:
    from latex_tables import BENCHMARKS, _default_results_root, _tex_escape
    from variants import Variant, judge_of, llm_of, model_label, variant_of

# The tool types broken out (in column order). ``search_corpus`` == vector search.
# ``semantic_filter`` is only present in variants whose tool set includes it.
TOOL_KEYS: list[str] = ["search_corpus", "grep_corpus", "read_document", "semantic_filter"]
TOOL_HEADER: dict[str, str] = {
    "search_corpus": "Vec.",
    "grep_corpus": "Grep",
    "read_document": "Read",
    "semantic_filter": "Sem.Filt.",
}

# Rows are the variants actually present in the results (see `variants.py`), ordered
# canonically, rather than a fixed system list: the SearchAgent tool sets are one system now,
# so which rows exist depends on what was run. RAG-LLM is still collapsed to one row (no k
# split) for this diagnostic table.


# --------------------------------------------------------------------------- #
# per-question trace parsing (structured skunk.trace.TraceEvent streams)
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
    """Index where the answer stage begins: the SECOND ``system`` event (the compute
    agent's system prompt) — by count, not position, because prelude events (the
    related-working-sets fetch) precede the retrieval agent's own system prompt.
    Retrieval stage == events[:idx]."""
    seen = 0
    for i, e in enumerate(events):
        if e.get("kind") == "system":
            seen += 1
            if seen == 2:
                return i
    return len(events)


def _action_name(code: str) -> str:
    """First non-comment line's ``funcname(`` — the step's primary tool call."""
    for line in code.split("\n"):
        t = line.strip()
        if t and not t.startswith("#"):
            m = re.match(r"([A-Za-z_]\w*)\s*\(", t)
            return m.group(1) if m else ""
    return ""


@dataclass
class AgentStep:
    """One executed agent step reconstructed from the structured event stream: the
    closing ``agent_step`` event's payload plus the step's tool result event
    (``kind="tool_call"``), matched positionally within the step's window (the
    events since the previous ``agent_step``)."""

    step: int | None
    turn: int | None
    code: str | None          # the step's parsed tool code (None on a misfire)
    is_final: bool
    error: str | None
    tool: str | None          # authoritative tool name, from the tool_call event
    tool_kwargs: dict | None  # structured kwargs, from the tool_call event
    latency: float | None     # data.tool_latency_s: sandboxed tool execution wall time
    step_latency_s: float | None  # data.step_latency_s: full step wall time (LLM gen included)


def _iter_agent_steps(events: list[dict]) -> list[AgentStep]:
    """Group a flat (retrieval-stage) event stream into executed agent steps.

    Each step closes with an ``agent_step`` event whose ``data`` carries the timing
    measured by the agent itself: ``tool_latency_s`` (around the sandboxed tool
    execution — present for errored / hallucinated calls too, since the sandbox
    captures their exceptions) and ``step_latency_s`` (the whole step). The step's
    ``kind="tool_call"`` result event, emitted iff the tool executed, supplies the
    authoritative tool name and structured kwargs. ``tool_latency_s`` is None on
    traces predating the field (then no latency sample is recorded) and on steps
    whose code never reached the sandbox (misfires, hard-context-limit rejections)."""
    steps: list[AgentStep] = []
    tool_event: dict | None = None
    for e in events:
        k = e.get("kind")
        if k == "tool_call":
            if tool_event is None:
                tool_event = e
        elif k == "assistant":
            d = e.get("data") or {}
            td = (tool_event or {}).get("data") or {}
            steps.append(AgentStep(
                step=e.get("step"), turn=e.get("turn"),
                code=d.get("code"), is_final=bool(d.get("is_final")), error=d.get("error"),
                tool=td.get("tool"), tool_kwargs=td.get("tool_kwargs"),
                latency=d.get("tool_latency_s"), step_latency_s=d.get("step_latency_s"),
            ))
            tool_event = None
    return steps


@dataclass
class QuestionToolMetrics:
    """One question's retrieval-stage tool usage."""

    n_steps: int = 0
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    latencies: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))


def _parse_search_agent(events: list[dict]) -> QuestionToolMetrics:
    retrieval = events[: _answer_start_idx(events)]
    m = QuestionToolMetrics()
    for s in _iter_agent_steps(retrieval):
        if s.is_final or not s.code:
            continue
        m.n_steps += 1
        # The tool_call event's tool name is authoritative when the call executed (it also
        # settles semantic_filter). For errored / hallucinated calls there is no tool_call
        # event, so fall back to parsing the emitted code — every ATTEMPTED call is counted.
        # This is deliberate: a nonzero count for a tool the configuration doesn't actually
        # include (which raises "Forbidden function evaluation") is a useful signal of a
        # prompting bug — the system prompt still references `search_corpus` in its
        # grep/prune tool docs even when the tool is off, and QAMPARI's entity questions
        # make the agent take the bait. Errored calls DO carry a tool_latency_s sample
        # (the sandbox captures their exception); they return in ~0s, pulling that tool's
        # latency average toward zero.
        name = s.tool or _action_name(s.code)
        if name not in TOOL_KEYS:
            continue
        m.counts[name] += 1
        if s.latency is not None:
            m.latencies[name].append(s.latency)
    return m


def _parse_rag_llm(events: list[dict]) -> QuestionToolMetrics:
    """RAG-LLM: one direct vector search — an ``embed`` ``call`` event followed by
    the ``retrieved`` observation. Latency = the ``retrieved`` event's measured
    ``latency_s`` when present, else embed latency + the embed-to-event gap."""
    m = QuestionToolMetrics(n_steps=1)
    m.counts["search_corpus"] = 1
    retrieved = next(
        (e for e in events if e.get("kind") == "observation" and e.get("id") == "retrieved" and e.get("t") is not None),
        None,
    )
    if retrieved is None:
        return m
    measured = (retrieved.get("data") or {}).get("latency_s")
    if measured is not None:
        m.latencies["search_corpus"].append(float(measured))
        return m
    embeds = [
        e for e in events
        if e.get("kind") == "call" and (e.get("data") or {}).get("call_site") == "embed"
        and e.get("t") is not None and e["t"] <= retrieved["t"]
    ]
    if embeds:
        embed = embeds[-1]
        embed_lat = float((embed.get("data") or {}).get("latency_s") or 0.0)
        m.latencies["search_corpus"].append(embed_lat + max(0.0, retrieved["t"] - embed["t"]))
    return m


def parse_run(traces_dir: Path, system: str) -> list[QuestionToolMetrics]:
    """Per-question tool metrics for one run. `system` selects the trace shape: RAG-LLM has no
    agent loop, every SearchAgent configuration does."""
    parser = _parse_rag_llm if system == "rag_llm" else _parse_search_agent
    return [parser(evs) for evs in _iter_question_events(traces_dir).values()]


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


@dataclass
class Cell:
    """Aggregated metrics for one (benchmark, variant) row."""

    n_runs: int = 0
    n_questions: int = 0
    avg_steps: float | None = None
    avg_count: dict[str, float] = field(default_factory=dict)  # tool -> avg calls/question
    avg_latency: dict[str, float | None] = field(default_factory=dict)  # tool -> avg per-call latency


@dataclass(frozen=True)
class RunId:
    """What a run dir is: its variant (row), its system (parser), and its models."""

    variant: Variant
    llm: str
    judge: str | None


def _run_id(run_dir: Path) -> RunId | None:
    """Read a run's identity from its persisted ``config.yaml``, or None if unreadable."""
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return None
    cfg = yaml.safe_load(config_path.read_text()) or {}
    llm = llm_of(cfg)
    return RunId(
        variant=variant_of(cfg, fallback_system=run_dir.parent.name),
        llm=llm,
        judge=judge_of(cfg, llm),
    )


def aggregate_cell(run_dirs: list[Path], system: str) -> Cell | None:
    """Average across runs of one (benchmark, variant): per-run means of steps and
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


def _run_dirs_by_variant(
    results_root: Path, benchmark: str, llm: str | None
) -> dict[tuple[Variant, str, str | None], list[Path]]:
    """Every traced run under ``results/<benchmark>/``, grouped by (variant, agent, judge).

    Walks all system dirs rather than one per row: the SearchAgent variants share the
    ``search_agent`` dir, so the tool set has to come from each run's config. The judge model
    is part of the key so two semantic-filter runs that differ only in `semantic_filter_llm_model`
    stay separate rows.
    """
    base = results_root / benchmark
    if not base.is_dir():
        return {}
    groups: dict[tuple[Variant, str, str | None], list[Path]] = defaultdict(list)
    for system_dir in sorted(base.iterdir()):
        if not system_dir.is_dir():
            continue
        for run_dir in sorted(system_dir.iterdir()):
            if not run_dir.is_dir() or not any((run_dir / "traces").glob("*.jsonl")):
                continue
            rid = _run_id(run_dir)
            if rid is None:
                continue
            if llm is not None and rid.llm != llm:
                continue
            groups[(rid.variant, rid.llm, rid.judge)].append(run_dir)
    return groups


def build_cells(
    results_root: Path,
    *,
    benchmarks: list[str],
    llm: str | None = None,
) -> dict[str, list[tuple[str, Cell]]]:
    """benchmark -> ordered list of (row label, Cell) for the variants that have runs."""
    out: dict[str, list[tuple[str, Cell]]] = {}
    for bench in benchmarks:
        rows: list[tuple[tuple, str, Cell]] = []
        groups = _run_dirs_by_variant(results_root, bench, llm)
        for (variant, agent, judge), run_dirs in groups.items():
            cell = aggregate_cell(run_dirs, variant.system)
            if cell is None:
                continue
            # Only disambiguate the row label by model when the table mixes models; with
            # --llm pinned (the recommended usage) the label stays just the variant.
            label = variant.label
            if llm is None:
                label = f"{label} ({model_label(agent, judge, short=True)})"
            rows.append(((variant.order, agent, judge or ""), label, cell))
        rows.sort(key=lambda r: r[0])
        out[bench] = [(label, cell) for _, label, cell in rows]
    return out


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _num(v: float | None, fmt: str) -> str:
    return "--" if v is None else fmt.format(v)


def render_tool_table(
    cells: dict[str, list[tuple[str, Cell]]],
    *,
    benchmarks: list[str],
    label: str = "tab:tool-breakdown",
) -> str:
    # Columns: Benchmark | Config | steps | Vec# Grep# Sem# | VecLat GrepLat SemLat
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
    first_count = 3  # 1-based column index of first count col (after Benchmark, Config, Steps)
    out.append(
        rf"    & & & \multicolumn{{{n_tools}}}{{c}}{{Avg.\ \# tool calls}} "
        rf"& \multicolumn{{{n_tools}}}{{c}}{{Avg.\ tool latency (s)}} \\"
    )
    out.append(
        rf"    \cmidrule(lr){{{first_count + 1}-{first_count + n_tools}}}"
        rf"\cmidrule(lr){{{first_count + n_tools + 1}-{first_count + 2 * n_tools}}}"
    )
    out.append("    Benchmark & Config & Steps & " + " & ".join(count_headers + lat_headers) + r" \\")
    out.append(r"    \midrule")

    for bi, bench in enumerate(benchmarks):
        bench_rows = cells.get(bench, [])
        bench_label = BENCHMARKS.get(bench, bench)
        if not bench_rows:
            out.append("    " + " & ".join([bench_label, r"\emph{no runs}"] + ["--"] * (1 + 2 * n_tools)) + r" \\")
        for ri, (row_label, cell) in enumerate(bench_rows):
            # Benchmark name on the first row of the group only; \midrule separates groups.
            row = [bench_label if ri == 0 else "", _tex_escape(row_label)]
            row.append(_num(cell.avg_steps, "{:.1f}"))
            row += [_num(cell.avg_count.get(t, 0.0), "{:.1f}") for t in TOOL_KEYS]
            row += [_num(cell.avg_latency.get(t), "{:.2f}") for t in TOOL_KEYS]
            out.append("    " + " & ".join(row) + r" \\" + f"  % n_runs={cell.n_runs}, n_q={cell.n_questions}")
        if bi != len(benchmarks) - 1:
            out.append(r"    \midrule")

    out.append(r"    \bottomrule")
    out.append(r"  \end{tabular}")
    out.append(
        r"  \caption{Retrieval-stage tool usage per configuration. Each row is one retrieval "
        r"tool set (\emph{SearchAgent} = grep+vector, \emph{QATFD} = grep+sem); "
        r"\texttt{read\_document} and \texttt{prune} are always available. \emph{Steps} is the "
        r"average number of agent steps (1.0 for RAG-LLM). \emph{Vec./Grep/Read/Sem.Filt.} are the "
        r"average number of vector-search, grep, read-document, and semantic-filter tool "
        r"calls per question, and their average per-call latency in seconds. A tool absent from a "
        r"configuration scores 0. Counts include every "
        r"attempted call, so a nonzero count for a tool the agent lacks (e.g.\ a sem-only "
        r"configuration hallucinating \texttt{search\_corpus} on QAMPARI) signals a prompting bug; such "
        r"errored calls return in ${\sim}0$s and pull that tool's latency toward zero. "
        r"Latency is measured around the sandboxed tool execution, under concurrent "
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
