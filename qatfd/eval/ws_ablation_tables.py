"""Working-set ablation tables from the run_ws_ablation_overnight.sh sweep.

Produces, for each phase (p1..p4) found under ``results/*/search_agent/``:

1. A **high-level table** (one row per ablation cell) from ``results.jsonl``:
   mean score, doc recall, page recall, adjusted page recall (a fully-correct answer
   counts as page recall 1.0), cost (all-in $/question), and latency (wall_s).
   Failed questions are included in the means (score/recall 0), and reported in `n`.

2. A **tool-usage table pair** from ``traces/<qid>.jsonl`` (parsing machinery shared
   with ``tool_metrics.py``): average agent steps and per-question counts of
   vector-search / grep / read-document / prune calls — vector and grep broken out by
   fetch-only, read-only, and fetch+read — plus the mean execution latency of each
   tool (per call, pooled across questions), broken out the same way. The latency
   split is what shows whether the per-working-set collection speeds up read-path
   vector/grep calls.

Notes on the tool breakdown:
* fetch/read mode is classified from ``read=True`` / ``fetch=True`` in the emitted
  call text (kwargs only — a positional ``True`` is not recognized); calls that name
  neither are shown under ``none`` (the tools force fetch+read in the no-working-set
  cell, error otherwise).
* Counts include attempted-but-errored calls (consistent with tool_metrics.py);
  errored calls return fast and pull latency averages down.
* Only questions present in ``results.jsonl`` are counted, so the script is safe to
  run while the sweep is still going (in-flight questions are skipped).
* If a phase+cell has several timestamped run dirs, the LATEST is used (noted).

Usage
-----
    python3 eval/ws_ablation_tables.py                # all phases -> stdout
    python3 eval/ws_ablation_tables.py --phases p1,p3
    python3 eval/ws_ablation_tables.py --results-root results
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

try:
    from .tool_metrics import _group_turns, _iter_question_events, _tool_code_str, _action_name
except ImportError:
    from tool_metrics import _group_turns, _iter_question_events, _tool_code_str, _action_name

CELL_ORDER = ["ws_coll_on_id_on", "ws_coll_on_id_off", "ws_coll_off_id_on", "ws_coll_off_id_off"]
MODED_TOOLS = ["search_corpus", "grep_corpus"]        # broken out by fetch/read mode
PLAIN_TOOLS = ["read_document", "prune"]              # no fetch/read args
MODES = ["fetch", "read", "fetch+read", "none"]

_RUN_DIR_RE = re.compile(r"^(p\d[a-z0-9_]*?)_(ws_coll_(?:on|off)_id_(?:on|off))_(\d{8}_\d{6})$")


# --------------------------------------------------------------------------- #
# Run discovery
# --------------------------------------------------------------------------- #

def discover_runs(results_root: Path) -> dict[str, dict[str, Path]]:
    """{phase: {cell: latest run dir}} across all benchmarks' search_agent results."""
    found: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for d in sorted(results_root.glob("*/search_agent/*")):
        m = _RUN_DIR_RE.match(d.name)
        if m and d.is_dir():
            found[m.group(1)][m.group(2)].append(d)
    picked: dict[str, dict[str, Path]] = {}
    for phase, cells in found.items():
        picked[phase] = {}
        for cell, dirs in cells.items():
            latest = max(dirs, key=lambda p: p.name)
            if len(dirs) > 1:
                print(f"[note] {phase}/{cell}: {len(dirs)} runs found, using latest ({latest.name})")
            picked[phase][cell] = latest
    return picked


# --------------------------------------------------------------------------- #
# High-level table (results.jsonl)
# --------------------------------------------------------------------------- #

def load_rows(run_dir: Path) -> list[dict]:
    path = run_dir / "results.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def high_level_row(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    n = len(rows)
    n_failed = sum(1 for r in rows if r.get("failed"))

    def rm(r: dict, key: str) -> float:
        m = r.get("recall_metrics") or {}
        if isinstance(m, str):
            try:
                m = json.loads(m)
            except json.JSONDecodeError:
                m = {}
        return float(m.get(key) or 0.0)

    scores = [float(r.get("score") or 0.0) for r in rows]
    page = [rm(r, "page_recall") for r in rows]
    adj_page = [1.0 if s == 1.0 else p for s, p in zip(scores, page)]
    return {
        "n": n,
        "n_failed": n_failed,
        "score": sum(scores) / n,
        "doc_recall": sum(rm(r, "doc_recall") for r in rows) / n,
        "page_recall": sum(page) / n,
        "adj_page_recall": sum(adj_page) / n,
        "cost": sum(float(r.get("cost") or 0.0) for r in rows) / n,
        "wall_s": sum(float(r.get("wall_s") or 0.0) for r in rows) / n,
    }


# --------------------------------------------------------------------------- #
# Tool-usage tables (traces/<qid>.jsonl)
# --------------------------------------------------------------------------- #

def _answer_start(events: list[dict]) -> int:
    """Index where the retrieval stage ends. tool_metrics' heuristic (first ``system``
    event at i>=1) breaks on these traces twice over: reuse phases prepend selector
    events (their PromptedCall emits its own ``system``), and the compute answerer is
    itself a code-running agent whose ``tool_code`` steps must not be counted. The
    robust anchor is SearchAgent's ``doc_ids_validated`` note — emitted exactly once,
    after the agent loop and any doc-id-correction turns, before compute begins. A
    trace without it (question failed in retrieval) is retrieval end-to-end."""
    for i, e in enumerate(events):
        if e.get("kind") == "note" and (e.get("message") or "").startswith("doc_ids_validated"):
            return i
    return len(events)


def _call_mode(code: str) -> str:
    """fetch/read mode of the step's primary call, from kwargs in the call text."""
    has_read = re.search(r"\bread\s*=\s*True\b", code) is not None
    has_fetch = re.search(r"\bfetch\s*=\s*True\b", code) is not None
    if has_fetch and has_read:
        return "fetch+read"
    if has_fetch:
        return "fetch"
    if has_read:
        return "read"
    return "none"


@dataclass
class ToolUsage:
    """Aggregated tool usage over one run's finished questions."""

    n_questions: int = 0
    steps: int = 0
    # counts[(tool, mode)] for moded tools; counts[(tool, "-")] for plain tools
    counts: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    latencies: dict[tuple[str, str], list[float]] = field(default_factory=lambda: defaultdict(list))

    def avg_count(self, tool: str, mode: str = "-") -> float:
        return self.counts[(tool, mode)] / self.n_questions if self.n_questions else 0.0

    def avg_latency(self, tool: str, mode: str = "-") -> float | None:
        lats = self.latencies[(tool, mode)]
        return sum(lats) / len(lats) if lats else None


def parse_tool_usage(run_dir: Path, finished_qids: set[str]) -> ToolUsage:
    usage = ToolUsage()
    for qid, events in _iter_question_events(run_dir / "traces").items():
        if qid not in finished_qids:
            continue
        usage.n_questions += 1
        retrieval = events[: _answer_start(events)]
        for turn in _group_turns(retrieval):
            if turn.tool_code is None:
                continue
            usage.steps += 1
            code = _tool_code_str(turn.tool_code.get("message", ""))
            name = _action_name(code)
            if name in MODED_TOOLS:
                key = (name, _call_mode(code))
            elif name in PLAIN_TOOLS:
                key = (name, "-")
            else:
                continue
            usage.counts[key] += 1
            first_t = next((r["t"] for r in turn.results if r.get("t") is not None), None)
            if first_t is not None and turn.tool_code.get("t") is not None:
                usage.latencies[key].append(max(0.0, first_t - turn.tool_code["t"]))
    return usage


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("$", r"\$").replace("#", r"\#")


def _latex_table(caption: str, headers: list[str], rows: list[list[str]]) -> str:
    """One booktabs table; first column left-aligned, the rest right-aligned."""
    colspec = "l" + "r" * (len(headers) - 1)
    out = [
        r"\begin{table}[t]",
        r"  \centering\small",
        rf"  \caption{{{_tex_escape(caption)}}}",
        rf"  \begin{{tabular}}{{{colspec}}}",
        r"    \toprule",
        "    " + " & ".join(_tex_escape(h) for h in headers) + r" \\",
        r"    \midrule",
    ]
    for row in rows:
        out.append("    " + " & ".join(_tex_escape(c) for c in row) + r" \\")
    out += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(out)


def _fmt_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    def line(vals: list[str]) -> str:
        return "  ".join(v.ljust(w) if i == 0 else v.rjust(w) for i, (v, w) in enumerate(zip(vals, widths)))
    sep = "  ".join("-" * w for w in widths)
    return "\n".join([line(headers), sep, *(line(r) for r in rows)])


def _f(x: float | None, nd: int = 3) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def render_phase(phase: str, cells: dict[str, Path], latex: bool = False) -> None:
    if latex:
        print(f"\n% =================== {phase} ===================")
    else:
        print(f"\n{'=' * 100}\n{phase}\n{'=' * 100}")

    # ---- table 1: high level ----
    rows_out = []
    usages: dict[str, ToolUsage] = {}
    for cell in CELL_ORDER:
        run_dir = cells.get(cell)
        if run_dir is None:
            rows_out.append([cell, "(no run)", "", "", "", "", "", ""])
            continue
        rows = load_rows(run_dir)
        hl = high_level_row(rows)
        if hl is None:
            rows_out.append([cell, "(empty)", "", "", "", "", "", ""])
            continue
        rows_out.append([
            cell, f"{hl['n']} ({hl['n_failed']} fail)", _f(hl["score"]), _f(hl["doc_recall"]),
            _f(hl["page_recall"]), _f(hl["adj_page_recall"]), _f(hl["cost"], 4), _f(hl["wall_s"], 1),
        ])
        usages[cell] = parse_tool_usage(run_dir, {str(r["qid"]) for r in rows})
    hl_headers = ["cell", "n", "score", "doc_rec", "page_rec", "adj_page_rec", "cost($)", "wall_s"]
    if latex:
        print(_latex_table(f"{phase}: high-level results per ablation cell.", hl_headers, rows_out))
    else:
        print("\n-- high level --")
        print(_fmt_table(hl_headers, rows_out))

    # ---- table 2a: tool-call counts / question ----
    count_rows, lat_rows = [], []
    for cell in CELL_ORDER:
        u = usages.get(cell)
        if u is None or u.n_questions == 0:
            count_rows.append([cell, "-"] + [""] * 10)
            lat_rows.append([cell] + [""] * 10)
            continue
        def cnt(tool: str, mode: str = "-") -> str:
            return _f(u.avg_count(tool, mode), 2)
        count_rows.append([
            cell, _f(u.steps / u.n_questions, 1),
            cnt("search_corpus", "fetch"), cnt("search_corpus", "read"), cnt("search_corpus", "fetch+read"), cnt("search_corpus", "none"),
            cnt("grep_corpus", "fetch"), cnt("grep_corpus", "read"), cnt("grep_corpus", "fetch+read"), cnt("grep_corpus", "none"),
            cnt("read_document"), cnt("prune"),
        ])
        def lat(tool: str, mode: str = "-") -> str:
            return _f(u.avg_latency(tool, mode), 2)
        lat_rows.append([
            cell,
            lat("search_corpus", "fetch"), lat("search_corpus", "read"), lat("search_corpus", "fetch+read"), lat("search_corpus", "none"),
            lat("grep_corpus", "fetch"), lat("grep_corpus", "read"), lat("grep_corpus", "fetch+read"), lat("grep_corpus", "none"),
            lat("read_document"), lat("prune"),
        ])
    count_headers = ["cell", "steps",
                     "vec:f", "vec:r", "vec:f+r", "vec:none",
                     "grep:f", "grep:r", "grep:f+r", "grep:none",
                     "read_doc", "prune"]
    lat_headers = ["cell",
                   "vec:f", "vec:r", "vec:f+r", "vec:none",
                   "grep:f", "grep:r", "grep:f+r", "grep:none",
                   "read_doc", "prune"]
    if latex:
        print(_latex_table(
            f"{phase}: tool calls per question (vec = search_corpus; f/r = fetch/read mode).",
            count_headers, count_rows,
        ))
        print(_latex_table(
            f"{phase}: mean tool latency, seconds per call (`-' = no calls).",
            lat_headers, lat_rows,
        ))
    else:
        print("\n-- tool calls / question (counts; vec = search_corpus) --")
        print(_fmt_table(count_headers, count_rows))
        print("\n-- mean tool latency, seconds / call ('-' = no calls) --")
        print(_fmt_table(lat_headers, lat_rows))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default=None, help="results root (default: <qatfd>/results)")
    ap.add_argument("--phases", default=None, help="comma-separated phase-name prefixes (e.g. p1,p3)")
    ap.add_argument("--latex", action="store_true", help="emit booktabs LaTeX tables instead of plain text")
    args = ap.parse_args()

    root = Path(args.results_root) if args.results_root else Path(__file__).resolve().parent.parent / "results"
    runs = discover_runs(root)
    if not runs:
        print(f"no phase run dirs found under {root}")
        return
    wanted = [p.strip() for p in args.phases.split(",")] if args.phases else None
    for phase in sorted(runs):
        if wanted and not any(phase == w or phase.startswith(w + "_") or phase.startswith(w) for w in wanted):
            continue
        render_phase(phase, runs[phase], latex=args.latex)


if __name__ == "__main__":
    main()
