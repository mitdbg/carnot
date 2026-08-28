"""Ablation tables from ``search_agent`` runs.

The ``search_agent`` system (qatfd/systems/search_agent.py) runs with a config-chosen retrieval
tool set — any subset of vector-search / grep / semantic-filter, with read_document + prune
always on — and an optional separate model for the semantic-filter judge
(``semantic_filter_llm_model``). Every ablation is therefore a `search_agent` run: this script reads
all of them under ``results/`` and reports, per configuration, the *usefulness* metrics
(accuracy, recall, cost, latency), so the tools can be compared head-to-head.

Each configuration is identified by three things read from the run's ``config.yaml`` (resolved
by ``variants.py``, which the main tables share):
  * the tool set (from ``include_search_corpus`` / ``include_grep_corpus`` /
    ``include_semantic_filter``), with a non-default working-set mode noted in the label,
  * the agent model (``inference.llm_model``),
  * the semantic-filter judge model (``semantic_filter_llm_model``, or the agent model when unset;
    ``--`` when there is no semantic filter).
Runs that share all three are averaged (so repeats collapse into one row).

Two tables are produced:
  1. the full breakdown — one row per (tool set, agent, judge) with all metrics;
  2. an accuracy pivot — tool set (rows) x model column (cols), for a quick cross-model read.

Usage (interpreter needs pyyaml, e.g. the skunk venv):
    python3 qatfd/eval/ablation_tables.py                        # text tables -> stdout
    python3 qatfd/eval/ablation_tables.py --benchmark officeqa
    python3 qatfd/eval/ablation_tables.py --latex --out ablation.tex
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Sibling module: relative import as a package, plain import when run as a script.
try:
    from .latex_tables import _default_results_root, discover_runs
    from .variants import AGENT_SYSTEM, model_sort_key
except ImportError:
    from latex_tables import _default_results_root, discover_runs
    from variants import AGENT_SYSTEM, model_sort_key

# Every ablation is now a `search_agent` run whose tool flags differ (the dedicated
# `ablation_search_agent` system is gone), so this reads every SearchAgent run and lets the
# config identify the point in the tool lattice. The canonical order + labels live in
# `variants.py` and are shared with the main tables.
ABLATION_SYSTEM = AGENT_SYSTEM

# Preferred model column order (ascending capability/size, roughly); unknown models sort after.
_MODEL_ORDER: list[str] = [
    "qwen/qwen3.6-27b",
    "qwen/qwen3.6-35b-a3b",
    "google/gemini-3.1-flash-lite",
    "google/gemini-3.5-flash",
]

# Metrics displayed, in column order: (report.csv key, header, scale, format). `adj_page_recall`
# is derived per-question in `_adj_page_recall` (not a report.csv column).
_METRICS: list[tuple[str, str, float, str]] = [
    ("score", "acc%", 100.0, "{:.1f}"),
    ("page_recall", "pageR%", 100.0, "{:.1f}"),
    ("adj_page_recall", "adjPR%", 100.0, "{:.1f}"),
    ("doc_recall", "docR%", 100.0, "{:.1f}"),
    ("cost", "$/q", 1.0, "{:.4f}"),
    ("wall_s", "lat/q", 1.0, "{:.1f}"),
]


def _to_float(x: object) -> float | None:
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _adj_page_recall(run_dir: Path) -> float | None:
    """Per-run mean of an *adjusted* page recall: each question scores its measured
    `page_recall`, EXCEPT a fully-correct question (score == 1) is credited 1.0 even if its
    measured page recall is lower. Rationale: a correct answer implies a usable page was
    retrieved, but the labelled gold page is not the only page that can answer the question, so
    plain page_recall undercounts retrieval. Returns None for benchmarks with no `page_recall`
    column (nothing to adjust)."""
    report = run_dir / "report.csv"
    if not report.exists():
        return None
    with report.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "page_recall" not in reader.fieldnames:
            return None
        vals: list[float] = []
        for r in reader:
            pr = _to_float(r.get("page_recall"))
            if pr is None:
                continue
            sc = _to_float(r.get("score"))
            vals.append(1.0 if (sc is not None and sc >= 1.0) else pr)
    return sum(vals) / len(vals) if vals else None


def _model_key(m: str) -> tuple[int, str]:
    return model_sort_key(m, _MODEL_ORDER)


def _short(model: str | None) -> str:
    return "--" if model is None else model.split("/")[-1]


@dataclass
class AblationCell:
    """One (config, agent, judge) identity, averaged across its runs."""

    order: int
    config_label: str
    agent: str
    judge: str | None
    means: dict[str, float] = field(default_factory=dict)  # metric key -> mean across runs
    n_runs: int = 0
    n_questions: int = 0

    def sort_key(self) -> tuple:
        return (self.order, _model_key(self.agent), _model_key(self.judge or ""))


def load_cells(results_root: Path, benchmark: str | None) -> list[AblationCell]:
    """Every ablation run, grouped into (config, agent, judge) cells and averaged."""
    # key -> list of (Run.means, n_questions); plus the identity fields.
    groups: dict[tuple, list[tuple[dict, int]]] = defaultdict(list)
    meta: dict[tuple, tuple[int, str, str, str | None]] = {}
    for run in discover_runs(results_root):
        if run.system != ABLATION_SYSTEM:
            continue
        if benchmark and run.benchmark != benchmark:
            continue
        # `discover_runs` already resolved the run's identity from its config.yaml: the tool
        # set (with the working-set mode folded into the label) and the sem-filter judge model.
        order, label, judge = run.variant.order, run.variant.tools_label, run.judge
        key = (label, run.llm, judge)
        means = dict(run.means)
        adj = _adj_page_recall(run.run_dir)
        if adj is not None:
            means["adj_page_recall"] = adj
        groups[key].append((means, run.n_questions))
        meta[key] = (order, label, run.llm, judge)

    cells: list[AblationCell] = []
    for key, runs in groups.items():
        order, label, agent, judge = meta[key]
        means: dict[str, float] = {}
        for mkey, _, _, _ in _METRICS:
            vals = [m[mkey] for m, _ in runs if mkey in m]
            if vals:
                means[mkey] = sum(vals) / len(vals)
        cells.append(AblationCell(
            order=order, config_label=label, agent=agent, judge=judge,
            means=means, n_runs=len(runs), n_questions=sum(n for _, n in runs),
        ))
    cells.sort(key=lambda c: c.sort_key())
    return cells


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _fmt(cell: AblationCell, key: str, scale: float, fmt: str) -> str:
    v = cell.means.get(key)
    return "--" if v is None else fmt.format(v * scale)


def render_text(cells: list[AblationCell]) -> str:
    headers = [h for _, h, _, _ in _METRICS]
    lines = ["## Full breakdown (one row per config x agent x judge)"]
    head = f"{'config':32} {'agent':22} {'judge':22} " + " ".join(f"{h:>8}" for h in headers) + f" {'runs':>5} {'n_q':>4}"
    lines += [head, "-" * len(head)]
    for c in cells:
        cols = " ".join(f"{_fmt(c, k, sc, fm):>8}" for k, _, sc, fm in _METRICS)
        lines.append(f"{c.config_label:32} {_short(c.agent):22} {_short(c.judge):22} {cols} {c.n_runs:>5} {c.n_questions:>4}")

    # accuracy pivot: config (rows) x model column (cols); split-judge gets its own column label.
    # A column's identity is (agent, judge-if-different); ordered by the FULL model ids, labelled short.
    def col_of(c: AblationCell) -> str:
        if c.judge and c.judge != c.agent:
            return f"{_short(c.agent)} / {_short(c.judge)}"
        return _short(c.agent)

    def col_sort(c: AblationCell) -> tuple:
        return (_model_key(c.agent), _model_key(c.judge) if (c.judge and c.judge != c.agent) else (-1, ""))

    col_order: dict[str, tuple] = {}
    for c in cells:
        col_order.setdefault(col_of(c), col_sort(c))
    col_labels = sorted(col_order, key=lambda lbl: col_order[lbl])
    row_labels: list[str] = []
    seen = set()
    for c in sorted(cells, key=lambda c: c.order):
        if c.config_label not in seen:
            seen.add(c.config_label)
            row_labels.append(c.config_label)
    acc: dict[tuple[str, str], str] = {}
    for c in cells:
        v = c.means.get("score")
        if v is not None:
            acc[(c.config_label, col_of(c))] = f"{v * 100:.1f}"

    lines += ["", "## Accuracy (%) pivot — config x model"]
    head2 = f"{'config':32} " + " ".join(f"{cl:>24}" for cl in col_labels)
    lines += [head2, "-" * len(head2)]
    for rl in row_labels:
        cells_row = " ".join(f"{acc.get((rl, cl), '--'):>24}" for cl in col_labels)
        lines.append(f"{rl:32} {cells_row}")
    return "\n".join(lines)


def render_latex(cells: list[AblationCell], *, label: str = "tab:ablation") -> str:
    n_metrics = len(_METRICS)
    colspec = "lll" + "r" * n_metrics
    headers = [h.replace("%", r"\%").replace("$", r"\$") for _, h, _, _ in _METRICS]
    out = [
        "% --- auto-generated by qatfd/eval/ablation_tables.py ---",
        "% requires \\usepackage{booktabs}",
        r"\begin{table}[t]", r"  \centering", r"  \small",
        rf"  \begin{{tabular}}{{{colspec}}}", r"    \toprule",
        "    Config & Agent & Judge & " + " & ".join(headers) + r" \\", r"    \midrule",
    ]
    prev_order = None
    for c in cells:
        if prev_order is not None and c.order != prev_order:
            out.append(r"    \midrule")
        prev_order = c.order
        cols = " & ".join(_fmt(c, k, sc, fm) for k, _, sc, fm in _METRICS)
        row = f"{c.config_label} & {_short(c.agent)} & {_short(c.judge)} & {cols}"
        out.append("    " + row + r" \\" + f"  % n_runs={c.n_runs}, n_q={c.n_questions}")
    out += [
        r"    \bottomrule", r"  \end{tabular}",
        r"  \caption{Retrieval tool ablation on the SearchAgent (read\_document + prune always "
        r"present). Each row is one tool set at a given agent model and semantic-filter judge model "
        r"(``--'' = no semantic filter), averaged across runs. Metrics are per-question means: "
        r"accuracy, page/doc recall (\%), cost (USD/question), and latency (s/question). "
        r"\emph{adjPR\%} is page recall crediting 1.0 to any fully-correct question (a correct "
        r"answer implies a usable page was retrieved even when it is not the labelled gold page).}",
        rf"  \label{{{label}}}", r"\end{table}",
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", type=Path, default=_default_results_root())
    parser.add_argument("--benchmark", type=str, default="officeqa", help="benchmark to filter to (blank = all)")
    parser.add_argument("--latex", action="store_true", help="emit the LaTeX breakdown table instead of text")
    parser.add_argument("--out", type=str, default="-", help="output file, or '-' for stdout")
    args = parser.parse_args(argv)

    if not args.results_root.exists():
        print(f"[ablation_tables] results root not found: {args.results_root}", file=sys.stderr)
        return 1

    cells = load_cells(args.results_root, args.benchmark or None)
    if not cells:
        print(f"[ablation_tables] no {ABLATION_SYSTEM} runs found under {args.results_root}"
              + (f" for benchmark={args.benchmark}" if args.benchmark else ""), file=sys.stderr)
        return 1

    text = render_latex(cells) if args.latex else render_text(cells)
    if args.out == "-":
        print(text)
    else:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"[ablation_tables] wrote table to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
