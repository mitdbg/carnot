# qatfd/eval — results post-processing

Scripts that turn the results under `../results/` into paper-ready artifacts.
These modules read `report.csv` / `config.yaml` / `traces/<qid>.jsonl` directly
and do **not** import the `qatfd` runtime package, so they run without the eval
dependencies installed (only `pyyaml` is needed).

## `latex_tables.py` — main-result tables

Generates **one LaTeX table per benchmark** (OfficeQA, BrowseComp-Plus,
TREC-BioGen, FinanceBench, QAMPARI, FreshStack).

- **Rows** = system × LLM: `RAG-LLM (k=10/100/1000)`, `SearchAgent`, `QATFD`,
  one row per LLM the system was run with.
- **Columns** = `Accuracy`, the benchmark's retrieval recall column(s), avg.
  per-question `Cost`, avg. per-question `Latency`.
- **Best per column is bolded** — highest for accuracy/recall, lowest for
  cost/latency.
- When a (benchmark, system, LLM, top_k) config has **multiple runs**, the
  metrics are **averaged across runs** (equal weight per run; run count is
  emitted as a `% n_runs=...` comment on each row).
- Benchmarks with no results yet (e.g. FreshStack) render a placeholder table.

### Usage

```bash
# all tables -> stdout
python eval/latex_tables.py

# write to a file
python eval/latex_tables.py --out tables.tex

# a subset of benchmarks, no recall columns
python eval/latex_tables.py --benchmarks officeqa,qampari --no-recall

# only runs that used a particular LLM
python eval/latex_tables.py --llm gemini-3.5-flash

# drop failed questions before averaging; skip empty-benchmark placeholders
python eval/latex_tables.py --exclude-failed --no-empty
```

The tables use `booktabs` (`\toprule`/`\midrule`/`\bottomrule`) — add
`\usepackage{booktabs}` to your preamble.

## `tool_metrics.py` — retrieval-stage tool breakdown

Generates a single table with **one row-group per benchmark**, the three systems
(RAG-LLM, SearchAgent, QATFD) repeated within each group, reporting where time
goes inside each system's *retrieval* stage:

- avg. **# search steps** (agent steps; 1.0 for RAG-LLM),
- avg. **# vector-search / grep / read-document / semantic-filter** tool calls per question,
- avg. **execution latency** of each of those tool types (seconds).

Two things to know about the numbers:

- **Counts include every attempted call**, errored or not. This is deliberate: a
  nonzero count for a tool the agent doesn't actually have is a useful signal of a
  prompting bug — e.g. QATFD's agent writes `search_corpus(...)` (a tool it lacks;
  the interpreter rejects it with "Forbidden function evaluation") on QAMPARI, because
  the QATFD system prompt still references `search_corpus` in its grep/prune tool docs.
  Errored calls return in ~0s, so they pull that tool's latency average toward zero.
- **Latency is wall-clock, not isolated CPU time.** It's `observation.t − tool_code.t`
  measured under the runner's concurrent execution (`workers=N`), and `grep`/vector
  search run as ChromaDB calls on a shared collection. So a tool's latency includes
  contention from sibling questions (and, for SearchAgent, from its own concurrent
  vector queries) — useful as real serving latency, but not a clean per-call CPU cost.
  For a contention-free number you'd need to instrument the tools with `perf_counter`
  and re-run (ask if you want this).

These come from the per-question `traces/<qid>.jsonl` streams (the same ones the
trace viewer renders), reconstructed the way the viewer does: split the retrieval
stage from the answer stage at the 2nd `system` event, group the retrieval stage
into agent turns, identify each step's tool from the first non-comment line of its
emitted code, and take tool latency as `observation.t - tool_code.t` (pure tool
time, LLM reasoning excluded). Semantic filter is QATFD-only (0 elsewhere).

```bash
python eval/tool_metrics.py                       # table -> stdout
python eval/tool_metrics.py --out tools.tex
python eval/tool_metrics.py --llm gemini-3.5-flash    # pin one model (recommended)
python eval/tool_metrics.py --benchmarks officeqa,qampari
```

By default a system's row averages across **all** its runs (all LLMs, all
RAG-LLM `k`). Pass `--llm` to pin a single model for an apples-to-apples table.
Only `booktabs` is required (benchmark labels sit on the first row of each group,
so no `multirow`).

### Adding a system / benchmark / LLM

- New benchmark: add an entry to `BENCHMARKS` (dir-name → display name). Recall
  columns are auto-detected from the report's header (everything between
  `scorer` and `retrieved_docs`); add a pretty header in `RECALL_DISPLAY` if you
  want a nicer name.
- New system: add to `SYSTEM_DISPLAY` and `SYSTEM_ORDER`.
- New LLM: nothing to do — rows are keyed on `systems.llm_model` from each run's
  `config.yaml`.

## `ablation_tables.py` — tool-ablation tables

Reads every `ablation_search_agent` run (the SearchAgent variant whose retrieval
tool set is chosen by config — any subset of vector / grep / semantic-filter, with
`read_document` + `prune` always on, plus an optional cheaper
`semantic_filter_model` for the judge calls) and reports each configuration's
**usefulness** metrics so the tools can be compared head-to-head.

Each configuration is identified by (tool set, agent model, judge model), read from
the run's `config.yaml` (`tool_vector` / `tool_grep` / `tool_semantic_filter`,
`llm_model`, `semantic_filter_model`); runs sharing all three are averaged. Two
tables are printed: a **full breakdown** (one row per config × agent × judge with
accuracy, page/doc recall, cost, latency) and an **accuracy pivot** (tool set ×
model column, where a split judge gets its own `agent / judge` column).

The breakdown also reports **`adjPR%` (adjusted page recall)**: per-question page
recall, but crediting 1.0 to any *fully-correct* question (`score == 1`) even when
its measured page recall is lower. Because a correct answer implies a usable page
was retrieved — and the labelled gold page is not the only page that can answer a
question — plain page recall undercounts retrieval; `adjPR` is a cheap proxy for the
hand audit of "did a usable page get retrieved" (so `adjPR ≥ max(acc, pageR)` always).

```bash
python eval/ablation_tables.py                        # text tables -> stdout
python eval/ablation_tables.py --benchmark officeqa   # default is officeqa
python eval/ablation_tables.py --latex --out ablation.tex
```
