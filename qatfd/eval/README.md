# qatfd/eval — results post-processing

Scripts that turn the results under `../results/` into paper-ready artifacts.
These modules read `report.csv` / `config.yaml` / `traces/events.jsonl` directly
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

These come from the structured `traces/events.jsonl` stream (the same one the
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
