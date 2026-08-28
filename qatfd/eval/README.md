# qatfd/eval — results post-processing

Scripts that turn the results under `../results/` into paper-ready artifacts.
These modules read `report.csv` / `config.yaml` / `traces/<qid>.jsonl` directly
and do **not** import the `qatfd` runtime package, so they run without the eval
dependencies installed (only `pyyaml` is needed).

## `variants.py` — how a run is identified

The SearchAgent variants used to be separate qatfd systems (`search_agent`,
`qatfd_search_agent`, `ablation_search_agent`), so a run's directory —
`results/<benchmark>/<system>/<run>/` — doubled as its identity. They are now **one
`search_agent` system configured by its retrieval-tool flags**, so every variant lands
in `results/<benchmark>/search_agent/` and the identity is recovered from the run's
persisted `config.yaml` instead. `variants.py` is the single place that does that, and
all three table scripts share it:

| Tool set (`include_search_corpus`, `include_grep_corpus`, `include_semantic_filter`) | Label |
|---|---|
| `F, T, F` | Grep-only |
| `T, F, F` | Vector-only |
| `F, F, T` | Sem-only |
| `T, T, F` | **SearchAgent** |
| `F, T, T` | **QATFD** |
| `T, T, T` | All tools |

`read_document` and `prune` are always present, so they are not part of the identity. A
non-default working-set mode (`working_set_off` / `id_tracking_only`) is appended to the
label — `QATFD (WS: off)` — and is part of the key, so a working-set ablation never
averages into the row of an otherwise-identical run. The agent model is read from
`inference.llm_model` (it moved out of the `systems` group when the inference config group
was split out), and the semantic-filter judge model (`systems.semantic_filter_model`) is
also part of the key so two sem-filter runs that differ only in their judge stay separate
rows.

`experiments.run_name` still controls the run **dir** name and is worth setting so the
results tree stays legible — but nothing in eval/ parses it.

## `latex_tables.py` — main-result tables

Generates **one LaTeX table per benchmark** (OfficeQA, BrowseComp-Plus,
TREC-BioGen, FinanceBench, QAMPARI, FreshStack).

- **Rows** = variant × LLM: `RAG-LLM (k=10/100/1000)` plus one row per SearchAgent
  tool set (see `variants.py`), one row per LLM it was run with. The `Model` column
  shows `agent / judge` when the semantic filter ran on its own model.
- **Columns** = `Accuracy`, the benchmark's retrieval recall column(s), avg.
  per-question `Cost`, avg. per-question `Latency`.
- **Best per column is bolded** — highest for accuracy/recall, lowest for
  cost/latency.
- When a (benchmark, variant, LLM, judge, top_k) config has **multiple runs**, the
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

Generates a single table with **one row-group per benchmark**, one row per retrieval
configuration present in the results (RAG-LLM plus each SearchAgent tool set — see
`variants.py`), reporting where time goes inside that configuration's *retrieval* stage:

- avg. **# search steps** (agent steps; 1.0 for RAG-LLM),
- avg. **# vector-search / grep / read-document / semantic-filter** tool calls per question,
- avg. **execution latency** of each of those tool types (seconds).

Two things to know about the numbers:

- **Counts include every attempted call**, errored or not. This is deliberate: a
  nonzero count for a tool the configuration doesn't include is a useful signal of a
  prompting bug — e.g. a sem-only agent writes `search_corpus(...)` (a tool it lacks;
  the interpreter rejects it with "Forbidden function evaluation") on QAMPARI, because
  the system prompt still references `search_corpus` in its grep/prune tool docs even
  when the tool is off.
  Errored calls return in ~0s, so they pull that tool's latency average toward zero.
- **Latency is wall-clock, not isolated CPU time.** It's the agent's own
  `tool_latency_s`, measured around the sandboxed tool execution under the runner's
  concurrent execution (`workers=N`), and `grep`/vector search run as ChromaDB calls
  on a shared collection. So a tool's latency includes contention from sibling
  questions (and, for SearchAgent, from its own concurrent vector queries) — useful
  as real serving latency, but not a clean per-call CPU cost. For a contention-free
  number you'd need to instrument the tools with `perf_counter` and re-run (ask if
  you want this).

These come from the per-question `traces/<qid>.jsonl` streams of structured
`skunk.trace.TraceEvent` lines: split the retrieval stage from the answer stage at
the 2nd `system` event (the compute agent's system prompt), take each executed step
from its closing `agent_step` event, identify the step's tool from its
`kind="tool_call"` result event (`data.tool`) — falling back to the first
non-comment line of `data.code` for errored calls — and take tool latency from the
`agent_step` event's `tool_latency_s` (measured by the agent around the sandboxed
execution; LLM reasoning excluded). A tool absent from a configuration scores 0.

```bash
python eval/tool_metrics.py                       # table -> stdout
python eval/tool_metrics.py --out tools.tex
python eval/tool_metrics.py --llm gemini-3.5-flash    # pin one model (recommended)
python eval/tool_metrics.py --benchmarks officeqa,qampari
```

By default each row averages across all runs of that variant, and the row label is
suffixed with the model it used (all LLMs, all RAG-LLM `k`). Pass `--llm` to pin a
single model for an apples-to-apples table — the labels then drop the model suffix.
Only `booktabs` is required (benchmark labels sit on the first row of each group,
so no `multirow`).

### Adding a system / benchmark / LLM

- New benchmark: add an entry to `BENCHMARKS` (dir-name → display name). Recall
  columns are auto-detected from the report's header (everything between
  `scorer` and `retrieved_docs`); add a pretty header in `RECALL_DISPLAY` if you
  want a nicer name.
- New system: add it to `SYSTEM_LABELS` in `variants.py` (a new *tool set* of the
  existing SearchAgent needs nothing — unrecognised combinations get an auto-generated
  label; add an entry to `TOOLSETS` to give it a name and a sort position).
- New LLM: nothing to do — rows are keyed on `inference.llm_model` from each run's
  `config.yaml`.

## `ablation_tables.py` — tool-ablation tables

Reads every `search_agent` run (the retrieval tool set is chosen by config — any subset
of vector / grep / semantic-filter, with `read_document` + `prune` always on, plus an
optional cheaper `semantic_filter_model` for the judge calls) and reports each
configuration's **usefulness** metrics so the tools can be compared head-to-head.

Each configuration is identified by (tool set, agent model, judge model) via
`variants.py`; runs sharing all three are averaged. Two
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
