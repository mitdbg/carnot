# Architecture

## Why this design

OfficeQA questions ask about specific tables and figures inside a corpus of 696 monthly U.S. Treasury Bulletin PDFs (1939–2025). A naive date-driven retrieval agent — "the question mentions 1940, look in the 1940 bulletin" — fails on **81%** of the benchmark, because the answer-bearing table for period *P* often lives in a *later* bulletin (publication lag) or in a *retrospective summary* in a single mid-year issue. The median miss is +2 months; the long tail goes out to +9 years.

The fix: retrieve at the **page** level, indexed by **what each page reports**, not by when its bulletin was published. A page-level catalog tells the planner where to look; the actual reading is a separate concern handled by the extract subagent.

This decoupling means retrieval and extraction can be evaluated independently — both labels exist in the benchmark (`source_docs?page=N` for retrieval, `answer` for extraction) — so we can attribute errors cleanly.

## High-level flow

```
                    ┌────────────┐
   question ───────▶│  planner   │  one LLM call
                    │   (DSL)    │
                    └─────┬──────┘
                          │  ChainNode AST
                          ▼
                    ┌────────────┐
                    │orchestrator│  walks AST; parallel branches; speculative pre-warm
                    └─────┬──────┘
                          │
       ┌──────────────────┼──────────────────┐
       │                  │                  │
       ▼                  ▼                  ▼
  ┌─────────┐        ┌─────────┐        ┌─────────┐
  │retrieve │        │ extract │        │read_visu│   per-op subagents
  │ catalog │        │ tier 1-3│        │  vision │
  └─────────┘        └─────────┘        └─────────┘
       │                  │                  │
       ▼                  ▼                  ▼
                    ┌──────────────────┐
                    │     compute      │   chain terminator:
                    │ plan → code →    │   self-plans, codegens,
                    │ exec → verify    │   execs, verifies output
                    └────────┬─────────┘   format/unit
                             │
                             ▼
                          answer

   ┌────────────────── eval harnesses ─────────────────────┐
   │  eval_retrieval     retrieve only vs golden pages     │
   │  eval_extraction    extract only on golden pages      │
   │  eval_e2e           full pipeline; gap = retrieval cost│
   └───────────────────────────────────────────────────────┘
```

## Page number convention

The codebase uses a single canonical page-number meaning everywhere: **`PageRef.page` is the 1-based PDF page index**. PyMuPDF, cache filenames, the JSON-backed Tier 1 index, and the Fraser benchmark URLs (`source_docs?page=N`) all agree on this convention.

The bulletin's *printed* page-number footer (e.g. "69" stamped at the bottom of PDF page 76) is recoverable when needed via `skunk.common.parsed_json.get_printed_page(ref, ctx)`, which reads the `page_number`-typed element the JSON parser preserves per page. We surface it for trace enrichment and Gemini prompt headers (`--- PDF page 76 (bulletin printed page "69") ---`), but never use it as a lookup key.

The benchmark's `source_docs?page=N` parameter is the PDF page index in Fraser's viewer (verified against the June 2025 issue: `?page=76` lands on the ESF-1 table on PDF page 76, whose printed footer reads "69"). `eval/golden.py:GoldenPage.page` therefore maps directly to `PageRef.page`.

## The page index lives in retrieve

The retrieve subagent owns its page index — the format, schema, and build process are implementation details of that subagent. The index must be able to answer: *given a concept and a period, which pages in the corpus report on that?*

The load-bearing insight is that `periods_covered` (what period a page *reports on*) is distinct from the bulletin's publication date. A page in the January 1941 bulletin that contains the CY1940 annual summary should be returned for a query on `period='CY1940'` — not the January 1941 bulletins. Any index the retrieve subagent builds must capture this distinction.

## The 5 operators

- **`retrieve(concept, period, source_bulletin?)`** — only chain head. Looks up relevant pages via the retrieve subagent's internal page index, returns a `DocHandle` whose `PageRef`s have `(file_path, year, month, page)` fully specified (where `page` is the 1-based PDF page index).
- **`extract()`** — reads `ctx.question` and the located pages via tier dispatch (parsed JSON → PyMuPDF text → vision). Emits a `TypedValue` with `dtype='named'` whose `.value` is a dict mapping snake_case names to scalars/lists/tables relevant to the question. The agent decides what's worth extracting; no `concept`/`mode` args.
- **`read_visual()`** — same output shape as extract, but always uses vision; reserved for charts and figures.
- **`lookup_external(nl)`** — chain-head capable. Single Gemini call: takes a natural-language description of external factual data (`nl`) and returns a `TypedValue`. Use for CPI-U, FX rates, event dates, named entities (bureau names), and any fact not in the bulletin corpus. The subagent infers the appropriate `dtype` and `unit` (including `text` for strings).
- **`compute()`** — chain terminator that subsumes formatting. Reads `ctx.question` plus the upstream extracted/looked-up values; runs a plan-then-codegen LLM call (`CODE\n<python>` or `MISSING:<reason>`), execs the code, then a verifier LLM checks the output's *form* (precision, unit, percent vs decimal, comma rules, list bracketing) against the question. Up to 3 attempts; each retry receives feedback from the prior failure. Returns a `FormattedString`. Fails with `StepFailed("compute", "missing data: …")` when extracted values are insufficient.

## Per-page tier dispatch in extract

Once retrieve has named specific pages, extract chooses how to read each one:

```
Tier 1  parsed-JSON elements bucketed by page_id   — structured text + HTML tables (common/parsed_json.py)
Tier 2  cache/pages/{YYYY-MM}/p{NNN}.txt           — per-page PyMuPDF text, on-demand cache
Tier 3  cache/pages/{YYYY-MM}/p{NNN}.png           — PNG render + vision LLM, always works
```
(NNN = `PageRef.page` = 1-based PDF page index)

The choice is per-page and deterministic. Tier escalation only happens when the chosen tier reports "value not present". There is no cross-page search — if retrieve picked the wrong pages, the bug is in retrieve, not extract. This is what makes the eval decomposition work.

## Eval decomposition

The benchmark CSV (`data/officeqa_pro.csv`) has both retrieval-level and answer-level golden truth:

- `source_docs` — URLs containing `?page=N` for every question (verified 100% coverage on the 133-row pro split).
- `answer` — fuzzy-matchable expected output.

Three independent harnesses isolate the failure modes:

| harness | input | output metric |
|---|---|---|
| `eval/eval_retrieval.py` | question | bulletin-precision, page-precision-strict, page-precision-loose(±2) vs golden pages |
| `eval/eval_extraction.py` | question + golden pages | answer accuracy (fuzzy match) — measures extraction quality assuming perfect retrieval |
| `eval/eval_e2e.py` | question | answer accuracy — gap to extraction-only quantifies retrieval cost |

Each subagent therefore exposes a standalone callable function (not just `Subagent.run`), so the eval harnesses can invoke it without the orchestrator.

## Caches and reproducibility

Per-step cache key: `(question_uid, op_index, args_hash)`. Cached values:
- retrieve results (page rankings)
- per-page CSV table loads (already cached by `prep/tables.py`)
- LLM completions for extract / read_visual / compute (the Python body and result)

`HarnessContext.cache_only=True` blocks live API calls (BLS, FRED, FX, vision LLM) — useful for deterministic eval re-runs against a frozen snapshot.

## What is intentionally NOT in this design

- **No agentic search loops.** Each subagent executes once per op; failure is recorded in the trace, not retried via re-planning.
- **No per-table/per-figure catalog rows.** Page-level granularity matches the benchmark's `source_docs?page=N` labels and the existing `cache/tables/` structure. Going finer adds rows without improving recall.
- **No PZ runtime dependency.** This repo is plain Python + Anthropic SDK; PZ stays out of the runtime path.
