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
                    ┌─────────────┐
                    │   compute   │
                    │ format/text │
                    └──────┬──────┘
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

Two distinct "page numbers" exist for every bulletin page. The codebase always uses bulletin page as the canonical reference:

| field | meaning | where |
|---|---|---|
| `page` | bulletin **printed** page number — what appears on the physical page | `PageRef.page`, catalog, eval golden |
| `pdf_page` | 1-based PDF page **index** — what PyMuPDF and cache filenames use | `PageRef.pdf_page`, cache filenames |

`cache/page_maps/{YYYY-MM}.json` provides the bidirectional translation per bulletin:
```json
{
  "bulletin_to_pdf": {"1": 3, "2": 4, ...},
  "pdf_to_bulletin": {"1": null, "2": null, "3": 1, "4": 2, ...}
}
```
Built by `prep/page_map.py:build_page_map_from_ocr`. The helper `pdf_page_for_ref(ref, cache_dir)` resolves a `PageRef` to the correct PDF page index for cache/render access.

The benchmark's `source_docs?page=N` parameter is a bulletin page number, so `eval/golden.py:GoldenPage.page` maps directly to `PageRef.page`.

## The page index lives in retrieve

The retrieve subagent owns its page index — the format, schema, and build process are implementation details of that subagent. The index must be able to answer: *given a concept and a period, which pages in the corpus report on that?*

The load-bearing insight is that `periods_covered` (what period a page *reports on*) is distinct from the bulletin's publication date. A page in the January 1941 bulletin that contains the CY1940 annual summary should be returned for a query on `period='CY1940'` — not the January 1941 bulletins. Any index the retrieve subagent builds must capture this distinction.

## The 6 operators

- **`retrieve(concept, period, source_bulletin?)`** — only chain head. Looks up relevant pages via the retrieve subagent's internal page index, returns a `DocHandle` whose `PageRef`s have `(file_path, year, month, page, pdf_page)` fully specified (`page` = bulletin printed page number, `pdf_page` = PDF index).
- **`extract(concept, mode?)`** — reads value(s) from the located pages via tier dispatch. `mode='value'|'list'|'table'` controls scalar / list / DataFrame output.
- **`read_visual(concept)`** — same input as extract but always uses Tier 3 vision; reserved for charts and figures.
- **`lookup_external(resource, **params)`** — chain-head capable. Fetches CPI-U, FX rates, GDP from cached CSV (or live API). The `event_year` / `event_date` resources resolve knowledge-bound dates ("Korean War start", "Black Monday") — these feed retrieve when a question's period is implicit.
- **`compute(code)`** — the only transformation op. LLM writes a Python body that reads `prev` and sets `result`, runs in a sandbox with numpy/pandas/statsmodels. Covers everything from `result = sum(prev.value)` to OLS, Hodrick-Prescott, Box-Cox.
- **`format(precision?, unit?, layout?)`** — chain terminator. Deterministic application of presentation rules; no LLM call.

## Per-page tier dispatch in extract

Once retrieve has named specific pages, extract chooses how to read each one:

```
Tier 1  cache/tables/{YYYY-MM}/p{NNN}-*.csv       — pre-extracted DataFrames, cheap
Tier 2  cache/pages/{YYYY-MM}/p{NNN}.txt          — per-page OCR text, free, may be noisy
Tier 3  cache/pages/{YYYY-MM}/p{NNN}.png          — PNG render + vision LLM, always works
```
(NNN = PDF page index, resolved from PageRef.page via `prep/page_map.pdf_page_for_ref`)

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
