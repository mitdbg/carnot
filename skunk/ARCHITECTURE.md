# Architecture

## Why this design

OfficeQA questions ask about specific tables and figures inside a corpus of 696 monthly U.S. Treasury Bulletin PDFs (1939–2025). A naive date-driven retrieval agent — "the question mentions 1940, look in the 1940 bulletin" — fails on **81%** of the benchmark, because the answer-bearing table for period *P* often lives in a *later* bulletin (publication lag) or in a *retrospective summary* in a single mid-year issue. The median miss is +2 months; the long tail goes out to +9 years.

The fix: retrieve at the **page** level, indexed by **what each page reports**, not by when its bulletin was published. A page-level catalog tells the planner where to look; the actual reading is a separate concern handled by the extract operator.

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
  ┌─────────┐        ┌─────────┐        ┌─────────────┐
  │retrieve │        │ extract │        │lookup_extern│   per-op operators
  │ catalog │        │ tier 1-3│        │   gemini    │
  └─────────┘        └─────────┘        └─────────────┘
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
   │  eval_e2e           full pipeline end-to-end          │
   └───────────────────────────────────────────────────────┘
```

## Page number convention

The codebase uses a single canonical page-number meaning everywhere: **`PageRef.page` is the 1-based PDF page index**. PyMuPDF, cache filenames, the JSON-backed Tier 1 index, and the Fraser benchmark URLs (`source_docs?page=N`) all agree on this convention.

The benchmark's `source_docs?page=N` parameter is the PDF page index in Fraser's viewer (verified against the June 2025 issue: `?page=76` lands on the ESF-1 table on PDF page 76, whose printed footer reads "69"). This maps directly to `PageRef.page`.

## The page index lives in retrieve

The retrieve operator owns its page index — the format, schema, and build process are implementation details of that operator. The index must be able to answer: *given a concept and a period, which pages in the corpus report on that?*

The load-bearing insight is that `periods_covered` (what period a page *reports on*) is distinct from the bulletin's publication date. A page in the January 1941 bulletin that contains the CY1940 annual summary should be returned for a query on `period='CY1940'` — not the January 1941 bulletins. Any index the retrieve operator builds must capture this distinction.

**Current state (placeholder).** `RetrieveExecutor` in `src/skunk/retrieve.py` is intentionally unwired pending integration of an external retrieve implementation. It honors `ctx.config.golden_pages` (so `eval/eval_e2e.py --golden` works) and otherwise raises `NotImplementedError`. The previous page-index pipeline (L1 chapter pick → year-window filter → optional BM25 rerank) is preserved as a prototype at `src/skunk/page_index/retrieve_prototype.py` (`PageIndexRetrievePrototype`) for regression comparison.

**Optional BM25 rerank** (experimental scaffold, default OFF; lives on the prototype path, not the placeholder). When `SkunkConfig.bm25_enabled` is set, an in-memory per-chapter BM25 index over `title + column_headers + row_headers_sample + keywords` reranks the year-filtered candidate set. If the top BM25 score clearly dominates the field (`top1 / median20 ≥ bm25_dominance_threshold`, default 2.0) the candidate list is truncated to `bm25_top_k` (default 20); otherwise all year-filtered survivors are returned in BM25 order. BM25 modules live in `src/skunk/page_index/{bm25,bm25_runtime}.py`. Earlier embedding-based reranking experiments were subtractive at comparable compression ratios (see `data/page_index/README.md`); BM25's IDF scoring is meant to sidestep the semantic-clustering failure mode that hurt embeddings on this corpus.

## The 4 operators

- **`retrieve(key, period)`** — only chain head. Returns `list[PageRef]` with `(month, page)` populated (where `page` is the 1-based PDF page index). Currently a placeholder honoring `ctx.config.golden_pages` only; pending external-retriever integration (see "Current state" above).
- **`extract(key, period, visual_only?)`** — reads `ctx.question` and the located pages via tier dispatch (parsed JSON → vision). Returns `list[AnnotatedValue]` — each entry has a `description`, `value`, `unit`, and one of three **kinds**: `scalar`, `vector` (1-D series with one varying dim), or `table` (2-D grid with row/col dims). Vector/table cells are always primitive scalars; nesting beyond those shapes is rejected by the extract parser before an `AnnotatedValue` is constructed. `extract` is invoked automatically by the orchestrator on every `RetrieveBranch` — `visual_only` is set on the branch and threaded through. Pass `visual_only=True` to skip Tier 1 and go straight to vision (use for charts/figures).
- **`lookup_external(nl)`** — chain-head capable. Single Gemini call: takes a natural-language description of external factual data (`nl`) and returns `list[AnnotatedValue]` (one entry). Use for CPI-U, FX rates, event dates, named entities (bureau names), and any fact not in the bulletin corpus. The operator infers the appropriate `kind` and `unit` (including `text` for strings).
- **`compute()`** — chain terminator that subsumes formatting. Reads `ctx.question` plus the upstream extracted/looked-up values; runs a single unified loop (`ComputeExecutor.run`) of codegen → in-process exec → self-critique under one shared budget (`compute_max_attempts`, default 3). Each iteration's codegen call may return Python (success), raise `MissingData` (the structured insufficient-data signal), or raise `_ParseFailure` (unparseable reply); exec failures and critique REVISE verdicts feed the next iteration's `prev_code` + `prev_failure`. The critique sees the same domain context as codegen (`question`, `prev`, `computation`/`presentation` JSON, the code, the result). Returns the final answer as a plain `str` on ACCEPT, or — if the budget exhausts after at least one successful exec — the most recent uncritiqued result as a fallback. Fails with `StepFailed("compute", …)` when no iteration ever execs cleanly; propagates `MissingData` (with the same fallback rule) — the orchestrator catches that and runs up to `recovery_max_rounds` re-planning rounds before giving up.

## Plan shape

A **Plan** is a pydantic model: a `branches` list plus a nested `computation` submodel (free-form `task` + optional `qualifiers`) and a nested `presentation` submodel (`units_out`, `precision`, `answer_form`). The Python layout mirrors the wire JSON exactly, so `Plan.model_validate_json` / `Plan.model_dump_json` round-trip with no custom translation. There is no multi-step decomposition in the AST — every plan is exactly one terminal compute. `PlannerPromptedCall` (in `src/skunk/plan.py`) emits a `Plan`; the orchestrator runs every branch in parallel, then feeds the merged `list[AnnotatedValue]` to compute.

Two branch shapes:

- `RetrieveBranch(key, period, visual_only=False)` — `retrieve` then `extract` are dispatched together by the orchestrator (extract reads `visual_only` from the branch).
- `LookupBranch(target, src=None)` — single `lookup_external` call.

Canonical JSON shape (one branch + computation + presentation):

```json
{
  "branches": [
    {"kind": "retrieve", "key": "national defense expenditures", "period": "CY1940"}
  ],
  "computation": {"task": "Report total CY1940 national defense expenditures."},
  "presentation": {"units_out": "in millions of dollars", "precision": 1}
}
```

Per-field semantics (units_out phrasing, when to use `null`, etc.) live in `PlannerPromptedCall.system_prompt` in `src/skunk/plan.py` — that prompt is the canonical schema spec. Runtime contracts (`PageRef`, `AnnotatedValue`, the scalar / vector / table kind taxonomy with primitive-only cells) live in the dataclasses in the same file.

The `AnnotatedValue.kind` taxonomy:

- `scalar` — `value` is `int | float | str`. Single facts.
- `vector` — `value` is `{index_label: scalar}`; `index_name` names the varying dim.
- `table` — `value` is `{row_label: {col_label: scalar}}`; `row_name` / `col_name` name the two dims.

Cells must be primitive scalars (`int` / `float` / `str`, no `bool`, no nesting beyond these shapes). The extract parser rejects deeper structures before constructing an `AnnotatedValue`.

## Per-page tier dispatch in extract

Once retrieve has named specific pages, extract chooses how to read each one:

```
Tier 1  parsed-JSON elements bucketed by page_id   — structured text + HTML tables
                                                     (treasury_bulletin_{YYYY}_{MM}.json
                                                     under $OFFICEQA_PARSED_JSON_DIR;
                                                     see skunk/page_index/pdf.py)
Tier 2  PNG render at 300 dpi + vision LLM          — live, in-memory bytes (_render_pdf_page_b64)
```
(`page` here = `PageRef.page` = 1-based PDF page index)

Tier escalation only happens when the chosen tier reports "value not present". There is no cross-page search — if retrieve picked the wrong pages, the bug is in retrieve, not extract. This is what makes the eval decomposition work.

## Eval decomposition

The benchmark CSV (`data/officeqa_pro.csv`) has both retrieval-level and answer-level golden truth:

- `source_docs` — URLs containing `?page=N` for every question (verified 100% coverage on the 133-row pro split).
- `answer` — fuzzy-matchable expected output.

One harness drives the pipeline end-to-end:

| harness | input | output metric |
|---|---|---|
| `eval/eval_e2e.py` | question | answer accuracy — full pipeline end-to-end |

Each operator exposes a standalone `run(prev, ctx, **kwargs)` callable on its operator-level class so the harness can invoke it without the orchestrator (e.g. `eval_e2e --golden` injects golden pages and skips retrieve).

## Caches

The only on-disk caches in the runtime path are page-index artifacts:

- `cache/page_index_v3/` — full build output (catalog rows, L1 spans, chapter tree); the shipped tree at `data/page_index/concept_tree.json` is promoted from here. See `src/skunk/page_index/pipeline.py`.
- Parsed-JSON corpus at `$OFFICEQA_PARSED_JSON_DIR` (default `~/Desktop/officeqa/treasury_bulletins_parsed/jsons/`) — read by Tier 1 of extract and by the page-index builder.

LLM completions are **not** cached. Tier 2 PNG renders are computed live per call (no disk cache). There is no DSL plan cache — the planner runs once per question.

## TODO after merge (search agent integration)

The teammate's `SearchAgent` was merged in (`refs/heads/skunk`) under
`src/skunk/search_agent/` as a self-contained subtree, wired through
`RetrieveExecutor` when `config.retriever == "search_agent"` (default).
The first-pass merge was intentionally conservative — the agent keeps
its own `OpenRouter` shim, its own `LocalPythonExecutor`, and its own
`Tracer` instead of the framework's `LLMClient` / `pyexec` / `trace`.
Open follow-ups:

- Unify `src/skunk/pyexec.py` and `src/skunk/search_agent/utils/local_python_executor.py`. The agent loop needs `final_answer` semantics + sandboxing; compute / lookup_external currently use the simpler in-process exec.
- Port `SearchAgent` to `skunk.common.LLMClient` so it picks up the `effort` knob, the RPM limiter, and retry. The shim at `src/skunk/search_agent/openrouter_client.py` (a local `openrouter` stand-in built on the `openai` SDK) goes away when this lands.
- Subclass `PromptedCall` for the agent system prompt so corpus / few-shots / lessons overrides in `prompt_overrides.yaml` flow through (today `prompts.yaml` is loaded directly).
- Replace `src/skunk/search_agent/tracer.py` with `ctx.emit(...)` events; the orchestrator's `StepTrace` already records timing and I/O.
- Thread `branch.key` / `branch.period` into the agent — today they're logged but discarded; only `ctx.question` reaches `SearchAgent.retrieve()`. Lossy for multi-retrieve plans with distinct keys.
- Add an embedding entry-point to `LLMClient` and route `vector_search` through it (today it calls `OpenRouter.embeddings.generate` directly).
- Reconcile `src/skunk/search_agent/prep/officeqa_eval.py` helpers with anything we already have under `eval/util.py`.
- Decide what to do with the `src/carnot/agents/utils.py` style tweak that came with the merge (`parse_code_blobs` dedent → f-string) — it's a main-Carnot file outside the skunk subdir.

## What is intentionally NOT in this design

- **Agentic loops confined to retrieve.** When `config.retriever == "search_agent"`, the retrieve operator runs an iterative tool-using LLM loop (`SearchAgent` in `src/skunk/search_agent/`). The other three operators (`extract`, `lookup_external`, `compute`) execute once per call; failure is recorded in the trace. Missing-data recovery is the bounded replan loop in the orchestrator, not per-operator agent loops.
- **No per-table/per-figure catalog rows.** Page-level granularity matches the benchmark's `source_docs?page=N` labels and the existing `cache/tables/` structure. Going finer adds rows without improving recall.
- **No PZ runtime dependency.** This repo is plain Python + Gemini API (google-generativeai); PZ stays out of the runtime path.
