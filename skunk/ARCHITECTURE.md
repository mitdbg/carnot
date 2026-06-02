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

**Two retrieval methods.** `RetrieveOp` in `src/skunk/retrieve.py` picks one of three backends per call (each a private `_run_*` method): it honors `ctx.config.golden_pages` first (so `eval/eval_e2e.py --golden` bypasses retrieval entirely via the golden branch), then routes on `config.retriever` to one of two alternatives:

- **`search_agent`** (default) — the iterative ChromaDB + LLM-loop retriever vendored under `src/skunk/search_agent/`.
- **`page_index`** — the page-index retriever at `src/skunk/page_index/query.py` (`PageIndexRetriever`), querying the offline-built catalog + concept tree. Its query path is three passes:
  1. **ToC chapter pick** (`query_toc.py`) — one LLM call selects up to two canonical chapters from the concept tree; every page under them becomes a candidate.
  2. **Year filter** — drops candidates whose page-side `dates` (verbatim strings parsed back to ISO intervals) don't intersect the query period; date-less pages are kept (recall net). Honors the `periods_covered` ≠ publication-date distinction above. No-op on an unparseable period.
  3. **Semantic filter** (`query_semfilter.py`) — a two-stage cascade that prunes the survivors to a tight candidate set: a cheap **coarse** pass over page metadata (titles/headers/dates/keywords), then a precise **fine** pass over full page text that must quote a concrete value cell to keep a page. Both stages are batched and run in parallel across batches; gated on `config.semfilter_enabled` (off → ToC + year-filter only, for ablation).

## The 4 operators

- **`retrieve(key, period)`** — only chain head. Returns `list[PageRef]` with `(month, page)` populated (where `page` is the 1-based PDF page index). Dispatched by `RetrieveOp` to one of two backends (or short-circuited by `ctx.config.golden_pages`) — see "Two retrieval methods" above.
- **`extract(key, period, visual_only?)`** — reads `ctx.question` and the located pages via tier dispatch (parsed JSON → vision). Returns `list[AnnotatedValue]` — each entry has a `description`, `value`, `unit`, and one of three **kinds**: `scalar`, `vector` (1-D series with one varying dim), or `table` (2-D grid with row/col dims). Vector/table cells are always primitive scalars; nesting beyond those shapes is rejected by the extract parser before an `AnnotatedValue` is constructed. `extract` is invoked automatically by the orchestrator on every `RetrieveBranch` — `visual_only` is set on the branch and threaded through. Pass `visual_only=True` to skip Tier 1 and go straight to vision (use for charts/figures).
- **`lookup_external(nl)`** — chain-head capable. Single Gemini call: takes a natural-language description of external factual data (`nl`) and returns `list[AnnotatedValue]` (one entry). Use for CPI-U, FX rates, event dates, named entities (bureau names), and any fact not in the bulletin corpus. The operator infers the appropriate `kind` and `unit` (including `text` for strings).
- **`compute()`** — chain terminator that subsumes formatting. Reads `ctx.question` plus the upstream extracted/looked-up values; runs a single unified loop (`ComputeOp.run`) of codegen → in-process exec → self-critique under one shared budget (`compute_max_attempts`, default 3). Each iteration's codegen call may return Python (success), raise `MissingData` (the structured insufficient-data signal), or raise `_ParseFailure` (unparseable reply); exec failures and critique REVISE verdicts feed the next iteration's `prev_code` + `prev_failure`. The critique sees the same domain context as codegen (`question`, `prev`, `computation`/`presentation` JSON, the code, the result). Returns the final answer as a plain `str` on ACCEPT, or — if the budget exhausts after at least one successful exec — the most recent uncritiqued result as a fallback. Fails with `StepFailed("compute", …)` when no iteration ever execs cleanly; propagates `MissingData` (with the same fallback rule) — the orchestrator catches that and runs up to `recovery_max_rounds` re-planning rounds before giving up.

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
                                                     see skunk/corpus.py)
Tier 2  PNG render at 300 dpi + vision LLM          — live, in-memory bytes (corpus.render_page_b64)
```
(`page` here = `PageRef.page` = 1-based PDF page index)

Tier escalation only happens when the chosen tier reports "value not present". There is no cross-page search — if retrieve picked the wrong pages, the bug is in retrieve, not extract. This is what makes the eval decomposition work.

## Logging & observability

There is **one** observability stream and **one** non-observability result
object — kept strictly separate:

- **Observability = the event stream.** All output renders through one function,
  `render_line` (in `src/skunk/trace.py`, the logging spine) — no external dependency. Process-scoped
  stdlib logs reach it via `_LineFormatter` on the root handler (`configure_obs`
  installs it; the `skunk.*` tree runs at INFO, the root at WARNING to mute
  third-party chatter). `HarnessContext.emit(source, message, **fields)` is the
  request-scoped entry point; each event is **captured** to `ctx.events`
  (per-question, for the trace dump) and the JSONL sink always; **streamed** to
  the question's own `.log` file when `ctx.log_path` is set (live, flushed
  per-event — so a single question's stream lands in its own file and survives a
  crash); and **echoed** to the shared console only when `ctx.verbose`. Capture
  and the per-question file live in `emit` itself (keyed on `ctx`), not in a
  process-global logging framework, because the eval runs questions concurrently
  — only `emit` knows which question an event belongs to. (The per-question file
  omits `uid`, implied by its filename; the interleaved console prepends it.)
- **Result = `ExecutionResult`** (`src/skunk/result.py`) — the orchestrator's
  typed *return contract*: `question` / `answer` / `failed` / `failure_reason`,
  read programmatically (the eval harness builds its report from it; a future UI
  would too). It is **not** logging and holds no per-step record.

**Operator boundaries are events, emitted by the caller.** The orchestrator's
`_execute_with_tracing` wraps every operator call and emits one `("orchestrator",
"step")` event per step (op / `elapsed_s` / `output_desc` / `output_full` /
`error`) — the single source of truth for "what ran, in what order, how long,
with what result". It is **caller-owned**: the orchestrator logs each operator's
boundary, so operators never self-report their own start/end. Every event
(boundary and internal alike) is stamped with the `(step_idx, op)` of the
operator it fired under: the orchestrator opens `ctx.step(step_idx, op)` around
each traced call (a thread-local frame, since branches fan out across worker
threads), and step ids are allocated atomically. The trace dump (`eval/util.py`)
groups events by `step_idx`, using the `"step"` event as each step's header and
the rest as its internals — no sentinel event, no `_step` reservation, no
separate step structure.

**The convention — log each fact at the layer that owns it, and only there:**

- **Caller owns boundaries.** Operator op/timing/result/error are the
  orchestrator's `("orchestrator", "step")` event. Operators MUST NOT emit their
  own `"starting"` / `"done"` events.
- **Callee owns its internals.** Only extract knows its tier/sample loop, only
  compute knows its codegen/critique loop, only the agent loop knows its tool
  observations — those emit from the operator itself.
- **LLM I/O is logged once, at the lowest layer that owns the fact.** Raw
  request/response/tokens/latency → `LLMClient` (`_call_gemini` / `stream`).
  Parse/validation retries → `PromptedCall`. Semantic decisions → the operator.
  No layer re-logs another layer's fact.
- **`message` is a stable snake_case event-key literal; every variable goes in
  `**fields`** — never interpolated into the message string (keeps events
  groupable/filterable), e.g.
  `emit("extract", "sample", tier="parsed_json", idx=i, n=n, n_entries=…)`.
- **Severity is automatic.** `emit` levels an event `warning` when its message
  ends in `_failed` or it carries an `error` field, else `info`; pass `level=`
  to override.
- **Request-scoped → `ctx.emit`; process-scoped → stdlib `logging`.** If a
  question's `ctx` is in scope, use `ctx.emit`. Code with no per-question ctx
  (build pipelines under `page_index/`, offline prep under `search_agent/`,
  library warnings such as `LLMClient`'s retry path) uses
  `logging.getLogger(__name__)` (or `trace.get_logger(__name__)`) — both render
  through the same `configure_obs` pipeline.

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

## Post-merge integration status

The teammate's `SearchAgent` was merged in (`refs/heads/skunk`) under
`src/skunk/search_agent/` as a self-contained subtree, wired through
`RetrieveOp` (the `search_agent` backend) when `config.retriever == "search_agent"` (default).
The first-pass merge was deliberately conservative; subsequent passes
have folded most of the agent into the framework's shared infrastructure.

### Done

- **`PromptedCall` for the agent prompt.** `make_search_agent_prompt()` (in `src/skunk/search_agent/prompted_call.py`) builds a `PromptedCall` that owns the agent's Jinja template and supplies `max_steps` / `max_pages` via a `template_vars` provider. Corpus / few-shots / lessons overrides targeting `search_agent` (or `"*"`) flow through automatically. The standalone `prompts.yaml` is gone.
- **Unified prompt-assembly engine.** `PromptedCall.assemble_system_prompt(ctx)` is now a single Jinja render: subclasses' `system_prompt` is a template, override sections (`corpus` / `few_shots` / `lessons`) are exposed as variables plus a pre-rendered `default_tail`. Layout is no longer hard-coded — subclasses decide where each section lands. The dataset blurb that the abandoned `officeqa_special_notes` slot in `prompts.yaml` was reaching for now lives in `config/prompts/treasury_bulletin.yaml` as a `corpus` override.
- **Tracer → `ctx.emit`.** Every per-step event in `SearchAgent.retrieve()` (system / question / observation / error / validation_failed) flows through `ctx.emit("search_agent", …)` into the orchestrator's event stream. The model's per-step output is no longer emitted by the agent — `LLMClient.stream` logs it once as the attributed call envelope (`output_text`), following the information-ownership rule that the LLM client owns the call envelope. `src/skunk/search_agent/tracer.py` is gone; the offline `prep/harness.py` script uses an inline `_OfflineCtx` stub that writes the same events to a per-question file.
- **`branch.key` / `branch.period` threading.** `SearchAgent.retrieve(ctx, question, *, branch_key=None, branch_period=None)` now folds the branch hints into the initial user message. `RetrieveOp._run_search_agent` forwards them.
- **One Python sandbox.** `src/skunk/pyexec.py` is now a thin wrapper over the smolagents-derived `LocalPythonExecutor`, a top-level shared module at `src/skunk/local_python_executor.py` (the same engine the search-agent loop uses via `multi_turn_agent`). It was hoisted out of `search_agent/utils/` post-merge — `search_agent/` is a consumer of the sandbox, not its owner. `compute` and `lookup_external` get the real allowed-import list, the `DANGEROUS_MODULES` / `DANGEROUS_FUNCTIONS` blocklists, and AST-walked evaluation that rejects `exec` / `eval` / `compile`. Public API (`exec_python_with_env`, `exec_python_capture_stdout`, `strip_code_fences`) is preserved; call-sites are unchanged. `numpy` / `pandas` / `statsmodels` plus `math` / `statistics` / `datetime` are pre-injected as global bindings AND listed in the executor's `additional_authorized_imports`, so code that writes either `np.exp(...)` or `import numpy as np; np.exp(...)` works. Callables passed in `local_vars` (e.g. `fetch_fred` / `fetch_bls`) route through `send_tools` so the sandboxed code can't rebind them. Not a hardened sandbox — numpy/pandas internals still run as trusted Python — but a meaningful jump from the prior bare `exec()`.

### Open follow-ups

- Route `vector_search`'s embedding through `skunk.common.LLMClient` so it picks up the RPM limiter and retry. The agent's **chat-stream is now done**: `MultiTurnAgent._generate` calls `LLMClient.stream(...)`, which shares the client and runs under the rate limiter + retry and emits the attributed call envelope. What remains is the embedding path — `vector_search` (`search_tools.py`) still calls `models.embed_content` on its own `genai.Client`; folding it into an `LLMClient.embed_query()` entry-point would finish the unification.

### Intentionally not unified

- **`prep/officeqa_eval.py` vs `eval/util.py`.** Not duplicated on inspection — corpus-specific URL / filename parsing vs harness-agnostic trace dumps.
- The `src/carnot/agents/utils.py` style tweak that came with the merge (`parse_code_blobs` dedent → f-string) lives outside this skunk subtree; left to the main-Carnot maintainers.

## What is intentionally NOT in this design

- **Agentic loops confined to retrieve.** When `config.retriever == "search_agent"`, the retrieve operator runs an iterative tool-using LLM loop (`SearchAgent` in `src/skunk/search_agent/`). The other three operators (`extract`, `lookup_external`, `compute`) execute once per call; failure is recorded in the trace. Missing-data recovery is the bounded replan loop in the orchestrator, not per-operator agent loops.
- **No per-table/per-figure catalog rows.** Page-level granularity matches the benchmark's `source_docs?page=N` labels and the existing `cache/tables/` structure. Going finer adds rows without improving recall.
- **No PZ runtime dependency.** This repo is plain Python + GCP Vertex AI (google-genai); PZ stays out of the runtime path.
