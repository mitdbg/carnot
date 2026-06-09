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
                          │  Plan (branches list)
                          ▼
                    ┌────────────┐
                    │orchestrator│  parallel branches; one shared retrieve sweep
                    └─────┬──────┘
                          │
       ┌──────────────────┼──────────────────┐
       │                  │                  │
       ▼                  ▼                  ▼
  ┌─────────┐        ┌─────────┐        ┌─────────────┐
  │retrieve │        │ extract │        │lookup_extern│   per-op operators
  │ catalog │        │ tier 1-2│        │   gemini    │
  └─────────┘        └─────────┘        └─────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
                    ┌──────────────────┐
                    │     compute      │   chain terminator:
                    │  code → exec     │   codegens from the
                    │                  │   question, execs
                    └────────┬─────────┘   to the answer
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

**Two retrieval methods.** `RetrieveOp` in `src/skunk/retrieve.py` picks a backend per call: it honors `ctx.config.golden_pages` first (so `eval/eval_e2e.py --golden` bypasses retrieval entirely via the golden branch), then routes on `config.retriever`:

- **`page_index_old`** (default) — the page-index retriever at `src/skunk/page_index/query.py` (`PageIndexRetriever`), querying the offline-built catalog + concept tree. (The `_old` suffix is a vestige of the merge; this is the live page-index path.) See its query path below.
- **`search_agent`** — the iterative ChromaDB + LLM-loop retriever vendored under `src/skunk/search_agent/`.

**Multi-scan query path (`page_index_old`).** A question's `retrieve` branches are run **together**, not in isolation, by `PageIndexRetriever.retrieve_all(ctx, branches)`, so the expensive LLM semantic filter scans each unique candidate page **at most once per question**:

  1. **Per-branch cheap candidates** (parallel, per branch — `_candidates_for`):
     - **ToC chapter pick** (`_pick_chapters` in `query.py`) — one LLM call **per era** judges every chapter in that era's taxonomy independently (any number may match; errs toward inclusion) against the branch's `key`/`period`; every page under a kept chapter becomes a candidate.
     - **Year filter** (`_year_filter`) — keeps only candidates whose page-side `date_interval` overlaps the branch period, and **drops undatable pages** (front matter / dividers / OCR-broken — no benchmark gold page is undatable). Honors the `periods_covered` ≠ publication-date distinction above. No-op on an unparseable period.
  2. **One global semantic filter** (`_multi_semantic_filter`) — the union of all branches' candidates is deduped, and each unique page is scored **once** against **all** branch targets at once. The filter sees only a compact per-page SUMMARY (titles/column-row labels/dates — no numbers) and returns a B×K boolean matrix (one row per page, one column per target); a batch whose reply is malformed degrades to keep-all (recall over precision; extract is the precision gate). Batched (`config.semfilter_batch_size`) and parallel across batches. Skippable via `config.retrieve_skip_semfilter` (ToC + year-filter only, for the recall-ceiling ablation). A page routes to branch *i* iff *i*'s cheap filters kept it **and** the matrix marks it relevant to target *i* — so sibling-concept branches (e.g. "total public debt" vs "debt held by the public") that share a page each keep only their own column.

  Continuation pages (a table/figure spilling onto the next page) are merged into their anchor at build time (`scan.merge_continuations`, run as the `merge_continuations` pipeline pass): the continuation's metadata is folded into the previous content page, the continuation is dropped, and the anchor records it in `continuation_pages`. So only the merged anchor is scored by the passes above; the retriever then expands each kept anchor into its **member refs** (the contiguous anchor + folded continuation run) so extract reads the anchor's merged text once (continuation pages carry no text of their own) and the vision tier renders every page of the continued table.

  The `search_agent` backend has no shared-scan benefit, so `run_all` runs it per branch (a fresh agent each) with per-branch failure isolation; a single branch's `StepFailed` is attributed to that branch alone and does not sink its siblings.

  **Page store** (`page_index/store.py`, `PageStore`) — the artifact's source of truth for page CONTENT, with two thread-safe access paths: `text(ref)` and `image(ref)`. The `page_store` pipeline stage writes `pages/<bulletin>.json` (`{page: text}`), where each anchor's text is its + its folded continuation pages' text joined (reusing the build's `chop_bulletin`/`elements_to_text`) with a figure note appended — so the store stays a dumb page→text map and continuation merge lives in the build. `extract.py` reads Tier-1 text and the vision tier's **page images** through `PageStore` and no longer touches the corpus parsed-JSON/PDFs; images are rendered on demand at 200 DPI (`renders/<bulletin>/<page>.png`) and cached, with at-most-once rendering per page under concurrency. (Extract therefore requires the page-store artifact to exist.)

## The 4 operators

- **`retrieve(key, period)`** — only chain head. Returns `list[PageRef]` with `(month, page)` populated (where `page` is the 1-based PDF page index). Dispatched by `RetrieveOp` to one of two backends (or short-circuited by `ctx.config.golden_pages`) — see "Two retrieval methods" above.
- **`extract(key, period, visual_only?)`** — reads `ctx.question` and the located pages via tier dispatch (parsed JSON → vision). Returns `list[AnnotatedValue]` — each entry has a `description`, `value`, `unit`, and one of three **kinds**: `scalar`, `vector` (1-D series with one varying dim), or `table` (2-D grid with row/col dims). Vector/table cells are always primitive scalars; nesting beyond those shapes is rejected by the extract parser before an `AnnotatedValue` is constructed. `extract` is invoked automatically by the orchestrator on every `RetrieveBranch` — `visual_only` is set on the branch and threaded through. Pass `visual_only=True` to skip Tier 1 and go straight to vision (use for charts/figures).
- **`lookup_external(nl)`** — chain-head capable. Runs a `LookupAgent` (a `MultiTurnAgent` tool loop: FRED / BLS / World Bank / Tavily / fetch_url, ≤`lookup_max_steps` steps) over a natural-language description of external factual data (`nl`); the agent commits a dict of `AnnotatedValue` fields, which `LookupExternalOp` parses into `list[AnnotatedValue]` (one entry) on the trusted side. Use for CPI-U, FX rates, event dates, named entities (bureau names), and any fact not in the bulletin corpus. The agent infers the appropriate `kind` and `unit` (including `text` for strings).
- **`compute()`** — chain terminator that subsumes formatting. Reads `ctx.question` plus the upstream extracted/looked-up values; runs a single loop (`ComputeOp.run`) of codegen → in-process exec under one shared budget (`compute_max_attempts`, default 3). Each iteration's codegen call may return Python (success), raise `MissingData` (the structured insufficient-data signal), or raise `_ParseFailure` (unparseable reply); parse and exec failures feed the next iteration's `prev_code` + `prev_failure`. Returns the result of the first clean exec as a plain `str`. Fails with `StepFailed("compute", …)` when no iteration ever execs cleanly; propagates `MissingData` — the orchestrator catches that and runs up to `recovery_max_rounds` re-planning rounds before giving up.

  Each recovery round, `Planner.replan` returns a **`PlanDiff`**, not a fresh plan: `add` branches to run and `drop` indices into the prior branch list (numbered in the replan prompt). The orchestrator applies it **positionally** (`Orchestrator._apply_diff`), keeping `outcomes[i]` aligned with `plan.branches[i]` — a dropped index removes the branch *and* the data it gathered (so compute never sees retracted data), added branches are appended, and unchanged branches keep their already-gathered outcomes and are never re-run. This replaces the old "diff replanned branches by value-equality and additively merge" scheme, which couldn't subtract and re-ran any reworded-but-unchanged branch.

## Plan shape

A **Plan** is a pydantic model: just a `branches` list. Compute reads everything it needs — the calculation to perform *and* the output format (units, precision, list shape) — from the verbatim question, not from a planner paraphrase, so the plan carries no separate requirements object. The Python layout mirrors the wire JSON exactly, so `Plan.model_validate_json` / `Plan.model_dump_json` round-trip with no custom translation. There is no multi-step decomposition — every plan is exactly one terminal compute. `PlannerPromptedCall` (in `src/skunk/plan.py`) emits a `Plan`; the orchestrator runs branches in parallel, then feeds the merged `list[AnnotatedValue]` to compute. Concretely it runs all `retrieve` branches through **one shared retrieve sweep** (`Orchestrator._run_retrieve_phase` → `RetrieveOp.run_all`, the multi-scan above), then dispatches each branch's `extract` (and every `lookup_external`) in parallel; results stay aligned with `plan.branches` by position.

Two branch shapes:

- `RetrieveBranch(key, period, as_of=None, visual_only=False)` — `retrieve` then `extract` are dispatched for the branch by the orchestrator (extract reads `visual_only` from the branch; `as_of` is the reporting/vintage month, recorded for extraction and not used by retrieval).
- `LookupBranch(target, src=None)` — single `lookup_external` call.

Canonical JSON shape (one branch):

```json
{
  "branches": [
    {"kind": "retrieve", "key": "national defense expenditures", "period": "CY1940"}
  ]
}
```

Per-branch field semantics (`key` / `period` / `as_of` / `visual_only`, when to use `null`, etc.) live in `PlannerPromptedCall.system_prompt` in `src/skunk/plan.py` — that prompt is the canonical schema spec. Runtime contracts (`PageRef`, `AnnotatedValue`, the scalar / vector / table kind taxonomy with primitive-only cells) live in the dataclasses in the same file.

The `AnnotatedValue.kind` taxonomy:

- `scalar` — `value` is `int | float | str`. Single facts.
- `vector` — `value` is `{index_label: scalar}`; `index_name` names the varying dim.
- `table` — `value` is `{row_label: {col_label: scalar}}`; `row_name` / `col_name` name the two dims.

Cells must be primitive scalars (`int` / `float` / `str`, no `bool`, no nesting beyond these shapes). The extract parser rejects deeper structures before constructing an `AnnotatedValue`.

## Per-page tier dispatch in extract

Once retrieve has named specific pages, extract chooses how to read each one. Both tiers read CONTENT through the page-index `PageStore` (`page_index/store.py`, `get_page_store`) keyed by `PageRef` — extract no longer touches the corpus parsed-JSON / PDFs at query time:

```
Tier 1  page-store text (TextExtractor)    — the anchor's merged member text via store.text(ref),
        emitted as tier=parsed_json          built offline from the parsed JSON; structured text + tables
Tier 2  page-store image + vision LLM       — store.image(ref): PNG rendered on demand at 200 DPI and
        (VisionExtractor)                    cached to renders/<bulletin>/<page>.png
```
(`page` here = `PageRef.page` = 1-based PDF page index. Tier-1 events keep the `tier=parsed_json` label for continuity, though the text now comes from the store, not the corpus.)

Tier escalation only happens when the chosen tier reports "value not present" (or `visual_only` skips straight to Tier 2). There is no cross-page search — if retrieve picked the wrong pages, the bug is in retrieve, not extract. This is what makes the eval decomposition work.

## Model routing

A run can mix models per call-site. Resolution mirrors the effort knob:

- **`PromptedCall` sites** (planner, extract tiers, compute.codegen, question_explainer, …) resolve their model as `config.model_overrides.get(name, config.llm_model)` — see `PromptedCall._resolve_model`. So the default `llm_model` (env `SKUNK_LLM_MODEL`) applies everywhere unless a site is pinned via `SKUNK_MODEL_OVERRIDES` (`name=model,...`). Example: run on Pro but keep `question_explainer` on cheap Flash.
- **Agent loops** (`LookupAgent`, search agent) use `config.agent_model_id or config.llm_model` (env `SKUNK_AGENT_MODEL`); they don't read the override map.
- **Per-model rate limiting.** Each distinct model gets its own token-bucket keyed `llm:<model>` (`_retry_call`/`_aretry_call` via `_llm_model_rpm`). A model's RPM comes from `SKUNK_MODEL_RPM` (`model=rpm,...`), falling back to `SKUNK_LLM_RPM` (default 1000) for any model not listed — so single-model runs are unchanged. Lets a 150-RPM Pro and a high-RPM Flash run concurrently without throttling each other.
- **Thinking-mode floor.** Gemini 3.x Pro is thinking-only and rejects both `thinking_budget=0` and `MINIMAL`; `_effort_to_thinking_config` floors `off`/`minimal` to `LOW` for such models (`_requires_thinking`), so an `effort="off"` call-site still works (at LOW thinking) when pointed at Pro.

## Logging & observability

There is **one** observability stream and **one** non-observability result
object — kept strictly separate:

- **Observability = the event stream.** All output renders through one function,
  `render_line` (in `src/skunk/trace.py`, the logging spine) — no external dependency. Process-scoped
  stdlib logs reach it via `_LineFormatter` on the root handler (`configure_obs`
  installs it; the `skunk.*` tree runs at INFO, the root at WARNING to mute
  third-party chatter). `ExecutionContext.emit(message, level=None)` is the
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
`_execute_with_tracing` wraps every operator call and emits one `step …` event
per step (`elapsed_s` and a short `output` description — or `error` — interpolated
into the message) — the single source of truth for "what ran, in what order, how
long, with what result". It is **caller-owned**: the orchestrator logs each
operator's boundary, so operators never self-report their own start/end. Every
event (boundary and internal alike) is stamped with the `(step_idx, op)` of the
operator it fired under: the orchestrator opens `ctx.step(op)` around each traced
call. That frame lives in a module-level `ContextVar` (a single frame, not a
stack — steps don't nest; per-`asyncio.Task` copy-on-write isolates the concurrent
branches, which run as tasks on the question's one worker thread), and the
`ExecutionContext` owns the monotonic step-index allocator. The trace dump
(`eval/util.py`) groups events by `step_idx`, headed by `Step N: op` — no sentinel
event, no `_step` reservation, no separate step structure.

**The convention — log each fact at the layer that owns it, and only there:**

- **Caller owns boundaries.** Operator op/timing/result/error are the
  orchestrator's `step …` event. Operators MUST NOT emit their own
  `"starting"` / `"done"` events.
- **Callee owns its internals.** Only extract knows its tier/sample loop, only
  compute knows its codegen/exec loop, only the agent loop knows its tool
  observations — those emit from the operator itself.
- **LLM I/O is logged once, at the lowest layer that owns the fact.** Raw
  request/response/tokens/latency → `LLMClient` (`_call_gemini` / `stream`).
  Parse/validation retries → `PromptedCall`. Semantic decisions → the operator.
  No layer re-logs another layer's fact.
- **`message` is one string: a stable snake_case event key, then variables
  interpolated inline.** There is no separate `source` (the active step's `op`
  identifies the emitter), e.g.
  `emit(f"sample tier=parsed_json idx={i} n={n} n_entries={k}")`. Lead with the
  event key so the stream stays greppable; keep large blobs (full prompts,
  transcripts, generated code) out of the **message** — log a count or short `repr`.
- **`kind` + `data` + `t` are the trace viewer's structured channel (optional).**
  The *message* stays a scannable one-liner, but three extra fields ride alongside
  it for the post-hoc trace viewer (`eval/trace_viewer/`). `kind` is the event's
  semantic role (`system` / `user` / `assistant` / `observation` / `error` / `call`
  / `plan` / `summary` / `step` / `note`) used for color-coding, inferred from the
  message's leading event key when omitted (`common.infer_kind`). `data` is a
  JSON-able structured payload. `t` is seconds since the question began (the
  viewer's timeline axis), stamped automatically on every event. The structured
  captures, each at the layer that owns the fact:
    - **Agent turns** (multi-turn `retrieve` / `lookup_external`): the system
      prompt, each assistant turn, and structured observation blocks
      (`multi_turn_agent`).
    - **Single-shot operator I/O** (`planner` / `question_explainer` / `extract` /
      `compute` / `replanner`): system + user + each assistant reply, emitted once
      at the `PromptedCall.call` chokepoint (the multi-turn branch never reaches it,
      so there is no double-log).
    - **Plan + node summaries** (orchestrator): one `plan` event per revision
      (branches carry a stable `branch_id`; replan revisions carry the
      `reason`/`missing` that triggered them), and each `step` boundary's
      `summarize_value` node summary (`result.summarize_value`).
  `kind`/`data`/`t` are captured to `ctx.events` and the JSONL sink but
  **deliberately excluded from the rendered console / `.log` line** (`render_line`
  skips them; the line already carries a wall-clock `HH:MM:SS`) so the human
  one-liner is unchanged. Use `data` only for the genuinely large/structured
  payloads a viewer needs; everyday events stay message-only.
- **Severity is automatic.** `emit` levels an event `warning` when its message
  contains a `_failed` event key or an `error=` field, else `info`; pass `level=`
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

The page-index artifact is the only on-disk index in the runtime path:

- `artifact/page_index/` — the shipped, in-repo build output (per-bulletin `catalog/*.jsonl`, `concept_tree.json`, `manifest.json`). Tracked in git so a fresh checkout works without a rebuild. The retriever reads it by default; override with `$SKUNK_PAGE_INDEX_DIR`. The build pipeline (`src/skunk/page_index/pipeline.py`) writes here too (`--output-dir`, default `artifact/page_index`).
- Parsed-JSON corpus at `$OFFICEQA_PARSED_JSON_DIR` (default `~/Desktop/officeqa/treasury_bulletins_parsed/jsons/`) — read by the page-index builder (which bakes the Tier-1 text into the page store); extract reads it at query time only through the store, not the corpus directly.

LLM completions are **not** cached. Tier 2 PNG renders are cached to disk by the page store (`renders/<bulletin>/<page>.png`, 200 DPI, rendered at most once per page under concurrency). There is no DSL plan cache — the planner runs once per question.

## Post-merge integration status

The teammate's `SearchAgent` was merged in (`refs/heads/skunk`) under
`src/skunk/search_agent/` as a self-contained subtree, wired through
`RetrieveOp` (the `search_agent` backend) when `config.retriever == "search_agent"` (the page-index backend, `page_index_old`, is the default).
The first-pass merge was deliberately conservative; subsequent passes
have folded most of the agent into the framework's shared infrastructure.

### Done

- **Dual-channel step protocol.** Each step the model emits exactly one fenced block, and the fence language is the channel: a ` ```python ` block is a tool call (exec'd in the `LocalPythonExecutor` with the agent's tools bound; its output becomes the next observation), and a ` ```json ` block is the final answer (parsed directly as data — never exec'd — and gated by `validate_final_answer`). The final answer is data, so it leaves as data: there is no `final_answer` tool, `FinalAnswerException`, or `is_final_answer` flag routing it back through the sandbox. A stray `final_answer(...)` call in a python block just raises `NameError` and is fed back as a misfire. `call()` holds all per-question state in locals, so one agent instance is reentrant across concurrent questions.
- **`MultiTurnAgent` exclusively owns the agent prompt structure.** The base holds the one canonical `_SYSTEM_TEMPLATE` skeleton — step protocol, `## Tools` section, `## Final answer` framing, and every Jinja slot (`{{ tools_doc }}`, `{{ max_steps }}`, …). A subclass (`SearchAgent`, `LookupAgent`) never writes a template or a slot; it supplies only CONTENT fragments — `name`, `briefing` (identity + task + operating guidance, rendered at the top), and `final_answer_doc` (the JSON payload schema) — which `MultiTurnAgent.__init__` drops into the skeleton (splicing each tool's `doc` into `{{ tools_doc }}` and wiring the vars). Variable definition lives in the base too: it supplies the universal `max_steps`, and a subclass adds extra tool-doc vars through `prompt_vars(ctx)` (search's `max_pages`). Corpus / few-shots / lessons overrides targeting the agent's `name` (or `"*"`) flow through the `PromptedCall` automatically. The standalone `prompts.yaml` and the old `make_search_agent_prompt()` factory are both gone.
- **Unified prompt-assembly engine.** `PromptedCall.assemble_system_prompt(ctx)` is now a single Jinja render: subclasses' `system_prompt` is a template, override sections (`corpus` / `few_shots` / `lessons`) are exposed as variables plus a pre-rendered `default_tail`. Layout is no longer hard-coded — subclasses decide where each section lands. The dataset blurb that the abandoned `officeqa_special_notes` slot in `prompts.yaml` was reaching for now lives in `config/prompts/treasury_bulletin.yaml` as a `corpus` override.
- **Tracer → `ctx.emit`.** Every per-step event in `SearchAgent.retrieve()` (system / question / observation / error / validation_failed) flows through `ctx.emit(message)` into the orchestrator's event stream, stamped with the enclosing `retrieve` step's `op`. The model's per-step output is no longer emitted by the agent — `LLMClient.stream` logs it once as the attributed call envelope (latency/tokens), following the information-ownership rule that the LLM client owns the call envelope. `src/skunk/search_agent/tracer.py` is gone; the offline `prep/harness.py` script uses an inline `_OfflineCtx` stub (its `emit(message, level=None)` matches the real ctx) that writes the same events to a per-question file.
- **`branch.key` / `branch.period` threading.** `SearchAgent.retrieve(ctx, question, *, branch_key=None, branch_period=None)` now folds the branch hints into the initial user message. `RetrieveOp._run_search_agent` forwards them.
- **One Python sandbox.** `src/skunk/pyexec.py` is now a thin wrapper over the smolagents-derived `LocalPythonExecutor`, a top-level shared module at `src/skunk/local_python_executor.py` (the same engine the search-agent loop uses via `multi_turn_agent`). It was hoisted out of `search_agent/utils/` post-merge — `search_agent/` is a consumer of the sandbox, not its owner. `compute` and `lookup_external` get the real allowed-import list, the `DANGEROUS_MODULES` / `DANGEROUS_FUNCTIONS` blocklists, and AST-walked evaluation that rejects `exec` / `eval` / `compile`. Public API (`exec_python_with_env`, `exec_python_capture_stdout`, `strip_code_fences`) is preserved; call-sites are unchanged. `numpy` / `pandas` / `statsmodels` plus `math` / `statistics` / `datetime` are pre-injected as global bindings AND listed in the executor's `additional_authorized_imports`, so code that writes either `np.exp(...)` or `import numpy as np; np.exp(...)` works. Callables passed in `local_vars` (e.g. `fetch_fred` / `fetch_bls`) route through `send_tools` so the sandboxed code can't rebind them. Not a hardened sandbox — numpy/pandas internals still run as trusted Python — but a meaningful jump from the prior bare `exec()`.

### Open follow-ups

- Route `vector_search`'s embedding through `skunk.common.LLMClient` so it picks up the RPM limiter and retry. The agent's **chat-stream is now done**: `MultiTurnAgent._generate` calls `LLMClient.stream(...)`, which shares the client and runs under the rate limiter + retry and emits the attributed call envelope. What remains is the embedding path — `vector_search` (`search_tools.py`) still calls `models.embed_content` on its own `genai.Client`; folding it into an `LLMClient.embed_query()` entry-point would finish the unification.

### Intentionally not unified

- **`prep/officeqa_eval.py` vs `eval/util.py`.** Not duplicated on inspection — corpus-specific URL / filename parsing vs harness-agnostic trace dumps.
- The `src/carnot/agents/utils.py` style tweak that came with the merge (`parse_code_blobs` dedent → f-string) lives outside this skunk subtree; left to the main-Carnot maintainers.

## What is intentionally NOT in this design

- **Agentic loops confined to retrieve and lookup_external.** Two operators run iterative tool-using LLM loops (both `MultiTurnAgent`s): retrieve when `config.retriever == "search_agent"` (`SearchAgent` in `src/skunk/search_agent/`), and `lookup_external` always (`LookupAgent`). `extract` and `compute` do not loop agentically — they execute once per call (compute's internal codegen→exec retries are a fixed bounded budget, not a tool loop), and failure is recorded in the trace. Crucially, *missing-data recovery* is the bounded replan loop in the orchestrator, not a per-operator retry loop: no operator re-plans or re-dispatches itself on failure.
- **No per-table/per-figure catalog rows.** Page-level granularity matches the benchmark's `source_docs?page=N` labels and the existing `cache/tables/` structure. Going finer adds rows without improving recall.
- **No PZ runtime dependency.** This repo is plain Python; LLM calls go through the AI Studio Gemini API (`google-genai`, authenticated by `GEMINI_API_KEY`) or OpenRouter (`SKUNK_LLM_PROVIDER=openrouter`) — not Vertex/ADC. PZ stays out of the runtime path.
