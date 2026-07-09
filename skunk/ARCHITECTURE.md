# Architecture

> **Layout note (2026-07-07).** The OfficeQA application layer moved out of this
> repo dir to `../grc-officeqa/` (library-hardening refactor, see
> REFACTOR_PLAN.md): the page-index build + page store are now
> `grc-officeqa/officeqa/page_index/`, the Treasury corpus accessors
> `officeqa/corpus.py`, the app config `officeqa/config.py` (`SkunkConfig`), and
> the eval harness `grc-officeqa/eval/`. Extract reads corpus content only
> through the `skunk.page_store.PageContentStore` protocol; the Treasury
> implementation is `officeqa/page_store.py`. This document describes the
> pipeline as exercised by that app — path references to `page_index/`, `eval/`,
> and `corpus.py` below refer to their new grc-officeqa homes.

## Why this design

OfficeQA questions ask about specific tables and figures inside a corpus of 696 monthly U.S. Treasury Bulletin PDFs (1939–2025). A naive date-driven retrieval agent — "the question mentions 1940, look in the 1940 bulletin" — fails on **81%** of the benchmark, because the answer-bearing table for period *P* often lives in a *later* bulletin (publication lag) or in a *retrospective summary* in a single mid-year issue. The median miss is +2 months; the long tail goes out to +9 years.

The fix: retrieve at the **page** level, indexed by **what each page reports**, not by when its bulletin was published. Retrieval names pages; the actual reading is a separate concern handled by the extract operator.

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
  │  agent  │        │ tier 1-2│        │   gemini    │
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

## Retrieval: the search agent

`RetrieveOp.run_all` in `src/skunk/retrieve.py` is the **single retrieval seam**: it is the
only layer that inspects `ctx.config.golden_pages` / `config.retriever`, and it normalizes
retrieval to one per-branch result — `BranchRetrieval(pages)` (`src/skunk/common.py`), the
whole pages extract will read. A branch that fails surfaces as a per-branch `StepFailed` in
the same list, so one branch failing doesn't sink its siblings. Dispatch:

- **`search_agent`** (the only backend) — the iterative ChromaDB + LLM-loop retriever under
  `src/skunk/search_agent/` (`SearchAgent`, a `MultiTurnAgent` with vector-search / grep /
  read-document / prune tools over the chunked corpus). Branches run in parallel, a fresh
  agent each, with per-branch failure isolation; the agent's page-key output is parsed to
  `PageRef`s.
- **golden bypass** — `config.golden_pages` (the eval harness's `--golden` ablation) injects
  the benchmark's gold pages verbatim and skips the agent entirely.

(The earlier catalog-based `page_index` retriever — ToC chapter picks, year filter, the
multi-branch semantic filter, and the block-selection tournament — was retired; the
page-index artifact now serves only extract, through the page store below. The offline
build pipeline (`src/skunk/page_index/pipeline.py`: scan → continuation/notes linking →
ToC → catalog → eras → page store) still produces that artifact, and extract still uses
the build's continuation/notes links to expand a page to the extra pages of a continued
table.)

  **Page store** (`page_index/store.py`, `PageStore`) — the artifact's source of truth for page CONTENT, with two thread-safe access paths: `text(ref)` and `image(ref)`. The `page_store` pipeline stage writes `pages/<bulletin>.json` (`{page: text}`): each page's own text (reusing the build's `chop_bulletin`/`elements_to_text`) with a figure note appended — a dumb page→text map, one entry per catalog row. `extract.py` reads Tier-1 text and the vision tier's **page images** through `PageStore` and no longer touches the corpus parsed-JSON/PDFs; images are rendered on demand at 200 DPI (`renders/<bulletin>/<page>.png`) and cached, with at-most-once rendering per page under concurrency. (Extract therefore requires the page-store artifact to exist.)


## The 4 operators

- **`retrieve(key, period)`** — only chain head. Returns `list[PageRef]` with `(stem, page)` populated (where `page` is the 1-based PDF page index). Dispatched by `RetrieveOp` to the search-agent backend (or short-circuited by `ctx.config.golden_pages`) — see "Retrieval: the search agent" above.
- **`extract(key, period, visual_only?)`** — reads `ctx.question` and the located pages via tier dispatch (parsed JSON → vision). Returns `list[AnnotatedValue]` — each entry has a `description`, `value`, `unit`, one of three **kinds** (`scalar`, `vector` (1-D series with one varying dim), or `table` (2-D grid with row/col dims)), and machine-stamped **provenance** (`bulletin` / `pages` / `requested_period` / `retrieve_key`, copied from the source page + branch — see the `AnnotatedValue.kind` section below). Vector/table cells are always primitive scalars; nesting beyond those shapes is rejected by the extract parser before an `AnnotatedValue` is constructed. `extract` is invoked automatically by the orchestrator on every `RetrieveBranch` — `visual_only` is set on the branch and threaded through. Pass `visual_only=True` to skip Tier 1 and go straight to vision (use for charts/figures).
- **`lookup_external(nl)`** — chain-head capable. Runs a `LookupAgent` (a `MultiTurnAgent` tool loop: FRED / BLS / World Bank / Tavily / fetch_url, ≤`lookup_max_steps` steps) over a natural-language description of external factual data (`nl`); the agent commits a dict of `AnnotatedValue` fields, which `LookupExternalOp` parses into `list[AnnotatedValue]` (one entry) on the trusted side. Use for CPI-U, FX rates, event dates, named entities (bureau names), and any fact not in the bulletin corpus. The agent infers the appropriate `kind` and `unit` (including `text` for strings).
- **`compute()`** — chain terminator that subsumes formatting. Reads `ctx.question` plus the upstream extracted/looked-up values; runs a single loop (`ComputeOp.run`) of codegen → in-process exec under one shared budget (`compute_max_attempts`, default 3). Each iteration's codegen call emits one of two `python` forms: **(a)** assign `result` (success); or **(b)** *missing data* — assign `keep` (a `{name: value}` dict of everything to carry into the next round: the inputs the model re-states plus any partial computation it derived, e.g. the qualifying month of a multi-hop question) **and** `missing` (the identifiers + a one-line reason). `keep` is drop-by-default — anything not in it is dropped for good (there is no give-up form that silently keeps everything). Parse, exec, and malformed-block failures feed the next iteration's `prev_code` + `prev_failure`. `run()` returns a union: `Final(answer)` on a clean exec that set `result`, or `NeedsMore(keep, missing)` — it never raises `MissingData`. Fails with `StepFailed("compute", …)` when no iteration ever resolves. Across `compute_best_of_n` trials, `_vote` returns the most-frequent outcome with all `NeedsMore`s pooled into a single "missing-data" candidate (a tie never breaks in favor of missing-data); when missing-data wins, the trials' `keep`s are unioned and their reasons rendered per agent ("agent 1: …, agent 2: …").

  Recovery state is a **value pool**, not branch outcomes: the orchestrator seeds it with the initial sweep's `AnnotatedValue`s, and on each `NeedsMore` the next pool is exactly compute's `keep` values — a re-stated input is carried **with its provenance** (the model references the `input_values[i]` entry), a freshly computed value is provenance-free and marked `notes="computed"`; drop is the default, so discards are gone for good — and each replan sweep then appends its newly gathered values. Each recovery round (≤ `recovery_max_rounds`), `Planner.replan` composes a **fresh `Plan`** containing only the branches to run now — no diff, no index alignment; branch identity dies at the end of its sweep. The replan message carries the kept pool's schema, the **previous plans** (every branch tried with its fate + failure diagnostics), and compute's missing description, and the replanner is instructed to diagnose why the gathering so far did not satisfy compute before emitting branches — never re-emitting an approach that already failed the same way. A replan sweep that yields zero new values **short-circuits**: compute is skipped (it would be a no-op on an unchanged pool) and the prior `NeedsMore` drives the next replan, still consuming a round. When the budget is exhausted, `execute()` raises `MissingData` — its only remaining role, the terminal signal the eval harness records.

## Plan shape

A **Plan** is a pydantic model: just a `branches` list. Compute reads everything it needs — the calculation to perform *and* the output format (units, precision, list shape) — from the verbatim question, not from a planner paraphrase, so the plan carries no separate requirements object. The Python layout mirrors the wire JSON exactly, so `Plan.model_validate_json` / `Plan.model_dump_json` round-trip with no custom translation. There is no multi-step decomposition — every plan is exactly one terminal compute. `PlannerPromptedCall` (in `src/skunk/plan.py`) emits a `Plan`; the orchestrator runs branches in parallel, then feeds the merged `list[AnnotatedValue]` to compute. Concretely it runs all `retrieve` branches through **one shared retrieve sweep** (`Orchestrator._run_retrieve_phase` → `RetrieveOp.run_all`), then ONE shared extract sweep (`run_extract`, reading each unique page once) over the retrieved pages — every `lookup_external` runs in parallel alongside it; results stay aligned with `plan.branches` by position.

**Both branch kinds are first-class on the initial plan; the choice is per value.** The planner is instructed (advisorily — there is no parse-hook enforcement) to prefer a `retrieve` branch whenever the data could reasonably be expected to exist in the corpus, and to emit a `lookup_external` branch for a value unlikely to be in the corpus or when the question names a clear external source — rather than defaulting everything to retrieve. At replan, `lookup_external` is unrestricted. There are no verbatim checks on retrieve `key`s or lookup `src`s — both are free natural language.

Two branch shapes:

- `RetrieveBranch(key, period, visual_only=False)` — `retrieve` then `extract` are dispatched for the branch by the orchestrator (extract reads `visual_only` from the branch). There is no issue/vintage pin field: issue choice (which print/reprint to read) belongs entirely to selection.
- `LookupBranch(target, src=None)` — single `lookup_external` call. `src` is an optional publisher hint, set when the question names a clear external source.

Canonical JSON shape (one branch):

```json
{
  "branches": [
    {"kind": "retrieve", "key": "national defense expenditures", "period": "CY1940"}
  ]
}
```

Per-branch field semantics (`key` / `period` / `visual_only`, when to use `null`, etc.) live in `PlannerPromptedCall.system_prompt` in `src/skunk/plan.py` — that prompt is the canonical schema spec. Runtime contracts (`PageRef`, `AnnotatedValue`, the scalar / vector / table kind taxonomy with primitive-only cells) live in the dataclasses in the same file.

The `AnnotatedValue.kind` taxonomy:

- `scalar` — `value` is `int | float | str`. Single facts.
- `vector` — `value` is `{index_label: scalar}`; `index_name` names the varying dim.
- `table` — `value` is `{row_label: {col_label: scalar}}`; `row_name` / `col_name` name the two dims.

Cells must be primitive scalars (`int` / `float` / `str`, no `bool`, no nesting beyond these shapes). The extract parser rejects deeper structures before constructing an `AnnotatedValue`.

Each `AnnotatedValue` also carries **machine-stamped provenance** — `bulletin` (the source issue `"YYYY-MM"` the value was printed in, i.e. the publication date), `pages` (source PDF page numbers), and the originating branch's `requested_period` / `retrieve_key`. These are copied from the extract inputs (the source `PageRef`s and the `RetrieveBranch`), **never authored by the LLM** — extract stamps them onto the parsed entries in `_stamp_provenance` (`extract.py`). `bulletin`/`pages` are attributed only when every page in a single extract call shares one issue (the common single-issue branch); a call spanning multiple issues leaves them empty, since the reply carries no per-page attribution. All provenance fields default to `None`/`()`, so `lookup_external` values (which have no bulletin) and older payloads stay valid. `bulletin` sorts lexically = chronologically, so compute selects/filters by publication date with it — the load-bearing case is several vintages of the *same* series+period whose LLM-written `description`s are identical and are distinguishable only by `bulletin`. Provenance surfaces in the compute/replanner input view via `input_values_desc` (`common.py`), which renders each entry's full frame cell-by-cell (values included, so generated code sees NaN/"n/a" cells and exact magnitudes), not just its schema.

## Per-page tier dispatch in extract

Once retrieve has named specific pages, extract chooses how to read each one. Both tiers read CONTENT through the injected `PageContentStore` (`skunk/page_store.py` protocol; `ctx.page_store`, wired by the app via `Orchestrator(page_store=...)`) keyed by `PageRef` — extract never touches a corpus directly:

```
Tier 1  page-store text (TextExtractor)    — the page's text via store.text(ref),
        emitted as tier=parsed_json          built offline from the parsed JSON; structured text + tables
Tier 2  page-store image + vision LLM       — store.image(ref): PNG rendered on demand at 200 DPI and
        (VisionExtractor)                    cached to renders/<bulletin>/<page>.png
```
(`page` here = `PageRef.page` = 1-based PDF page index. Tier-1 events keep the `tier=parsed_json` label for continuity, though the text now comes from the store, not the corpus.)

Tier escalation only happens when the chosen tier reports "value not present" (or `visual_only` skips straight to Tier 2). There is no cross-page search — if retrieve picked the wrong pages, the bug is in retrieve, not extract. This is what makes the eval decomposition work.

## Human-in-the-loop

Removed (2026-07-07, library-hardening refactor). The competition-era HITL layer —
`human.py` (policy/channel/facade), `human_intervention.py` (`request_human` tool),
the orchestrator's replan-approval and recompute paths, and the `SKUNK_HUMAN_*`
config flags — was deleted wholesale; it lives in git history if a future
application wants to reintroduce it behind a proper seam.

## Model routing

A run can mix models per call-site. Resolution mirrors the effort knob:

- **`PromptedCall` sites** (planner, extract tiers, compute.codegen, …) resolve their model as `config.model_overrides.get(name, config.llm_model)` — see `PromptedCall._resolve_model`. So the default `llm_model` (env `SKUNK_LLM_MODEL`) applies everywhere unless a site is pinned via `SKUNK_MODEL_OVERRIDES` (`name=model,...`). Example: run on Pro but keep `extract` on cheap Flash.
- **Agent loops** (`LookupAgent`, search agent) use `config.agent_model_id or config.llm_model` (env `SKUNK_AGENT_MODEL`); they don't read the override map.
- **Per-model rate limiting.** Each distinct model gets its own token-bucket keyed `llm:<model>` (`_retry_call`/`_aretry_call`). A model's RPM comes from config (`llm_model_rpm`, falling back to `llm_default_rpm` for any model not listed) — so single-model runs are unchanged. Lets a 150-RPM Pro and a high-RPM Flash run concurrently without throttling each other.
- **Provider selection.** A run uses one generation provider for the whole process — Gemini (`provider=genai`, default) or OpenRouter (`provider=openrouter`) per `config.llm_provider`. Both share the same rate-limit / retry / logging scaffolding; `_retry_call`/`_aretry_call` retry transient faults (`_is_retryable`: 429 / 5xx / transport blips) with exponential backoff within that provider. Embeddings always go to Gemini.
- **Thinking-mode floor.** Gemini 3.x Pro is thinking-only and rejects both `thinking_budget=0` and `MINIMAL`; `_effort_to_thinking_config` floors `off`/`minimal` to `LOW` for such models (`_requires_thinking`), so an `effort="off"` call-site still works (at LOW thinking) when pointed at Pro.

## Logging & observability

There is **one** observability stream and **one** non-observability result
object — kept strictly separate:

- **Observability = the event stream.** `ExecutionContext.emit(message,
  level=None)` is the single entry point; each event is **captured** to
  `ctx.events` (per-question, for the trace dump) always; **streamed** as one
  JSON line to the question's own `.jsonl` file when `ctx.log_path` is set (the
  durable machine-readable record — live, flushed per-event, so a single
  question's stream lands in its own file and survives a crash); and **echoed**
  to the shared console only when `ctx.verbose` (rendered via `render_line` in
  `src/skunk/trace.py` — pure functions, no state). Everything lives in `emit`
  itself (keyed on `ctx`), not in a process-global logging framework, because the
  eval runs questions concurrently — only `emit` knows which question an event
  belongs to, and per-question file handles need no locks. (The per-question file
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
  request/response/tokens/latency → `LLMClient` (`_call_gemini_once` / `stream`).
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
  it for the post-hoc trace viewer (`scripts/trace_viewer/`). `kind` is the event's
  semantic role (`system` / `user` / `assistant` / `observation` / `error` / `call`
  / `plan` / `summary` / `step` / `note`) used for color-coding, inferred from the
  message's leading event key when omitted (`common.infer_kind`). `data` is a
  JSON-able structured payload. `t` is seconds since the question began (the
  viewer's timeline axis), stamped automatically on every event. The structured
  captures, each at the layer that owns the fact:
    - **Agent turns** (multi-turn `retrieve` / `lookup_external`): the system
      prompt, each assistant turn, and structured observation blocks
      (`multi_turn_agent`).
    - **Single-shot operator I/O** (`planner` / `extract` /
      `compute` / `replanner`): system + user + each assistant reply, emitted once
      at the `PromptedCall.call` chokepoint (the multi-turn branch never reaches it,
      so there is no double-log).
    - **Plan + node summaries** (orchestrator): one `plan` event per revision
      (branches carry a stable `branch_id`; a recovery revision is labelled `replan`
      and carries the `reason`/`missing` that triggered it), and each `step` boundary's
      `summarize_value` node summary (`result.summarize_value`).
  Note the step-budget bookkeeping: a turn that misfires (prose with no runnable
  action block, or exec-machinery failure) advances the agent's internal `turn`
  counter but **not** its `step` counter, so misfires don't burn the `agent_max_steps`
  budget; a separate `agent_max_misfires` cap (total attempts ≤ steps + misfires)
  stops a never-progressing model from looping forever.
  `kind`/`data`/`t` are captured to `ctx.events` and the per-question `.jsonl`
  but **deliberately excluded from the rendered console line** (`render_line`
  skips them; the line already carries a wall-clock `HH:MM:SS`) so the human
  one-liner is unchanged. Use `data` only for the genuinely large/structured
  payloads a viewer needs; everyday events stay message-only.
- **Severity is automatic.** `emit` levels an event `warning` when its message
  contains a `_failed` event key or an `error=` field, else `info`; pass `level=`
  to override.
- **Everything routes through `ctx`.** If a question's `ctx` is in scope, use
  `ctx.emit` — including library warnings such as `LLMClient`'s retry path, which
  threads the caller's ctx down (`llm_client._warn`) so retries are attributable
  to their question. Code with genuinely no per-question ctx (offline corpus
  prep, app build pipelines) owns its own output (stdlib `logging` / `print`);
  skunk does not configure it.

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

- `artifact/page_index/` — the shipped, in-repo build output (per-bulletin `catalog/*.jsonl`, `concept_tree.json`, `manifest.json`, the page store). Tracked in git so a fresh checkout works without a rebuild. Extract's `PageStore` reads it; override the root with `$SKUNK_PAGE_INDEX_DIR`. The build pipeline (`src/skunk/page_index/pipeline.py`) writes here too (`--output-dir`, default `artifact/page_index`).
- Parsed-JSON corpus at `$OFFICEQA_PARSED_JSON_DIR` (default `~/Desktop/officeqa/treasury_bulletins_parsed/jsons/`) — read by the page-index builder (which bakes the Tier-1 text into the page store); extract reads it at query time only through the store, not the corpus directly.

LLM completions are **not** cached. Tier 2 PNG renders are cached to disk by the page store (`renders/<bulletin>/<page>.png`, 200 DPI, rendered at most once per page under concurrency). There is no DSL plan cache — the planner runs once per question. There is no retrieval cache; `eval/eval_e2e.py --golden` is the only retrieve bypass (it injects the benchmark's gold pages to measure the extract+compute ceiling).

## Post-merge integration status

The teammate's `SearchAgent` was merged in (`refs/heads/skunk`) under
`src/skunk/search_agent/` as a self-contained subtree, wired through
`RetrieveOp` as the sole retrieval backend (the earlier catalog-based
`page_index` backend was later retired).
The first-pass merge was deliberately conservative; subsequent passes
have folded most of the agent into the framework's shared infrastructure.

### Done

- **Dual-channel step protocol.** Each step the model emits **exactly one** fenced block, and the fence language is the channel: a ` ```python ` block is a single tool call (exec'd in the `LocalPythonExecutor` with the agent's tools bound; its output becomes the next observation), and a ` ```json ` block is the final answer (parsed directly as data — never exec'd — and gated by `validate_final_answer`). The final answer is data, so it leaves as data: there is no `final_answer` tool, `FinalAnswerException`, or `is_final_answer` flag routing it back through the sandbox. A stray `final_answer(...)` call in a python block just raises `NameError` and is fed back as a misfire. If a reply carries more than one fenced block, `_parse_step` runs the **first** ` ```python ` block and appends a one-line `[notice]` nudging one block per step (an earlier multi-parallel-tool-call mode was reverted — it didn't improve quality and added complexity). `call()` holds all per-question state in locals, so one agent instance is reentrant across concurrent questions.
- **Off-loop tool execution.** Within a question, retrieve branches run as concurrent `asyncio` tasks on **one event loop in one worker thread** (one loop per question — see `eval_e2e._run_uid`). A step's tool block is therefore executed via `await asyncio.to_thread(_run_block, self._executor, code)` in `MultiTurnAgent._execute_code`: the `await` yields the loop so sibling branches keep progressing while the blocking tool I/O (embeddings + ChromaDB) runs on a thread — a blocking `ThreadPoolExecutor(...).result()` here would stall the whole question. The block runs on the agent's **persistent** executor, so cross-step interpreter state survives (which `task_solver` relies on). Tools never touch `ctx`, so off-loop execution stays thread-safe; the process-wide `embed` rate limiter bounds real API concurrency.
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

- **Agentic loops confined to retrieve and lookup_external.** Two operators run iterative tool-using LLM loops (both `MultiTurnAgent`s): retrieve (`SearchAgent` in `src/skunk/search_agent/`), and `lookup_external` (`LookupAgent`). `extract` and `compute` do not loop agentically — they execute once per call (compute's internal codegen→exec retries are a fixed bounded budget, not a tool loop), and failure is recorded in the trace. Crucially, *missing-data recovery* is a bounded replan loop in the orchestrator — each round the replanner composes a fresh `Plan` of only the branches to run now (no diff) — not a per-operator retry loop: an operator never re-plans or re-dispatches itself on failure.
- **No per-table/per-figure catalog rows.** Page-level granularity matches the benchmark's `source_docs?page=N` labels and the existing `cache/tables/` structure. Going finer adds rows without improving recall.
- **No PZ runtime dependency.** This repo is plain Python; LLM calls go through the AI Studio Gemini API (`google-genai`, authenticated by `GEMINI_API_KEY`) or OpenRouter (`SKUNK_LLM_PROVIDER=openrouter`) — not Vertex/ADC. PZ stays out of the runtime path.
