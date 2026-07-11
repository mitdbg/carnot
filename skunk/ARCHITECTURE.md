# Architecture

> **Extract removed (2026-07-11).** The `extract` operator and the
> `PageContentStore` page-store seam are gone. The pipeline is now
> **retrieve → compute**: `retrieve` runs the search agent and returns the relevant
> pages WITH their text (`RetrievedDoc`), and `compute` reads that page text directly
> (transcribing numbers into code) to produce the answer. Page text comes from the same
> `clean_page_map` corpus the search agent reads — there is no separate content store to
> inject. Sections below describing extract's tier dispatch, the page store, and
> extract-stamped `AnnotatedValue` provenance are **historical**; `AnnotatedValue` now
> carries only `lookup_external`'s external values. grc-officeqa's `page_index/` build is
> orphaned by this change (kept pending its own removal).
>
> **Layout note (2026-07-07).** The OfficeQA application layer moved out of this
> repo dir to `../grc-officeqa/` (library-hardening refactor): the Treasury corpus
> accessors are `officeqa/corpus.py`, the app config `officeqa/config.py`
> (`SkunkConfig`), and the eval harness `grc-officeqa/eval/`. Path references to
> `eval/` and `corpus.py` below refer to their grc-officeqa homes.

## Why this design

OfficeQA questions ask about specific tables and figures inside a corpus of 696 monthly U.S. Treasury Bulletin PDFs (1939–2025). A naive date-driven retrieval agent — "the question mentions 1940, look in the 1940 bulletin" — fails on **81%** of the benchmark, because the answer-bearing table for period *P* often lives in a *later* bulletin (publication lag) or in a *retrospective summary* in a single mid-year issue. The median miss is +2 months; the long tail goes out to +9 years.

The fix: retrieve at the **page** level, indexed by **what each page reports**, not by when its bulletin was published. Retrieval names pages and returns their text; `compute` reads that text to answer.

This decoupling means retrieval and answering can be evaluated independently — both labels exist in the benchmark (`source_docs?page=N` for retrieval, `answer` for the final answer) — so we can attribute errors cleanly.

## High-level flow

```
                    ┌────────────┐
   question ───────▶│  planner   │  one LLM call
                    │   (DSL)    │
                    └─────┬──────┘
                          │  Plan (branches list)
                          ▼
                    ┌────────────┐
                    │orchestrator│  parallel branches
                    └─────┬──────┘
                          │
       ┌──────────────────┴──────────────────┐
       │                                     │
       ▼                                     ▼
  ┌─────────┐                          ┌─────────────┐
  │retrieve │  search agent →          │lookup_extern│   per-op operators
  │  agent  │  pages + text            │   gemini    │
  └─────────┘                          └─────────────┘
       │                                     │
       ▼                                     ▼
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

`run_retrieve_all` in `src/skunk/retrieve.py` is the **single retrieval seam**: it is the
only layer that inspects `ctx.config.golden_pages`, and it normalizes retrieval to one
per-branch result — `BranchRetrieval(documents)` (`src/skunk/common.py`), the retrieved
pages with their text that compute reads. A branch that fails surfaces as a per-branch
`StepFailed` in the same list, so one branch failing doesn't sink its siblings. Dispatch:

- **`search_agent`** (the only backend) — the iterative ChromaDB + LLM-loop retriever under
  `src/skunk/search_agent/` (`SearchAgent`, a `MultiTurnAgent` with vector-search / grep /
  read-document / prune tools over the chunked corpus, plus an opt-in `semantic_filter`
  LLM-judge tool callers can wire via `extra_tools`). Branches run in parallel, a fresh
  agent each, with per-branch failure isolation; the agent's page-key output is parsed to
  `PageRef`s.
- **golden bypass** — `config.golden_pages` (the eval harness's `--golden` ablation) injects
  the benchmark's gold pages verbatim and skips the agent entirely.

**Corpus data contract — "document" = retrieval unit.** The SearchAgent's vocabulary has two
levels, both defined by the Chroma collection it is handed (schema in
`search_agent/search_tools.py`'s module docstring; per-benchmark taxonomy in qatfd's
`CORPUS_MODEL.md`): a **chunk** (one Chroma row = one embedded element; row id = `chunk_id`,
ordered within its document by the int `element_id`) and a **document** (`doc_id` — the
RETRIEVAL UNIT: what `read_document` fetches whole via `document_map`, what prune/seen state
tracks, and what the final `{"doc_ids": [...]}` answer returns). The unit is chosen per corpus
at index-build time and is often NOT a whole source file: for the Treasury corpus it is a
single PDF page (`doc_id` "1946_11_41" = bulletin 1946-11, page 41 — which is why
`retrieve.py` parses the agent's doc_ids straight into `PageRef`s), while other corpora use a
web page, an abstract, a Wikipedia article, or a source file. Note the operator pipeline's
`AnnotatedValue.source_stem` (extract provenance) is the source document's filename stem — a
different, coarser identity than the search layer's `doc_id`, which is why it is not called
doc_id.

(*Historical:* an earlier catalog-based `page_index` retriever and, later, a
`PageContentStore` page-store seam that fed the `extract` operator both existed here.
Retrieval is now the search agent alone, and the extract/page-store layer is removed —
`retrieve` returns each page's cleaned text directly from `clean_page_map`, which `compute`
reads. grc-officeqa's `page_index/` build survives but is orphaned.)


## The operators

- **`retrieve(key, period)`** — chain head. Runs the search agent for the branch and returns its retrieved pages WITH their text, as `RetrievedDoc`s (`ref: PageRef`, `text: str`). Dispatched by `run_retrieve_all` to the search-agent backend (or short-circuited by `ctx.config.golden_pages`, which attaches the golden pages' text from `clean_page_map`) — see "Retrieval: the search agent" above. The page text is the same cleaned per-page corpus the search agent itself reads; there is no separate extract/read step.
- **`lookup_external(nl)`** — chain-head capable. Runs a `LookupAgent` (a `MultiTurnAgent` tool loop: FRED / BLS / World Bank / Tavily / fetch_url, ≤`lookup_max_steps` steps) over a natural-language description of external factual data (`nl`); the agent commits a dict of `AnnotatedValue` fields, which `LookupExternalOp` parses into `list[AnnotatedValue]` (one entry) on the trusted side. Use for CPI-U, FX rates, event dates, named entities (bureau names), and any fact not in the bulletin corpus. The agent infers the appropriate `kind` and `unit` (including `text` for strings). `AnnotatedValue` now exists solely for these external values.
- **`compute()`** — chain terminator that subsumes formatting. Reads `ctx.question`, the retrieved pages' text (as plain document context — the model transcribes numbers it needs into code literals), and any `lookup_external` values (`input_values`, present as `AnnotatedValue` frames in the exec env). Runs a single loop (`ComputeOp.run`) of codegen → in-process exec under one shared budget (`compute_max_attempts`, default 3). Each iteration's codegen call emits one of two `python` forms: **(a)** assign `result` (success); or **(b)** *missing data* — assign `missing` (the identifiers + a one-line reason). Parse, exec, and malformed-block failures feed the next iteration's `prev_code` + `prev_failure`. `run()` returns a union: `Final(answer)` on a clean exec that set `result`, or `NeedsMore(missing)` — it never raises `MissingData`. Fails with `StepFailed("compute", …)` when no iteration ever resolves. Across `compute_best_of_n` trials, `_vote` returns the most-frequent outcome with all `NeedsMore`s pooled into a single "missing-data" candidate (a tie never breaks in favor of missing-data); when missing-data wins, the trials' `missing` identifiers are unioned and their reasons rendered per agent ("agent 1: …, agent 2: …").

  Recovery state is a **pool** of everything gathered so far — retrieved pages (`RetrievedDoc`) plus any lookup values (`AnnotatedValue`). On each `NeedsMore` the whole pool is retained (there is no per-value `keep`/drop), and each replan sweep appends its newly gathered entries. Each recovery round (≤ `recovery_max_rounds`), `Planner.replan` composes a **fresh `Plan`** containing only the branches to run now — no diff, no index alignment; branch identity dies at the end of its sweep. The replan message carries the pool, the **previous plans** (every branch tried with its fate + failure diagnostics), and compute's missing description, and the replanner is instructed to diagnose why the gathering so far did not satisfy compute before emitting branches — never re-emitting an approach that already failed the same way. A replan sweep that yields zero new entries **short-circuits**: compute is skipped (it would be a no-op on an unchanged pool) and the prior `NeedsMore` drives the next replan, still consuming a round. When the budget is exhausted, `execute()` raises `MissingData` — its only remaining role, the terminal signal the eval harness records.

## Plan shape

A **Plan** is a pydantic model: just a `branches` list. Compute reads everything it needs — the calculation to perform *and* the output format (units, precision, list shape) — from the verbatim question, not from a planner paraphrase, so the plan carries no separate requirements object. The Python layout mirrors the wire JSON exactly, so `Plan.model_validate_json` / `Plan.model_dump_json` round-trip with no custom translation. There is no multi-step decomposition — every plan is exactly one terminal compute. `PlannerPromptedCall` (in `src/skunk/plan.py`) emits a `Plan`; the orchestrator runs branches in parallel, then feeds the merged pool (retrieved pages + lookup values) to compute. Concretely it runs all `retrieve` branches through the retrieve phase (`Orchestrator._run_retrieve_phase` → `run_retrieve_all`, one search-agent rollout per branch, each returning that branch's pages + text) — every `lookup_external` runs in parallel alongside it; results stay aligned with `plan.branches` by position.

**Both branch kinds are first-class on the initial plan; the choice is per value.** The planner is instructed (advisorily — there is no parse-hook enforcement) to prefer a `retrieve` branch whenever the data could reasonably be expected to exist in the corpus, and to emit a `lookup_external` branch for a value unlikely to be in the corpus or when the question names a clear external source — rather than defaulting everything to retrieve. At replan, `lookup_external` is unrestricted. There are no verbatim checks on retrieve `key`s or lookup `src`s — both are free natural language.

Two branch shapes:

- `RetrieveBranch(key, period)` — one search-agent rollout, dispatched by the orchestrator, returning the branch's pages + text. There is no issue/vintage pin field: issue choice (which print/reprint to read) belongs entirely to selection.
- `LookupBranch(target, src=None)` — single `lookup_external` call. `src` is an optional publisher hint, set when the question names a clear external source.

Canonical JSON shape (one branch):

```json
{
  "branches": [
    {"kind": "retrieve", "key": "national defense expenditures", "period": "CY1940"}
  ]
}
```

Per-branch field semantics (`key` / `period`, when to use `null`, etc.) live in `PlannerPromptedCall.system_prompt` in `src/skunk/plan.py` — that prompt is the canonical schema spec. Runtime contracts (`PageRef`, `RetrievedDoc`, `AnnotatedValue`) live in the dataclasses in `common.py`.

The `AnnotatedValue.kind` taxonomy:

- `scalar` — `value` is `int | float | str`. Single facts.
- `vector` — `value` is `{index_label: scalar}`; `index_name` names the varying dim.
- `table` — `value` is `{row_label: {col_label: scalar}}`; `row_name` / `col_name` name the two dims.

Cells must be primitive scalars (`int` / `float` / `str`, no `bool`, no nesting beyond these shapes). The `AnnotatedValue` validator (`_check_shape` in `common.py`) rejects deeper structures at construction.

`AnnotatedValue` now carries only `lookup_external`'s external values, so its provenance fields (`source_stem` / `pages` / `requested_period` / `retrieve_key` / `bulletin`, all default `None`/`()`) are largely vestigial — the lookup agent fills `description` / `unit` / `source` and leaves the machine-stamped corpus fields empty. (*Historical:* these were stamped by extract from the source `PageRef`s + `RetrieveBranch`; extract is gone.) Lookup values surface in the compute/replanner input view via `input_values_desc` (`common.py`), which renders each entry's full frame cell-by-cell (values included, so generated code sees NaN/"n/a" cells and exact magnitudes), not just its schema. Retrieved pages surface via `documents_desc` — plain page text the model reads directly.

## How compute reads pages

Retrieve names specific pages and returns each one's cleaned text (`RetrievedDoc.text`, from `clean_page_map`). `compute` receives that text as plain document context in its prompt (`documents_desc`) and transcribes the numbers it needs into code literals — the pages are text, not variables in the exec env (only `lookup_external`'s `input_values` enter the env as frames). There is no cross-page search and no vision tier: if retrieve picked the wrong pages, the bug is in retrieve, not compute. This is what makes the eval decomposition work.

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

On-disk artifacts in the runtime path:

- The search-agent corpus — a ChromaDB collection (`chromadb_dir` / `chromadb_collection`) plus `clean_page_map.json` (`doc_id → cleaned per-page text`). `retrieve` loads the page text from `clean_page_map` and connects to the ChromaDB server; both are built offline by the app (grc-officeqa's `prep/` + qatfd's `create_vector_db.py`).
- (*Historical:* grc-officeqa's `page_index/` artifact — `catalog/*.jsonl`, `concept_tree.json`, the page store, PNG renders — fed the retired extract/page-store layer and is no longer in the runtime path.)

LLM completions are **not** cached. There is no DSL plan cache — the planner runs once per question. There is no retrieval cache; `eval/eval_e2e.py --golden` is the only retrieve bypass (it injects the benchmark's gold pages, with their text, to measure the compute ceiling given perfect retrieval).

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

- **Agentic loops confined to retrieve and lookup_external.** Two operators run iterative tool-using LLM loops (both `MultiTurnAgent`s): retrieve (`SearchAgent` in `src/skunk/search_agent/`), and `lookup_external` (`LookupAgent`). `compute` does not loop agentically — it executes once per call (its internal codegen→exec retries are a fixed bounded budget, not a tool loop), and failure is recorded in the trace. Crucially, *missing-data recovery* is a bounded replan loop in the orchestrator — each round the replanner composes a fresh `Plan` of only the branches to run now (no diff) — not a per-operator retry loop: an operator never re-plans or re-dispatches itself on failure.
- **No per-table/per-figure catalog rows.** Page-level granularity matches the benchmark's `source_docs?page=N` labels and the existing `cache/tables/` structure. Going finer adds rows without improving recall.
- **No PZ runtime dependency.** This repo is plain Python; LLM calls go through the AI Studio Gemini API (`google-genai`, authenticated by `GEMINI_API_KEY`) or OpenRouter (`SKUNK_LLM_PROVIDER=openrouter`) — not Vertex/ADC. PZ stays out of the runtime path.
