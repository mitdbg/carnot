# CLAUDE.md

Guide for Claude Code sessions on this repo.

## 🚨 BEFORE RUNNING ANY EVALUATION SCRIPT — READ THIS FIRST 🚨

**32 of the 133 dev UIDs are reserved as a HELD-OUT TEST SET** (see
`eval/test_set_uids.json`). Tuning prompts/code against them, inspecting
their traces, or including them in any benchmark run counts as
contamination. The remaining **101 UIDs are the dev set**.

Before running ANY `--n N`, `--uids UIDxxxx,...`, or "all-questions" sweep:

1. **Verify the eval script excludes test UIDs by default.** Every eval
   harness in this repo (`eval/eval_e2e.py`, any new one) MUST load
   `eval/test_set_uids.json` and filter them out unless
   `--include-test-set` is explicitly passed.
2. **If you're writing a new eval script**, add the test-set filter
   BEFORE running anything. Don't sample-then-filter — that wastes calls.
3. **If you're invoking with `--uids`**, intersect the requested UIDs
   with `eval/test_set_uids.json["uids"]` and ABORT (not warn) if any
   match. There is no legitimate reason to point at test UIDs from a
   command line unless `--include-test-set` is set.
4. **When reporting numbers**, always state whether they're on dev
   (101 UIDs) or include the test set (133 UIDs). Numbers on the full
   133 set are contaminated for any claim about generalization.

If you find a script that doesn't enforce this filter, **fix the script
before running** — otherwise everything you tune downstream is biased.

See "Held-out test set" section below for the canonical list and seed.

## What this is

OfficeQA — a declarative QA pipeline over the U.S. Treasury Bulletin corpus (696 monthly PDFs, 1939–2025). Questions are answered by composing 4 operators into a typed DSL plan; an orchestrator walks the plan and dispatches each operator.

The benchmark is `data/officeqa_pro.csv` (133 questions: **101 dev + 32 test**; not tracked in git, keep locally).

## Architecture in one screen

- Question → planner emits DSL plan (text) → orchestrator walks AST → 4 operators.
- 4 ops: `retrieve`, `extract`, `lookup_external`, `compute`. `compute` is the chain terminator and subsumes formatting (it self-plans, codegens, execs, then self-critiques the result against the question with full context — same domain prompt as the producer, no info asymmetry). `extract` accepts `visual_only=True` to skip the parsed-text tier for charts and figures.
- Extract emits one of three **kinds** per entry: `scalar`, `vector` (1-D series indexed by one dim), or `table` (2-D grid). Vector/table cells are always primitive scalars — nesting is forbidden and enforced in the extract parser. See the "Plan shape" section in `ARCHITECTURE.md` for the contract.
- Retrieval is **page-level**: `PageRef.page` is the **1-based PDF page index** — the only page-number convention used in the codebase. `RetrieveExecutor` in `src/skunk/retrieve.py` is a dispatcher: golden bypass first (when `ctx.config.golden_pages` is set), then routes to one of two backends per `config.retriever`: `"search_agent"` (default — the iterative ChromaDB + LLM-loop retriever vendored under `src/skunk/search_agent/`, returns page keys that are mapped to `PageRef` via `models.page_key_to_pageref`) or `"page_index"` (the legacy chapter-pick + year-filter retriever preserved at `src/skunk/page_index/retrieve_prototype.py:PageIndexRetrievePrototype` for ablations). The search agent needs `cache/chromadb/` + `cache/clean_page_map.json` built offline (see `src/skunk/search_agent/prep/`); first non-golden call fails fast with a clear error if either is missing.
- Extract is per-page tier dispatch: parsed-table text → vision render. No cross-page search.
- See `ARCHITECTURE.md` for design intent + plan shape; the canonical JSON-schema spec for plans lives in `PlannerPromptedCall.system_prompt` in `src/skunk/plan.py`.

## Layout

The agent has no CLI of its own. Public API is `from skunk import Orchestrator, HarnessContext, SkunkConfig, Plan, PageRef, ...` (see `src/skunk/__init__.py`); CLI/UX layers (eval harnesses, future chat UI) live outside `skunk/` and build on top.

- `src/skunk/__init__.py` — public API re-exports
- `src/skunk/plan.py` — `Plan` dataclass (+ `Computation`, `Presentation`, `RetrieveBranch`, `LookupBranch`) + JSON serde + validator + `PlannerPromptedCall` (question → Plan)
- `src/skunk/models.py` — cross-cutting runtime types: `PageRef`, `AnnotatedValue` (with `.frame` pandas accessor), `HarnessContext`
- `src/skunk/orchestrator.py` — `Orchestrator(ctx).execute()` — Plan executor; instantiates one operator-level class per op (`RetrieveExecutor` / `ExtractExecutor` / `ComputeExecutor`, plus `LookupExternalPromptedCall` which doubles as both operator and call-site since `lookup_external` is a single LLM call) and dispatches through their `.run()` methods
- `src/skunk/{retrieve,extract,lookup_external,compute}.py` — one module per operator; each exposes an operator-level class with a `run(prev, ctx, **kwargs)` method (call-site `PromptedCall` subclasses live alongside as instance attributes on the operator class)
- `src/skunk/errors.py` — cross-module signals: `StepFailed`, `MissingData`
- `src/skunk/pyexec.py` — in-process Python exec for operator-generated code (not a security boundary; real isolation is future work): `exec_python_with_env`, `strip_code_fences`
- `src/skunk/prompted_call.py` — `PromptedCall` base: prompt-assembly for every LLM-prompted call-site inside an operator (planner, extract.text/vision/dedup, compute.codegen/critique, lookup_external)
- `src/skunk/common.py` — shared runtime: `LLMClient` (OpenRouter + direct-Gemini), `LLMResponse`, rate-limit + retry helpers
- `src/skunk/config.py` — `SkunkConfig` (model, RPM, retry, extract/compute knobs)
- `src/skunk/prompt_overrides.py` — `PromptOverride` YAML loader for corpus / few-shots / lessons
- `src/skunk/page_index/` — page-index builder (`pipeline.py`) + runtime helpers (`retrieve_probe`, `period`, `pdf`) + `retrieve_prototype.py` (legacy retrieve operator preserved as `PageIndexRetrievePrototype`)
- `src/skunk/search_agent/` — teammate's iterative search agent merged from `refs/heads/skunk`. Self-contained subtree: `search_agent.py` (the agent), `search_tools.py` (`vector_search` / `retrieve_page_info` / `run_grep` / `final_answer`), `prompts.yaml`, `tracer.py`, `utils/local_python_executor.py` (smolagents fork) + `utils/parsing.py`, `openrouter_client.py` (local shim around the `openai` SDK — stands in for the unknown `openrouter` PyPI package the teammate imported), `prep/` (offline corpus prep: `page_cleaner.py`, `create_vector_db.py`, `compute_element_embeddings.py`). See ARCHITECTURE.md "TODO after merge" for the planned utility unification (LLM client, code executor, tracer).
- `eval/` — `eval_e2e.py` (end-to-end harness); `util.py` for plan-cache + trace-dump helpers; `test_set_uids.json` for the held-out filter

## Conventions

- No new ops without updating: `ARCHITECTURE.md`, the planner system prompt in `src/skunk/plan.py`, the validator, AND the eval harnesses.
- **No Palimpzest imports.** This repo is intentionally PZ-free at runtime. Annotation tooling at the OfficeQA-in-PZ stage is a separate project.
- **No hand-rolled retry loops** inside operators. The orchestrator records failures in the trace; it does not re-plan.

## API keys (.env at repo root)

Two LLM paths (see `src/skunk/common.py` docstring):

- `OPENROUTER_API_KEY` — default path for every non-search call (planner, extract, compute, codegen, critique). Model is `config.llm_model`, default `google/gemini-3-flash-preview`.
- `GEMINI_API_KEY` — direct Gemini API; only used by `lookup_external` when `use_google_search=True` for native Google Search grounding. Model is `config.gemini_model`, default `gemini-3-flash-preview`.

Optional: `FRED_API_KEY` for the FRED tier of `lookup_external`. `.env.example` is committed; copy to `.env` and fill in.

## Data corpus (NOT in this repo)

- Treasury Bulletin PDFs live at `~/Desktop/officeqa/treasury_bulletin_pdfs/` (~20 GB, 696 files).
- Path is configurable via `OFFICEQA_PDF_DIR` env var; default is the path above.
- Pre-extracted tables and rendered pages live under `cache/` (gitignored), populated on first use.

## Pointers

- `ARCHITECTURE.md` — design intent + plan-shape spec; no iteration history
- `src/skunk/plan.py` — canonical Plan dataclasses + `PlannerPromptedCall.system_prompt` (the JSON-schema spec for plans)
- `data/officeqa_pro.csv` — benchmark (133 questions, with `source_docs?page=N` and `answer` golden truth; not tracked in git)

## Held-out test set (CRITICAL — do not run on these during development)

32 UIDs are reserved as a held-out test set. They were picked 2026-05-14 by random
sample (seed 20260514) from the UIDs we had **never inspected** at that point —
i.e. excluding the pre-existing curated 32, the 33 baseline-run problematic UIDs
(failures + >5% wrongs), and the 5 moderate-error UIDs whose traces we examined
in the failure catalog.

**Rule**: development runs of ANY eval harness (`eval/eval_e2e.py`,
any new script under `eval/`) must NOT touch
these UIDs. Looking at their traces, tuning prompts against them, or
selecting them by `--uids` counts as contamination. They exist to give
us a clean measurement of generalization when we want to publish a
final number.

The canonical list lives in `eval/test_set_uids.json` (machine-readable, with seed
and rationale). **Every eval harness MUST load this file and filter out
the test UIDs by default.** Pass `--include-test-set` to override only
when you have a deliberate reason (e.g. running the final number for a
writeup). If you add a new eval script, copy the filter logic from
`eval/eval_e2e.py`.

Verify the filter is wired before you run: search the script for
`test_set_uids`. If absent, ADD it before running — don't run first
and filter later. (See the 🚨 banner at the top of this file.)

Test UIDs (32):
```
UID0035, UID0036, UID0039, UID0050, UID0065, UID0068, UID0073, UID0093,
UID0094, UID0100, UID0108, UID0118, UID0120, UID0134, UID0147, UID0161,
UID0168, UID0170, UID0179, UID0182, UID0183, UID0187, UID0196, UID0204,
UID0211, UID0212, UID0216, UID0218, UID0227, UID0230, UID0238, UID0240
```

If you need to expand the test set later, do it by re-sampling — never by adding
UIDs the model has already been tuned against.

## Things to verify before claiming a feature is "done"

- Operator changes: the standalone `run(op, prev, ctx)` callable still has its signature; `eval/eval_e2e.py` still passes.
- New ops or new args: documented in the "Plan shape" section of `ARCHITECTURE.md`, and the planner system prompt in `src/skunk/plan.py` is updated.
- **Eval/benchmark numbers**: confirmed the run was on **dev only** (101
  UIDs after filtering `eval/test_set_uids.json`). Numbers on the full
  133-UID set are contaminated; state explicitly in any writeup whether
  the held-out 32 were included.
- **New eval scripts**: implement the test-set filter before any LLM
  call. Search the script for `test_set_uids` — if missing, it WILL
  contaminate.
