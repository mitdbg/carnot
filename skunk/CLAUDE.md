# CLAUDE.md

Guide for Claude Code sessions on this repo.

## What this is

OfficeQA — a declarative QA pipeline over the U.S. Treasury Bulletin corpus (696 monthly PDFs, 1939–2025). Questions are answered by composing 4 operators into a typed DSL plan; an orchestrator walks the plan and dispatches subagents.

The benchmark is `data/officeqa_pro.csv` (133 questions; not tracked in git, keep locally). An initial pass of DSL plan for each question is in `data/dsl_planning_pass.csv`.

## Architecture in one screen

- Question → planner emits DSL plan (text) → orchestrator walks AST → 4 subagents.
- 4 ops: `retrieve`, `extract`, `lookup_external`, `compute`. `compute` is the chain terminator and subsumes formatting (it self-plans, codegens, execs, then self-critiques the result against the question with full context — same domain prompt as the producer, no info asymmetry). `extract` accepts `visual_only=True` to skip text/OCR tiers for charts and figures.
- Extract emits one of three **kinds** per entry: `scalar`, `vector` (1-D series indexed by one dim), or `table` (2-D grid). Vector/table cells are always primitive scalars — nesting is forbidden and enforced in the extract parser. See `DSL.md` for the contract.
- Retrieval is **page-level**: the retrieve subagent maintains its own page index over the corpus. `PageRef.page` is the **1-based PDF page index** — the only page-number convention used in the codebase. The bulletin's printed-page footer is recoverable via `extract.get_printed_page` for trace/prompt enrichment.
- Extract is per-page tier dispatch: CSV tables → OCR text → vision render. No cross-page search.
- See `ARCHITECTURE.md` for design intent, `DSL.md` for grammar.

## Layout

- `src/skunk/dsl.py` — DSL parser, serializer, validator, AST types
- `src/skunk/planner.py` — LLM planner: question → DSL AST
- `src/skunk/orchestrator.py` — AST executor; dispatches subagent functions
- `src/skunk/subagents/` — one module per op (`retrieve`, `extract`, `lookup_external`, `compute`); each exposes a bare `run(op, prev, ctx)` function
- `src/skunk/subagents/base.py` — shared utilities: `StepFailed`, `MissingData`, `exec_python`, `exec_python_with_env`, `parse_llm_value`
- `src/skunk/common.py` — shared runtime: `LLMClient`, `HarnessContext`, `LLMResponse`
- `eval/` — evaluation harness: `eval_e2e.py`

## Conventions

- DSL string args use **single quotes**, always (JSON-safe). See `DSL.md` quoting rule.
- No new ops without updating: `DSL.md`, the planner few-shots, the validator, AND the eval harnesses.
- **No Palimpzest imports.** This repo is intentionally PZ-free at runtime. Annotation tooling at the OfficeQA-in-PZ stage is a separate project.
- **No hand-rolled retry loops** inside subagents. The orchestrator records failures in the trace; it does not re-plan.

## API keys (.env at repo root)

- `GEMINI_API_KEY` — Gemini 2.5 Flash; used by the planner and all subagents via `ctx.llm_client.call(...)` in `src/skunk/common.py`

`.env.example` is committed; copy it to `.env` and fill in keys.

## Data corpus (NOT in this repo)

- Treasury Bulletin PDFs live at `~/Desktop/officeqa/treasury_bulletin_pdfs/` (~20 GB, 696 files).
- Path is configurable via `OFFICEQA_PDF_DIR` env var; default is the path above.
- Pre-extracted tables and rendered pages live under `cache/` (gitignored), populated on first use.

## Pointers

- `ARCHITECTURE.md` — design intent, no iteration history
- `DSL.md` — formal grammar, type system, worked examples
- `data/dsl_planning_pass.csv` — 133 validated plans; used at runtime via `--cached-plan` to skip the LLM planner
- `data/officeqa_pro.csv` — benchmark (133 questions, with `source_docs?page=N` and `answer` golden truth; not tracked in git)

## Things to verify before claiming a feature is "done"

- DSL changes: `pytest tests/test_dsl_roundtrip.py` passes.
- Subagent changes: the standalone callable still has its signature; the eval harness for that subagent passes its acceptance bar (see the acceptance criteria in the subagent's TODO comment).
- New ops or new args: documented in `DSL.md`, and the planner few-shots in `src/skunk/planner.py` are updated.
