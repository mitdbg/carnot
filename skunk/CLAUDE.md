# CLAUDE.md

Operational guide for Claude Code sessions on this repo. For how the system works —
architecture, design intent, the plan-shape spec — read `ARCHITECTURE.md`.

## What this is

**skunk** — a library for agentic retrieval and computation over document corpora.
Two layers:

1. **Agent/LLM core** (what qatfd consumes): `llm_client`, `multi_turn_agent`,
   `search_agent`, `prompted_call`, `chroma_client`, `trace`, `usage`, `errors`,
   `common.ExecutionContext`, the `SystemConfig` → `SearchAgentConfig` →
   `PipelineConfig` config hierarchy.
2. **Operator pipeline**: question → `Planner` → typed DSL `Plan` → `Orchestrator`
   → {`retrieve`, `extract`, `lookup_external`, `compute`} with data-prep pooling
   and replan-on-MissingData recovery. Corpus content reaches `extract` only
   through the `skunk.page_store.PageContentStore` protocol, injected via
   `Orchestrator(page_store=...)`.

Applications live OUTSIDE this repo dir (the library has no benchmark, corpus
paths, or eval harness of its own):

- `../grc-officeqa/` — the OfficeQA / Grounded Reasoning Cup app (Treasury corpus
  accessors, page-index build, `SkunkConfig.from_env`, eval harness). Its
  CLAUDE.md carries the 🚨 held-out-split rules for any OfficeQA eval run.
- `../qatfd/` — the paper's system×benchmark evaluation harness.
- `../datagen/` — RL training-data synthesis (tinker).

The refactor that produced this layout is `REFACTOR_PLAN.md` (2026-07-07);
deleted subtrees (`harness_ui/`, HITL, `dais/`) live in git history.

## Commands

```bash
venv/bin/python3 -m pytest tests/   # unit tests
venv/bin/ruff check src/ tests/     # lint (clean at HEAD)
```

## Conventions

- **Use `python3` / `pip3`, not `python` / `pip`** — no `python` on PATH (exit 127). In a
  batched/parallel tool call, one `python: command not found` aborts the whole batch.
- The active interpreter is `venv/bin/python3` (NOT `.venv/`); skunk is installed
  editable there.
- **No new ops** without updating `ARCHITECTURE.md`, the planner system prompt in
  `src/skunk/plan.py`, the validator, AND the app-layer eval harness
  (`../grc-officeqa/eval/eval_e2e.py`).
- **Logging**: one event stream via `ctx.emit(...)`; full rules in `ARCHITECTURE.md` →
  "Logging & observability". The post-hoc viewer is `scripts/trace_viewer/`.
- **No formatting-only edits.** Do not reformat lines for length, change quote style
  (single → double), add blank lines between methods, or expand/collapse argument lists
  unless it is part of a substantive change to that line. Formatting-only diffs pollute
  `git diff`, make code review harder, and cause merge conflicts.
- Don't autonomously launch full eval sweeps (they cost real LLM spend); sanity-check
  on a couple of items and hand the run command to the user.

## Setup (.env at repo root)

LLM generation defaults to **OpenRouter** (`SKUNK_LLM_PROVIDER=openrouter`,
authenticated by `OPENROUTER_API_KEY`) — we no longer have an AI Studio Gemini key, so
the legacy `genai` path is unavailable. In OpenRouter mode `SKUNK_LLM_MODEL` and every
model override must be **full OpenRouter ids**, e.g. `google/gemini-2.5-flash`.
Copy `.env.example` → `.env`. Transient errors (429 / 5xx / transport blips) are
retried with exponential backoff (`llm_client._retry_call`). NOTE: OpenRouter wraps an
upstream provider failure as a 4xx `...ResponseError: Provider returned error`, which is
*not* retried — `llm_client._error_detail` logs the status + provider metadata + raw body
alongside the warning so these are diagnosable.

The legacy **AI Studio Gemini** path (`SKUNK_LLM_PROVIDER=genai`, `GEMINI_API_KEY`, bare
model ids like `gemini-3.5-flash`) still exists in code but is dormant until a key is
restored.

## Pointers

- `ARCHITECTURE.md` — architecture, design intent, plan-shape spec
- `REFACTOR_PLAN.md` — the library-hardening refactor plan (phases, decisions)
- `src/skunk/plan.py` — canonical Plan dataclasses + the planner system prompt
- `src/skunk/page_store.py` — the corpus-content protocol apps implement
