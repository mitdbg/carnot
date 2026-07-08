# CLAUDE.md

Operational guide for Claude Code sessions on this repo. For how the system actually
works — architecture, design intent, the plan-shape spec — read `ARCHITECTURE.md`.

## What this is

OfficeQA — a declarative QA pipeline over the U.S. Treasury Bulletin corpus (696 monthly
PDFs, 1939–2025). Questions are answered by composing 4 operators
(`retrieve` / `extract` / `lookup_external` / `compute`) into a typed DSL plan that an
orchestrator walks. Benchmark: `data/officeqa_pro.csv` (133 questions = 33 dev + 100 test;
not tracked in git, keep locally).

## 🚨 Held-out test split — READ BEFORE ANY EVAL RUN 🚨

**The canonical OfficeQA dev/test split is owned by qatfd** (the sibling evaluation
harness where systems are measured against benchmarks):
`../qatfd/benchmarks/officeqa/officeqa_splits.json` — `{"dev": [33 uids], "test":
[100 uids]}`, dev = the first 25% of CSV order, generated deterministically by
`qatfd/scripts/make_splits.py`. **Do not run on, tune against, or inspect traces of
the 100 test UIDs during development** — skunk's prompts are shared with qatfd
systems, so contamination here biases qatfd's reported numbers too.

- Every eval harness MUST load the splits file and filter the test UIDs out by
  default. `--include-test-set` overrides only for a deliberate final-number run.
- Writing a new eval script: copy the filter from `eval/eval_e2e.py` and wire it in
  BEFORE the first LLM call. `eval_e2e` ABORTs (not warns) when the splits file is
  missing or `--uids` names a test UID.
- Reporting numbers: always state dev-only (33) vs full (133). Full-set numbers are
  contaminated for any generalization claim.

History: skunk's competition-era scheme (32 held-out UIDs in
`eval/test_set_uids.json` + 70-UID dev list) was retired 2026-07-07 in favor of the
qatfd split; the old lists live in git history. The two schemes are incompatible
(the old dev70 overlaps the new test-100 by 41 UIDs), so numbers from the two
regimes must never be compared directly.

## Running experiments

- **Don't autonomously launch full dev sweeps.** Implement, sanity-check on a couple of
  dev UIDs, then hand the run command to the user.
- Harness is `eval/eval_e2e.py` (`--dev-set` runs the 33-UID qatfd dev split). State
  dev-only vs full in any reported number (see above).
- The default `search_agent` retriever needs `cache/chromadb/` + `cache/clean_page_map.json`
  built offline (`src/skunk/search_agent/prep/`); the first non-golden run errors clearly
  if either is missing.

## Conventions

- **Use `python3` / `pip3`, not `python` / `pip`** — no `python` on PATH (exit 127). In a
  batched/parallel tool call, one `python: command not found` aborts the whole batch.
- **No new ops** without updating `ARCHITECTURE.md`, the planner system prompt in
  `src/skunk/plan.py`, the validator, AND `eval/eval_e2e.py`.
- **Logging**: one event stream via `ctx.emit(...)`; full rules in `ARCHITECTURE.md` →
  "Logging & observability".
- **No formatting-only edits.** Do not reformat lines for length, change quote style
  (single → double), add blank lines between methods, or expand/collapse argument lists
  unless it is part of a substantive change to that line. Formatting-only diffs pollute
  `git diff`, make code review harder, and cause merge conflicts.

## Setup (.env at repo root)

LLM generation now defaults to **OpenRouter** (`SKUNK_LLM_PROVIDER=openrouter`,
authenticated by `OPENROUTER_API_KEY`) — we no longer have an AI Studio Gemini key, so
the legacy `genai` path is unavailable. In OpenRouter mode `SKUNK_LLM_MODEL` and every
model override (`SKUNK_MODEL_OVERRIDES`, the qatfd `agent_model_id`) must be **full
OpenRouter ids**, e.g. `google/gemini-2.5-flash` or `qwen/qwen-2.5-72b-instruct`.
Strongly recommended for `lookup_external`: `FRED_API_KEY`, `TAVILY_API_KEY`. Copy
`.env.example` → `.env`.

Embeddings are unaffected by the provider (they dispatch on the embedding model id). A run
uses a single provider for all generation; transient errors (429 / 5xx / transport blips)
are retried with exponential backoff (`llm_client._retry_call`). NOTE: OpenRouter wraps an
upstream provider failure as a 4xx `...ResponseError: Provider returned error`, which is
*not* retried — `llm_client._error_detail` logs the status + provider metadata + raw body
alongside the warning so these are diagnosable.

The legacy **AI Studio Gemini** path (`SKUNK_LLM_PROVIDER=genai`, `GEMINI_API_KEY`, bare
model ids like `gemini-3.5-flash`) still exists in code but is dormant until a key is
restored.

## Data corpus (not in this repo)

Treasury Bulletin PDFs at `~/Desktop/officeqa/treasury_bulletin_pdfs/` (~20 GB, 696 files);
override with `OFFICEQA_PDF_DIR`. Pre-extracted tables and rendered pages live under
`cache/` (gitignored), populated on first use.

## Pointers

- `ARCHITECTURE.md` — architecture, design intent, plan-shape spec
- `src/skunk/plan.py` — canonical Plan dataclasses + the planner system prompt (plan JSON schema)
- `data/officeqa_pro.csv` — benchmark (not in git)
