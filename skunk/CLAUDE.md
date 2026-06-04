# CLAUDE.md

Operational guide for Claude Code sessions on this repo. For how the system actually
works — architecture, design intent, the plan-shape spec — read `ARCHITECTURE.md`.

## What this is

OfficeQA — a declarative QA pipeline over the U.S. Treasury Bulletin corpus (696 monthly
PDFs, 1939–2025). Questions are answered by composing 4 operators
(`retrieve` / `extract` / `lookup_external` / `compute`) into a typed DSL plan that an
orchestrator walks. Benchmark: `data/officeqa_pro.csv` (133 questions = 101 dev + 32 test;
not tracked in git, keep locally).

## 🚨 Held-out test set — READ BEFORE ANY EVAL RUN 🚨

**32 of the 133 benchmark UIDs are a HELD-OUT TEST SET — do not run on these during
development.** The canonical list (plus the sampling seed + rationale) is
`eval/test_set_uids.json`; the other **101 are dev**. Tuning prompts/code against the 32,
inspecting their traces, or including them in a run is contamination that biases
everything downstream.

- Every eval harness MUST load `eval/test_set_uids.json` and filter the 32 out by
  default. `--include-test-set` overrides only for a deliberate final-number run.
- Writing a new eval script: copy the filter from `eval/eval_e2e.py` and wire it in
  BEFORE the first LLM call. Verify by grepping the script for `test_set_uids`; if it's
  missing, fix the script before running.
- Invoking with `--uids`: intersect with the test set and ABORT (not warn) on any match.
- Reporting numbers: always state dev-only (101) vs full (133). Full-set numbers are
  contaminated for any generalization claim.
- Expanding the test set later: re-sample; never add UIDs already tuned against.

Test UIDs: UID0035, UID0036, UID0039, UID0050, UID0065, UID0068, UID0073, UID0093,
UID0094, UID0100, UID0108, UID0118, UID0120, UID0134, UID0147, UID0161, UID0168, UID0170,
UID0179, UID0182, UID0183, UID0187, UID0196, UID0204, UID0211, UID0212, UID0216, UID0218,
UID0227, UID0230, UID0238, UID0240.

## Running experiments

- **Don't autonomously launch full dev sweeps.** Implement, sanity-check on a couple of
  dev UIDs, then hand the run command to the user.
- Harness is `eval/eval_e2e.py`. State dev-only vs full in any reported number (see above).
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

## Setup (.env at repo root)

All LLM calls go through the **AI Studio Gemini API** (`generativelanguage.googleapis.com`)
via the `google-genai` SDK, authenticated with an **AI Studio API key** — not Vertex/ADC:
1. Create a key at https://aistudio.google.com/apikey
2. `GEMINI_API_KEY=<key>` in `.env`

Default model `gemini-3.5-flash` (override via `SKUNK_LLM_MODEL`; use bare model names, no

`google/` prefix). Strongly recommended for `lookup_external`: `FRED_API_KEY`,
`TAVILY_API_KEY`. Copy `.env.example` → `.env`.

## Data corpus (not in this repo)

Treasury Bulletin PDFs at `~/Desktop/officeqa/treasury_bulletin_pdfs/` (~20 GB, 696 files);
override with `OFFICEQA_PDF_DIR`. Pre-extracted tables and rendered pages live under
`cache/` (gitignored), populated on first use.

## Pointers

- `ARCHITECTURE.md` — architecture, design intent, plan-shape spec
- `src/skunk/plan.py` — canonical Plan dataclasses + the planner system prompt (plan JSON schema)
- `data/officeqa_pro.csv` — benchmark (not in git)
