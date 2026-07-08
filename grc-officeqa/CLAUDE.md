# CLAUDE.md

Operational guide for Claude Code sessions on grc-officeqa. For how the pipeline
itself works — architecture, design intent, the plan-shape spec — read the skunk
library's `../skunk/ARCHITECTURE.md`.

## What this is

The **Grounded Reasoning Cup / OfficeQA application layer** over the skunk library:
a declarative QA pipeline for the U.S. Treasury Bulletin corpus (696 monthly PDFs,
1939–2025). Questions are answered by composing skunk's 4 operators
(`retrieve` / `extract` / `lookup_external` / `compute`) into a typed DSL plan.
Benchmark: `data/officeqa_pro.csv` (133 questions = 33 dev + 100 test; not tracked
in git, keep locally).

Layout:

- `officeqa/` — the app package: Treasury corpus accessors (`corpus.py`), the
  page-index build pipeline + page store (`page_index/`), the skunk
  `PageContentStore` implementation (`page_store.py`), app config
  (`config.py`: `SkunkConfig.from_env`), offline prep scripts (`prep/`).
- `eval/` — the e2e harness (`eval_e2e.py`), trace dump (`util.py`), scorer
  wrapper (`scoring.py`, delegates to qatfd's vendored v5 reward), and the
  table-corrections viewer.
- `config/prompts/` — prompt-override YAMLs (corpus blurbs, few-shots, lessons).
- `cache/`, `data/`, `eval/traces/` — local artifacts, not tracked.

History: this was carved out of `../skunk/` on 2026-07-07 (library-hardening
refactor, see `../skunk/REFACTOR_PLAN.md`); the competition web harness
(`harness_ui/`) and the human-in-the-loop layer were deleted outright and live
in git history.

## 🚨 Held-out test split — READ BEFORE ANY EVAL RUN 🚨

**The canonical OfficeQA dev/test split is owned by qatfd** (the sibling evaluation
harness): `../qatfd/benchmarks/officeqa/officeqa_splits.json` — `{"dev": [33 uids],
"test": [100 uids]}`, dev = the first 25% of CSV order, generated deterministically
by `qatfd/scripts/make_splits.py`. **Do not run on, tune against, or inspect traces
of the 100 test UIDs during development** — this app's prompts are shared with qatfd
systems, so contamination here biases qatfd's reported numbers too.

- Every eval harness MUST load the splits file and filter the test UIDs out by
  default. `--include-test-set` overrides only for a deliberate final-number run.
- Writing a new eval script: copy the filter from `eval/eval_e2e.py` and wire it in
  BEFORE the first LLM call. `eval_e2e` ABORTs (not warns) when the splits file is
  missing or `--uids` names a test UID.
- Reporting numbers: always state dev-only (33) vs full (133). Full-set numbers are
  contaminated for any generalization claim.

## Running experiments

- **Don't autonomously launch full dev sweeps.** Implement, sanity-check on a couple of
  dev UIDs, then hand the run command to the user.
- Harness is `eval/eval_e2e.py`, run from this directory (`--dev-set` runs the 33-UID
  qatfd dev split). State dev-only vs full in any reported number (see above).
- The default `search_agent` retriever needs `cache/chromadb/` + `cache/clean_page_map.json`
  built offline (`officeqa/prep/` + qatfd's `engaging-scripts/create_vector_db.py
  --benchmark officeqa`); the first non-golden run errors clearly if either is missing.

## Conventions

- **Use `python3` / `pip3`, not `python` / `pip`** — no `python` on PATH (exit 127).
- Use the skunk library checkout's venv: `../skunk/venv/bin/python3` (skunk is
  installed editable there; this app is imported via sys.path, no install step).
- **No new ops** without updating `../skunk/ARCHITECTURE.md`, the planner system
  prompt in `skunk/plan.py`, the validator, AND `eval/eval_e2e.py`.
- **No formatting-only edits.** Do not reformat lines for length, change quote style,
  add blank lines between methods, or expand/collapse argument lists unless it is part
  of a substantive change to that line.

## Setup (.env)

`eval_e2e` loads `.env` here first, then falls back to `../skunk/.env` (where the
keys historically live). LLM generation defaults to **OpenRouter**
(`SKUNK_LLM_PROVIDER=openrouter`, `OPENROUTER_API_KEY`); model ids must then be full
OpenRouter ids (e.g. `google/gemini-2.5-flash`). Strongly recommended for
`lookup_external`: `FRED_API_KEY`, `TAVILY_API_KEY`.

## Data corpus (not in this repo)

Treasury Bulletin PDFs at `~/Desktop/officeqa/treasury_bulletin_pdfs/` (~20 GB, 696
files); override with `OFFICEQA_PDF_DIR`. Parsed JSONs at `OFFICEQA_PARSED_JSON_DIR`.
Pre-extracted artifacts live under `cache/` (gitignored).
