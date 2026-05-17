# skunk — OfficeQA pipeline

A declarative QA pipeline over the U.S. Treasury Bulletin corpus. Questions are decomposed into 4-operator DSL plans; an orchestrator walks the plan and dispatches operators that retrieve specific PDF pages, extract values, look up external facts, and produce the final answer via a sandboxed-Python compute terminator with an LLM-based output-format verifier.

## Quick start

```bash
pip install -e ".[dev]"
cp .env.example .env   # then fill in GEMINI_API_KEY (+ OPENROUTER_API_KEY for non-search calls)
python -m skunk.run --uid UID0001 --csv data/officeqa_pro.csv --cached-plan --golden
```

## What's where

| Path | Purpose |
|---|---|
| `ARCHITECTURE.md` | Design intent + plan-shape spec + per-operator semantics |
| `CLAUDE.md` | Project guide for Claude Code sessions |
| `data/officeqa_pro.csv` | 133-question benchmark with golden retrieval pages and answers (not in git) |
| `data/dsl_planning_pass.csv` | 133 validated DSL plans (used at runtime via `--cached-plan`) |
| `data/page_index/` | Shipped page-index retriever (`concept_tree.json` + self-contained `retrieve.py`) |
| `src/skunk/` | Runtime: planner, orchestrator, four operator modules, page-index builder |
| `eval/` | Harnesses: `eval_e2e.py` (end-to-end), `eval_retrieve.py` (retrieval-only) |

## Status

The 4-op DSL is finalized. All four operators are implemented:
- `retrieve` runs the shipped page-index pipeline (L1 chapter pick + year-window filter over `data/page_index/concept_tree.json`); use `--golden` to bypass it with the benchmark's labeled pages;
- `extract` emits a question-driven list of `AnnotatedValue` entries (each a `scalar`, `vector`, or `table` — no nesting beyond those shapes) via parsed-JSON Tier 1, PyMuPDF Tier 2, and vision Tier 3; pass `visual_only=True` to go straight to vision for charts/figures;
- `lookup_external` does a single Gemini call (Google Search grounding, or FRED/BLS via emitted Python) for facts not in the bulletin corpus;
- `compute` terminates the chain with codegen → sandbox exec → LLM self-critique (ACCEPT/REVISE).

Plans are produced live by the planner; `data/dsl_planning_pass.csv` is rebuilt lazily by `skunk.run` on first execution per UID.
