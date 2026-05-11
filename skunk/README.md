# skunk — OfficeQA pipeline

A declarative QA pipeline over the U.S. Treasury Bulletin corpus. Questions are decomposed into 4-operator DSL plans; an orchestrator walks the plan and dispatches subagents that retrieve specific PDF pages, extract values, look up external facts, and produce the final answer via a sandboxed-Python compute terminator with an LLM-based output-format verifier.

## Quick start

```bash
pip install -e ".[dev]"
cp .env.example .env   # then fill in ANTHROPIC_API_KEY and GEMINI_API_KEY
pytest -v tests/
python tools/plan_dsl_with_gemini.py --smoke   # generate plans for 7 sample questions
```

## What's where

| Path | Purpose |
|---|---|
| `ARCHITECTURE.md` | Design intent (page-level retrieval, catalog, eval decomposition) |
| `DSL.md` | Formal DSL grammar and per-op semantics |
| `CLAUDE.md` | Project guide for Claude Code sessions |
| `data/officeqa_pro.csv` | 133-question benchmark with golden retrieval pages and answers |
| `data/dsl_planning_pass.csv` | 133 validated DSL plans (use as planner few-shots) |
| `src/skunk/` | Runtime: planner, orchestrator, subagents, prep |
| `eval/` | Independent retrieval / extraction / e2e harnesses |
| `tools/plan_dsl_with_gemini.py` | Regenerates `dsl_planning_pass.csv` (Gemini 2.5 Flash, no PZ) |
| `tools/measure_retrieval_drift.py` | Drift study that motivated the catalog-first design |

## Status

The 4-op DSL is finalized. All four subagents are implemented:
- `retrieve` short-circuits on golden page injection (full corpus index is future work);
- `extract` emits a question-driven dict of named entries (each a `scalar`, `vector`, or `table` — no nesting beyond those shapes) via JSON-backed Tier 1, PyMuPDF Tier 2, and vision Tier 3; pass `visual_only=True` to go straight to vision for charts/figures;
- `lookup_external` does a single Gemini call for facts (numeric, dates, named entities);
- `compute` terminates the chain with plan→code→exec→verify.

Plans are produced live by the planner; `data/dsl_planning_pass.csv` is rebuilt lazily as questions run, or eagerly via `tools/plan_dsl_with_gemini.py`.
