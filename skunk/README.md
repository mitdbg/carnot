# skunk — OfficeQA pipeline

A declarative QA pipeline over the U.S. Treasury Bulletin corpus. Questions are decomposed into 6-operator DSL plans; an orchestrator walks the plan and dispatches subagents that retrieve specific PDF pages, extract values, run sandboxed Python, and format the answer.

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

The 6-op DSL is finalized; all 133 benchmark questions have validated plans. Four subagents are implemented (`compute`, `format`, `lookup_external`, `read_visual`); `extract` remains a stub — see its TODO comment for the tier-dispatch work needed before extraction quality can be measured. The `retrieve` subagent is also pending its page-index implementation.
