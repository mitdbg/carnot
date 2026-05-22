# skunk — OfficeQA pipeline

A declarative QA pipeline over the U.S. Treasury Bulletin corpus. Questions are decomposed into 4-operator DSL plans; an orchestrator walks the plan and dispatches operators that retrieve specific PDF pages, extract values, look up external facts, and produce the final answer via a codegen-and-exec compute terminator with an LLM-based output-format verifier.

## Quick start

```bash
pip install -e ".[dev]"
cp .env.example .env   # then fill in GEMINI_API_KEY (+ OPENROUTER_API_KEY for non-search calls)
python -m eval.eval_e2e \
    --csv data/officeqa_pro.csv --report eval/e2e_report.csv \
    --uids UID0001 --golden
```

## What's where

| Path | Purpose |
|---|---|
| `ARCHITECTURE.md` | Design intent + plan-shape spec + per-operator semantics |
| `CLAUDE.md` | Project guide for Claude Code sessions |
| `data/officeqa_pro.csv` | 133-question benchmark with golden retrieval pages and answers (not in git) |
| `data/page_index/` | Shipped page-index retriever (`concept_tree.json` + self-contained `retrieve.py`) |
| `src/skunk/` | Runtime: planner, orchestrator, four operator modules, page-index builder |
| `eval/` | End-to-end harness `eval_e2e.py` + shared helpers |

## Status

The 4-op DSL is finalized. All four operators are implemented:
- `retrieve` is currently a placeholder pending integration of an external retriever; it honors `ctx.config.golden_pages` only (use `--golden` for eval) and raises `NotImplementedError` otherwise. The previous page-index pipeline (L1 chapter pick + year-window filter, optional BM25 rerank) is preserved at `src/skunk/page_index/retrieve_prototype.py` (`PageIndexRetrievePrototype`) for regression comparison;
- `extract` emits a question-driven list of `AnnotatedValue` entries (each a `scalar`, `vector`, or `table` — no nesting beyond those shapes) via parsed-JSON Tier 1 and vision Tier 2; pass `visual_only=True` to go straight to vision for charts/figures;
- `lookup_external` does a single Gemini call (Google Search grounding, or FRED/BLS via emitted Python) for facts not in the bulletin corpus;
- `compute` terminates the chain with codegen → in-process exec → LLM self-critique (ACCEPT/REVISE).

Plans are produced live by the planner on every run.
