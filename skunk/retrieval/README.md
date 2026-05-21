# Retrieval

Page-level retrieval for grounded document QA. Dev benchmark: [OfficeQA](https://github.com/databricks/officeqa). Primary metric: **`all_gold_pages_recall`**.

## Layout

```text
retrieval/
├── run.sh
├── README.md
├── scripts/
│   └── archive_legacy_tmp.sh    Move old /tmp/officeqa artifacts to _old/
└── skunk_retrieval/
    ├── cli.py                   CLI
    ├── config.py                PipelineConfig, .skunk metadata
    ├── dataset.py               Corpus path discovery
    ├── format_selector.py       LLM format judges + interactive choice
    ├── llm.py                   OpenRouter + Gemini
    ├── planner.py               Decompose + retry for hard questions
    ├── pipeline.py              StrongRetriever (RRF merge)
    ├── factory.py               Wire page-table + LateOn + strong wrapper
    ├── page_table.py            SQLite FTS index + search
    ├── lateon.py                ColBERT (LateOn)
    ├── officeqa.py              OfficeQA eval adapter only
    ├── corpus.py                Record builders
    ├── display.py               stderr progress
    ├── profiles/expansions.json Optional dev query expansions
    └── _old/                    Legacy file-chunk BM25 (ablations only)
```

## Pipeline (defaults on)

**Offline:** `choose-format` → `preprocess` (FTS + LateOn) — or `bash run.sh`.

**Online:** rule + LLM query plan → FTS (page/table/row/file) → LateOn rerank → decompose/retry on hard questions.

| Component | Default | Disable |
|-----------|---------|---------|
| Row FTS | `row_k=200` | `--row-k 0` |
| LateOn + expand pages/rows | on | `--no-lateon` |
| LLM planner | on | `--no-llm-planner` |
| Decompose / retry | on | `--no-decompose` / `--no-retry` |
| Eval k | `500` | `EVAL_K=100` |

## Quick start

```bash
cd skunk/retrieval
export OPENROUTER_API_KEY='...'
bash run.sh
bash run.sh --limit 10
```

Gemini:

```bash
export GEMINI_API_KEY='...' SKUNK_LLM_PROVIDER=gemini SKUNK_LLM_MODEL=gemini-2.5-flash
bash run.sh
```

## New corpus

```bash
export SKUNK_DATA_DIR=/path/to/corpus
export OPENROUTER_API_KEY='...'

PYTHONPATH=skunk/retrieval skunk/.venv/bin/python -m skunk_retrieval choose-format --data-dir "$SKUNK_DATA_DIR"
PYTHONPATH=skunk/retrieval skunk/.venv/bin/python -m skunk_retrieval preprocess --data-dir "$SKUNK_DATA_DIR" --skip-format-choice
```

Runtime search:

```bash
PYTHONPATH=skunk/retrieval skunk/.venv/bin/python -m skunk_retrieval search-page-table \
  --data-dir "$SKUNK_DATA_DIR" \
  --index-file "$SKUNK_DATA_DIR/page_table.sqlite" \
  --lateon-folder "$SKUNK_DATA_DIR/lateon" \
  --query "..." --k 50 --openrouter-model google/gemini-2.5-flash
```

## Artifacts (`$SKUNK_DATA_DIR`, default `/tmp/officeqa`)

```text
.skunk/canonical_source.json
.skunk/pipeline_config.json
page_table.sqlite
lateon/
results/strong_*_rows.jsonl
_old/                          # legacy bm25.pkl, old eval jsonl (after archive script)
```

Archive old `/tmp` runs:

```bash
bash skunk/retrieval/scripts/archive_legacy_tmp.sh
```

## CLI

```bash
PYTHONPATH=skunk/retrieval skunk/.venv/bin/python -m skunk_retrieval <command>
```

`preprocess` · `choose-format` · `build-page-table` · `build-lateon` · `search-page-table` · `eval-page-table` · `oracle`

## Environment

| Variable | Default |
|----------|---------|
| `SKUNK_DATA_DIR` | `/tmp/officeqa` |
| `SKUNK_LLM_PROVIDER` | `openrouter` |
| `SKUNK_LLM_MODEL` | `google/gemini-2.5-flash` |
| `EVAL_K` | `500` |
| `SKUNK_DISABLE` | `lateon`, `llm`, `decompose`, `retry`, `rows` |

## Metrics (OfficeQA Pro, n=133)

Historical `/tmp/officeqa/results/` (pre-strong pipeline):

| Config | `all_gold_pages_recall` |
|--------|-------------------------|
| FTS only, k=100 | 0.564 |
| FTS + LateOn + OpenRouter, k=200 | 0.714 |
| Same, k=2000 | 0.865 |

Re-run `bash run.sh` for current strong-default numbers.

## Dependencies

`skunk/.venv` — PyLate/LateOn, torch, huggingface_hub. For Gemini: `pip install google-genai`.
