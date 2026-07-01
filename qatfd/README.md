# qatfd evaluation harness

Runs a matrix of **systems × QA benchmarks** over large document corpora, for the
qatfd paper. Built on two subclassable abstractions and a shared parallel runner;
imports the sibling `skunk` package as a library.

## Layout

```
qatfd/
  benchmarks/   Benchmark ABC + OfficeQA, BrowseComp-Plus (+ LLM judge, KARL test ids)
  systems/      System ABC + the four systems (+ semantic filter)
  registry.py   name -> Benchmark/System
  runner.py     Hydra CLI + parallel runner -> results/<benchmark>/<system>/<run>/report.csv
configs/        Hydra config groups (experiments/, benchmarks/, systems/)
run_all.sh      meta driver over the {benchmarks} x {systems} matrix (Hydra multirun)
results/        run output (gitignored)
```

## Benchmarks

| name | questions | corpus / index | scorer |
|------|-----------|----------------|--------|
| `officeqa` | `skunk/officeqa_pro.csv` | `skunk/.chromadb` (collection from `SKUNK_CHROMADB_COLLECTION`) + `treasury_bulletins_cleaned/clean_page_map.json` | cup-kit numeric (`score_correct`) |
| `browsecomp_plus` | `skunk/browsecomp-plus/browsecomp_plus_decrypted.jsonl` | `skunk/.chromadb` collection `qwen-browsecomp-plus` + doc map from element metadata | LLM-as-judge (single nugget) |
| `trec-biogen` | `skunk/biogen/2025_task_a.json` (40 Task A questions) | `skunk/.chromadb` collection `qwen-biogen-0.6b` (26.8M PubMed abstracts, Qwen3-0.6B) + doc map from element metadata | LLM-as-judge nugget-completion recall (KARL D.1) |
| `financebench` | `skunk/financebench/financebench_open_source.jsonl` (150 open-source questions) | `skunk/.chromadb` collection `qwen-financebench` (368 SEC-filing PDFs indexed at page level, Qwen3-8B) + page-text doc map from element metadata | LLM-as-judge (single nugget) |

**Dev / test splits.** Every benchmark carries an explicit dev/test split in
`skunk/splits/<benchmark>.json` (`{"dev": [qids], "test": [qids]}`), wired via
`benchmarks.splits_path`. The runner defaults to `experiments.split=dev`; pass
`experiments.split=test` only for final-number runs. The splits are generated
deterministically by [`scripts/make_splits.py`](scripts/make_splits.py) — **that script is
the source of truth**; regenerate with `PYTHONPATH=. python3 scripts/make_splits.py`.

| benchmark | dev | test | construction |
|-----------|----:|-----:|--------------|
| `officeqa` | 33 | 100 | first 25% of CSV order = dev, last 75% = test (not shuffled) |
| `browsecomp_plus` | 50 | 230 | test = KARL's 230 calibrated-subset query_ids; dev = 50 sampled (seed 0) from the disjoint remainder |
| `trec_biogen` | 10 | 30 | shuffle the 40 Task A qids (seed 0), then slice |
| `financebench` | 50 | 100 | shuffle the 150 qids (seed 0), then slice |
| `qampari` | 50 | 1000 | test = all 1000 `test_data.jsonl` qids (KARL's eval set); dev = 50 sampled (seed 0) from `train_data.jsonl` (disjoint), materialized to `qampari/dev_data_sample50.jsonl` |
| `freshstack` | 184 | 203 | split by TOPIC: dev = all `laravel` queries, test = all `langchain` queries |

"shuffle (seed 0)" = sort the qids for a stable base order, then `random.Random(0).shuffle`,
then slice — so membership depends only on the seed, not on file/load order. FreshStack's
split is the topic itself (each topic is a separate corpus/collection): run the dev set with
the default `benchmarks.topic=laravel`, and the test set with
`benchmarks.topic=langchain experiments.split=test`.

Caveat: OfficeQA's dev set (first 25% by CSV order, ~UID0001–UID0057) overlaps several of
skunk's held-out 32 test UIDs, so qatfd OfficeQA **dev**-tuned numbers are not comparable to
skunk's OfficeQA **test** numbers.

The per-row `score` column is correctness in `[0,1]`: 0/1 for the binary benchmarks,
and the graded nugget-completion recall for `trec-biogen` (the run summary averages it).
TREC-BioGen scores against the **official** decomposed nuggets (`baseline_labels.json`,
~24.6/q), which are finer-grained than KARL's consolidated set (~7.1/q) — so the absolute
number is a valid nugget-recall but is **not** directly comparable to KARL's reported 85.0.
Its corpus/index is built offline from the 2025 Pyserini collection — see
`engaging-scripts/run_biogen_embeddings.slurm`. Runs must embed queries with the **same**
Qwen3-0.6B model: `systems.emb_model_id=qwen/qwen3-embedding-0.6b` (or `emb_provider=local`
with `QATFD_LOCAL_EMB_MODEL=Qwen/Qwen3-Embedding-0.6B`).

FinanceBench reports correctness `0/1` from a single-nugget judge (set `benchmarks.judge_model`),
and recall at two granularities (`page_recall`, `doc_recall`) since gold is labeled at the
(document, page) level. Its corpus/index is built offline from the 368 SEC-filing PDFs, decomposed
into text / table / figure **elements** and embedded with the default Qwen3-8B (same as the default
query embedder, so no `emb_model_id` override is needed). Retrieval stays page-level: the chroma
`doc_id` is the page key `{doc_name}::p{page_num}`. Three steps:

```bash
# 1. PREPROCESS (CPU + OpenRouter, no GPU): PDFs -> per-doc element JSONs, in 3 resumable phases:
#    RENDER (ProcessPool: page PNG + text-layer cache), LLM (ThreadPool: google/gemini-3.1-flash-lite
#    extracts table markdown + figure summaries from each non-blank page), ASSEMBLE.
#    Needs pymupdf + OPENROUTER_API_KEY. Validate on a few docs first with --sample. PAGE-LEVEL
#    resumable: each phase skips work already persisted, so a rerun only renders/LLMs the missing
#    pages then reassembles (a single failed page never reprocesses its whole doc). All dir args
#    accept a local path OR an s3:// prefix — on a low-disk cluster stream to/from S3 (see
#    run_financebench_preprocess.slurm; upload PDFs once with `aws s3 sync skunk/financebench/pdfs
#    s3://carnot-research/financebench/pdfs`). Two intermediate prefixes ({output_dir}-renders,
#    {output_dir}-pages) hold the caches; only --output_dir holds the final {doc}.json.
OPENROUTER_API_KEY=sk-or-... python engaging-scripts/preprocess_financebench_pdfs.py \
    --input_dir  ../skunk/financebench/pdfs \
    --output_dir ../skunk/financebench/financebench-elements
# 2. EMBED (GPU): element JSONs -> Qwen3-8B embeddings — see run_financebench_element_embeddings.slurm.
#    --input_dir/--output_dir also accept s3:// prefixes (stream JSONs in, embeddings/metadata out).
python engaging-scripts/compute_financebench_element_embeddings.py \
    --input_dir  ../skunk/financebench/financebench-elements \
    --output_dir ../skunk/financebench/financebench-element-embeddings
# 3. load them into the chroma collection (on a box with disk; if step 2 wrote to S3, first
#    `aws s3 sync s3://carnot-research/financebench/financebench-element-embeddings ./fb-embeddings`)
python -m skunk.search_agent.prep.create_vector_db \
    --embeddings-dir ../skunk/financebench/financebench-element-embeddings \
    --collection-name qwen-financebench --chroma-path ../skunk/.chromadb --benchmark finance_bench
```

## Systems (registry names)

- `rag_llm` — one vector search, stuff top-k chunks, single LLM call.
- `search_agent` — skunk SearchAgent (grep / vector-search / read / prune).
- `qatfd_search_agent` — `search_agent` plus a `semantic_filter` tool the agent can call mid-loop.

## Setup

```bash
pip install -e ../skunk   # make `skunk` importable
pip install -e .          # this package
```

Configuration is read from `skunk/.env` (loaded automatically before importing skunk)
and the `SKUNK_*` env vars (`SkunkConfig.from_env()`): model, provider, chromadb dir,
collection, embedding model, agent step budget, etc. API keys (`OPENROUTER_API_KEY`
or `GEMINI_API_KEY`) must be present in the environment.

### Query embedding backend

The corpora are Qwen3-Embedding-8B (4096-dim). Pick how queries are embedded with
`systems.emb_provider` (defaults to `openrouter`):
- `openrouter`: OpenRouter `qwen/qwen3-embedding-8b` (needs `OPENROUTER_API_KEY`).
- `local`: local `sentence-transformers` `Qwen/Qwen3-Embedding-8B` (`QATFD_LOCAL_EMB_MODEL` to override).

### Cost / token accounting

Each report row carries `total_input_tokens`, `total_output_tokens`,
`total_cache_input_tokens` (system answer only; the judge is excluded) and `cost`.
Cost needs a price table, set via `llm_prices` in `configs/systems/base.yaml`
(e.g. `qwen3: {in: 0.1, out: 0.4, cached: 0.025}` — USD per 1M tokens, keyed by
model substring; `cached` is optional and bills the cached input portion); unknown
models cost 0.

## Run

The runner is a [Hydra](https://hydra.cc) app. Three config groups compose a run —
`experiments=` (run-level knobs), `benchmarks=`, `systems=` — and any leaf is
overridable on the CLI (e.g. `experiments.sample=5`). See `configs/config.yaml`.

```bash
# One system on a few OfficeQA dev questions
python3 -m qatfd.runner systems=rag_llm benchmarks=officeqa experiments.sample=5 experiments.workers=4

# Specific qids (dev only; aborts on test-set collision)
python3 -m qatfd.runner systems=search_agent benchmarks=officeqa experiments.qids=[UID0001,UID0002] experiments.workers=2

# BrowseComp-Plus with a chosen judge model
python3 -m qatfd.runner systems=rag_llm benchmarks=browsecomp_plus experiments.sample=5 benchmarks.judge_model=gemini-3.5-flash

# FinanceBench (all 150 are the held-out test set; pick a judge model)
python3 -m qatfd.runner systems=rag_llm benchmarks=financebench experiments.split=test experiments.sample=5 systems.top_k=20 benchmarks.judge_model=gemini-3.5-flash

# Embed queries locally instead of via OpenRouter
python3 -m qatfd.runner systems=rag_llm benchmarks=officeqa systems.emb_provider=local

# Whole matrix (or a subset) via Hydra multirun
SAMPLE=20 BENCHMARKS=officeqa SYSTEMS="rag_llm,search_agent" ./run_all.sh
```

Output: `results/<benchmark>/<system>/<run-name>_<timestamp>/` containing `report.csv`,
`traces/events.jsonl` (structured event stream), and per-question `.txt` / `.log` traces.
