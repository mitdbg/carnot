# qatfd evaluation harness

Runs a matrix of **systems × QA benchmarks** over large document corpora, for the
qatfd paper. Built on two subclassable abstractions and a shared parallel runner;
imports the sibling `skunk` package as a library.

## Layout

```
qatfd/
  qatfd/
    benchmarks/ Benchmark ABC + the six benchmarks (+ LLM judge)
    systems/    System ABC + the four systems (+ semantic filter)
    registry.py name -> Benchmark/System
    runner.py   Hydra CLI + parallel runner -> results/<benchmark>/<system>/<run>/report.csv
  configs/      Hydra config groups (experiments/, benchmarks/, systems/)
  benchmarks/   Benchmark DATA + indices, one dir per benchmark (gitignored; see below)
  run_all.sh    meta driver over the {benchmarks} x {systems} matrix (Hydra multirun)
  results/      run output (gitignored)
```

Note the two `benchmarks/`: `qatfd/qatfd/benchmarks/` is the code (loaders/scorers);
`qatfd/benchmarks/` is the data root (corpora, Chroma indices, questions, splits).

How corpora map onto chunks / retrieval units (`doc_id`) / source groups — and what those
ids mean per benchmark — is specified in [CORPUS_MODEL.md](CORPUS_MODEL.md).

## Benchmarks

All benchmark **data** (indices, questions, corpora, split files, and the optional
prompt-override YAML `benchmarks.prompts_path`) lives under `qatfd/benchmarks/<benchmark>/`
(override the root with `QATFD_BENCHMARKS_DIR`). Each embedded Chroma store is
`<benchmark>/chromadb` (TREC-BioGen shards it into `chromadb/r0..r3`; FreshStack is per-topic
at `freshstack/<topic>/chromadb`).

| name | questions | corpus / index | scorer |
|------|-----------|----------------|--------|
| `officeqa` | `officeqa/officeqa_pro.csv` | `officeqa/chromadb` (`officeqa-qwen-8b`, Qwen3-8B) + `officeqa/treasury_bulletins_cleaned/clean_page_map.json` | cup-kit numeric (`score_correct`) |
| `browsecomp_plus` | `browsecomp-plus/browsecomp_plus_decrypted.jsonl` | `browsecomp-plus/chromadb` (`browsecomp-plus-qwen-8b`, Qwen3-8B) + doc map from element metadata | LLM-as-judge (single nugget) |
| `trec_biogen` | `trec-biogen/2025_task_a.json` (40 Task A questions) | `trec-biogen/chromadb` (`qwen-biogen-0.6b`, 26.8M PubMed abstracts, Qwen3-0.6B, 4 shards) + lazy chroma doc map | LLM-as-judge nugget-completion recall (KARL D.1) |
| `financebench` | `financebench/financebench_open_source.jsonl` (150 open-source questions) | `financebench/chromadb` (`financebench-qwen-8b`, 368 SEC-filing PDFs indexed at page level, Qwen3-8B) + page-text doc map from element metadata | LLM-as-judge (single nugget) |
| `qampari` | `qampari/test_data.jsonl` (1000) + `qampari/dev_data_sample50.jsonl` | `qampari/chromadb` (`qwen-qampari-0.6b`, ~25.9M Wikipedia chunks, Qwen3-0.6B; served over HTTP) + lazy chroma doc map | LLM-as-judge nugget (entity) recall |
| `freshstack` | `freshstack/<topic>/queries.jsonl` (langchain=test, laravel=dev) | `freshstack/<topic>/chromadb` (`freshstack-<topic>-qwen-0.6b`, Qwen3-0.6B) + doc map from `corpus.jsonl` | LLM-as-judge nugget-completion recall |

**Dev / test splits.** Every benchmark carries an explicit dev/test split co-located with its
data at `qatfd/benchmarks/<benchmark>/<benchmark>_splits.json` (`{"dev": [qids], "test": [qids]}`),
wired via `benchmarks.splits_path`. The runner defaults to `experiments.split=dev`; pass
`experiments.split=test` only for final-number runs. The splits are generated
deterministically by [`scripts/make_splits.py`](scripts/make_splits.py) — **that script is
the source of truth**; regenerate with `PYTHONPATH=. python3 scripts/make_splits.py`.

| benchmark | dev | test | construction |
|-----------|----:|-----:|--------------|
| `officeqa` | 33 | 100 | first 25% of CSV order = dev, last 75% = test (not shuffled) |
| `browsecomp_plus` | 50 | 230 | test = KARL's 230 calibrated-subset query_ids; dev = 50 sampled (seed 0) from the disjoint remainder |
| `trec_biogen` | 10 | 30 | shuffle the 40 Task A qids (seed 0), then slice |
| `financebench` | 50 | 100 | shuffle the 150 qids (seed 0), then slice |
| `qampari` | 50 | 1000 | test = all 1000 `test_data.jsonl` qids (KARL's eval set); dev = 50 sampled (seed 0) from `train_data.jsonl` (disjoint), materialized to `benchmarks/qampari/dev_data_sample50.jsonl` |
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
Qwen3-0.6B model: `systems.emb_model_id=qwen/qwen3-embedding-0.6b` (or `emb_provider=vllm`
with `emb_model_id=Qwen/Qwen3-Embedding-0.6B` served locally — see "Running on vLLM").

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
#    run_financebench_preprocess.slurm; upload PDFs once with `aws s3 sync benchmarks/financebench/pdfs
#    s3://carnot-research/financebench/pdfs`). Two intermediate prefixes ({output_dir}-renders,
#    {output_dir}-pages) hold the caches; only --output_dir holds the final {doc}.json.
OPENROUTER_API_KEY=sk-or-... python engaging-scripts/preprocess_financebench_pdfs.py \
    --input_dir  benchmarks/financebench/pdfs \
    --output_dir benchmarks/financebench/financebench-elements
# 2. EMBED (GPU): element JSONs -> Qwen3-8B embeddings — see run_financebench_element_embeddings.slurm.
#    --input_dir/--output_dir also accept s3:// prefixes (stream JSONs in, embeddings/metadata out).
#    The metadata_rank*.json this writes are ALSO the page-text doc map (benchmarks.fb_metadata_glob),
#    so keep the financebench-element-embeddings dir alongside the built index.
python engaging-scripts/compute_financebench_element_embeddings.py \
    --input_dir  benchmarks/financebench/financebench-elements \
    --output_dir benchmarks/financebench/financebench-element-embeddings
# 3. load them into the chroma collection (on a box with disk; if step 2 wrote to S3, first
#    `aws s3 sync s3://carnot-research/financebench/financebench-element-embeddings ./fb-embeddings`)
python -m skunk.search_agent.prep.create_vector_db \
    --embeddings-dir benchmarks/financebench/financebench-element-embeddings \
    --collection-name financebench-qwen-8b --chroma-path benchmarks/financebench/chromadb --benchmark financebench
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
and the `SKUNK_*` env vars (`SkunkConfig.from_env()`): model, provider, embedding model,
agent step budget, etc. `OPENROUTER_API_KEY` must be present in the environment (and
`VLLM_API_KEY` only when a vLLM server was started with `--api-key`). The per-benchmark
chromadb dir + collection come from the benchmark config
(under `qatfd/benchmarks/`), not `SKUNK_*`.

### Query embedding backend

The corpora are Qwen3-Embedding-8B (4096-dim). Pick how queries are embedded with
`systems.emb_provider` (defaults to `openrouter`):
- `openrouter`: OpenRouter `qwen/qwen3-embedding-8b` (needs `OPENROUTER_API_KEY`).
- `vllm`: a local vLLM embedding server; set `systems.emb_model_id` to the served model name
  and give it a `vllm_base_urls` entry (see "Running on vLLM" below).

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

# Whole matrix (or a subset) via Hydra multirun
SAMPLE=20 BENCHMARKS=officeqa SYSTEMS="rag_llm,search_agent" ./run_all.sh
```

Output: `results/<benchmark>/<system>/<run-name>_<timestamp>/` containing `report.csv` and
per-question `traces/<qid>.jsonl` (structured event stream) / `<qid>.txt` (human-readable dump) traces.

## Running on vLLM

Any model in a run — the agent model, the semantic-filter model, the embedder — can be
served by a **local vLLM server** instead of OpenRouter. Routing is per call through one
`LLMClient`: a model listed in `systems.vllm_base_urls` (model id → server base URL) goes
to its vLLM server; every other model (e.g. `benchmarks.judge_model`) stays on
`systems.llm_provider`. vLLM serves ONE model per server process, so a multi-model run
needs one server per model.

On the GPU box (vLLM installed there; it is deliberately not a qatfd dependency):

```bash
tmux new -s vllm
./scripts/run_vllm_servers.sh 'Qwen/Qwen3-32B:gpus=0:mem=0.9' \
                              'Qwen/Qwen3-8B:gpus=1:mem=0.45' \
                              'Qwen/Qwen3-Embedding-0.6B:gpus=1:mem=0.2:task=embed'
```

The script starts one `vllm serve` per spec (ports `BASE_PORT`+i, per-server
`CUDA_VISIBLE_DEVICES` / tensor-parallel / memory-fraction / max-len knobs), waits until
every `/v1/models` answers, writes `scripts/vllm_manifest.json` (model → base URL), and
prints the paste-ready Hydra override. `--served-model-name` defaults to the model id, so
the map keys, `systems.llm_model` / `semantic_filter_model` / `emb_model_id`, and the
server all agree. `--dry-run` previews the commands without vllm or a GPU.

Then point a run at the servers (from this box or the GPU box — `ADVERTISE_HOST` in the
printed URLs makes them reachable remotely):

```bash
# Agent + semantic filter local, judge on OpenRouter (see configs/systems/ablation_search_agent_vllm.yaml)
python3 -m qatfd.runner systems=ablation_search_agent_vllm benchmarks=officeqa \
    experiments.sample=2 benchmarks.judge_model=google/gemini-3.5-flash

# Ad-hoc: route just the agent model of a vanilla system to a server
python3 -m qatfd.runner systems=search_agent benchmarks=officeqa systems.llm_model=Qwen/Qwen3-32B \
    '++systems.vllm_base_urls={Qwen/Qwen3-32B: "http://<gpu-host>:8100/v1"}'
```

Notes: models listed in `vllm_base_urls` are costed $0 regardless of `llm_prices` — a model
priced for OpenRouter runs stays free when served locally; give served models a high `llm_model_rpm` entry so the default
1000 RPM client-side bucket doesn't throttle them; `systems.vllm_extra_body` merges extra
JSON into every vLLM chat request (e.g. `{chat_template_kwargs: {enable_thinking: false}}`
to disable Qwen3-style thinking — skunk's `effort` tiers are OpenRouter-only and ignored on
the vLLM path); `resume_dir` refuses runs recorded before these config keys existed (the
composed config genuinely changed).
