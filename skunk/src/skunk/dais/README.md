# DAIS slim pipeline (one-off scaffolding)

Stands up SearchAgent tooling + a synthetic QA **test** set for a *new* corpus produced by
the same PDF parser as OfficeQA, but over different documents — the U.S. **"Combined
Statement of Receipts, Expenditures, and Balances"** ("Account of Receipts and
Expenditures"), an **annual** federal fiscal report (~1793–2024). This package is
intentionally isolated from the rest of `skunk` and can be deleted once the DAIS
experiments are done.

Inputs (one parsed JSON + one PDF per source file; dirs may contain spaces — quote them):
- a directory of PDFs (`<file_id>.pdf`)
- a directory of parsed/unclean element JSONs (`<file_id>.json`) — `document.elements[*]`
  with `id, type, content, bbox[0].{coord, page_id}`. **`page_id` is 0-indexed** and bbox
  coords are in **300-DPI pixel space**.

Identifiers: `file_id` = file stem, `doc_id = f"{file_id}_{page_id}"`,
`chunk_id = f"{file_id}_{page_id}_{element_id}"`. The corpus is annual, so vector-DB metadata
carries **`year` (int)** + **`source`** (era family: `historical`/`modern`/`transition`/
`govinfo_receipts`) — no month. One year may span many files (e.g. modern per-chapter files).

## Environment

- `GEMINI_API_KEY` — Gemini (default LLM for cleaning / generation / rollouts / quality filter).
- `OPENROUTER_API_KEY` — required for Qwen embeddings, dedup judge, and the automatic
  Gemini→OpenRouter fallback on the cleaning calls.
- Run with `python3` (no `python` on PATH).

LLM default is `gemini-3.5-flash` via `google.genai`; the cleaning step
([`clean_corpus.py`](clean_corpus.py)) auto-falls-back to OpenRouter `google/gemini-3.5-flash`
on persistent rate limits (see `DaisLLM` in [`dais_common.py`](dais_common.py)). The
SearchAgent-based stages (rollouts, quality filter) run on Gemini with built-in 429 backoff;
pass `--provider openrouter` to `datagen.py` to switch those wholesale if Gemini throttles.

## Run order

```bash
# 1. Clean: correct TABLES only (pages kept in parser order) -> clean_page_map.json.
#    Opens each PDF once; one LLM call per table; pages with no tables make no LLM call.
python3 -m skunk.dais.clean_corpus \
    --input-json-dir "officeqa_gdrive/Parsed JSON" --pdfs-dir officeqa_gdrive/PDFs \
    --output-dir dais_cleaned

# 2. Embeddings: Qwen3-Embedding-8B over RAW parsed JSON (qwen-v2 recipe) via OpenRouter
OPENROUTER_API_KEY=... python3 -m skunk.dais.compute_qwen_embeddings \
    --input_dir "officeqa_gdrive/Parsed JSON" --output_dir dais_embeddings

# 3. Vector DB: build the slim collection (OFFLINE — server must be down)
python3 -m skunk.dais.create_vector_db \
    --embeddings-dir dais_embeddings --collection-name dais-slim \
    --chroma-path .chromadb-dais-slim

# 4. Serve the slim collection
SKUNK_CHROMADB_DIR=.chromadb-dais-slim ./scripts/run_chroma_server.sh dais-slim   # :8001

# 5. Datagen: full funnel (generate -> dedup -> Gemini rollouts -> quality filter)
OPENROUTER_API_KEY=... python3 -m skunk.dais.datagen \
    --clean-page-map dais_cleaned/clean_page_map.json \
    --chroma-collection-name dais-slim --num-new-seeds 20
```

Notes:
- **Table correction only** — page-element reordering was intentionally dropped (this corpus
  is table-heavy with little free text). Cleaned pages are elements in parser order with each
  table's HTML replaced by corrected Markdown.
- Steps 1 and 2 are independent and decoupled: cleaning feeds `read_document`; embeddings
  are computed from the **raw** parsed JSON (`preprocess_text(..., strip_years=False)`),
  byte-for-byte the `qwen-v2` recipe.
- `--bbox-dpi` (default 300) sets the coord→point scale for table crops; `--text-fallback`
  emits LLM-free parser-order pages for any PDF-less JSON (defensive — the full corpus
  currently has a PDF for every JSON).
- `run_chroma_server.sh` needs to serve the slim dir (`SKUNK_CHROMADB_DIR=.chromadb-dais-slim`
  or the equivalent arg) rather than the default `cache/chromadb`.
- Datagen reuses OfficeQA assets (few-shot pairs from `officeqa_pro.csv`, datagen guidance,
  dedup/QF prompts) as *style* anchors, but uses **DAIS-specific corpus notes** (annual, year
  int / source / page_id / type; no month) so the agent filters correctly. Pass
  `--few-shot-csv` if `officeqa_pro.csv` isn't at the repo-root default.

## Output

`datagen.py` writes the validated synthetic test set to `--out-dir` (default
`dais_synthetic_qa/`):
- `dais_synthetic_test_set.csv` — `officeqa_pro.csv` schema (`uid, question, answer,
  source_docs, source_files, difficulty`)
- `dais_synthetic_test_set.json` — full provenance (answer nuggets, chunk/doc ids, QF reasoning)
- `traces/` — per-seed generation / rollout / quality-filter traces

"Validated" = survived dedup, was *challenging* (rollouts neither all-pass nor all-fail under
the global `doc_output_recall` threshold), and passed the quality filter. No logprobs / Tinker
artifacts are produced — this set is for **evaluating** the SearchAgent, not training it.

## Verification

- **Step 1 crop correctness**: on a known table page, confirm the rendered `fdoc[page_id]`
  (0-indexed) and the table crop (at `--bbox-dpi 300`) line up with the table — this validates
  the page-index and coord-scale handling that differ from the treasury data.
- **Step 2 parity** (do before a full embedding run): embed one element via OpenRouter and
  compare cosine vs a stored `qwen-v2` vector — expect ≈1.0 (confirms OpenRouter ↔
  SentenceTransformer parity for `qwen/qwen3-embedding-8b`, 4096-dim). Spot-check
  `metadata.json` `year`/`source` (e.g. `…transition__annrpt95` → 1995/`transition`;
  `govinfo_receipts__1893__…` → 1893/`govinfo_receipts`).
- **Step 3**: open `.chromadb-dais-slim` and confirm collection count + metadata keys
  (`doc_id, chunk_id, file_id, page_id, element_id, type, source, year, chroma:document`).
- **Step 5 (no Tinker)**: `python3 -c "import skunk.dais.datagen"` must succeed without the
  `tinker` package installed; smoke-test with `--num-new-seeds 2 --n-rollouts 2`.
