# Corpus model: chunks, retrieval units, and source groups

How every benchmark's corpus maps onto the (skunk) SearchAgent's two-level vocabulary, and
what each ID means. This is the contract between the offline index builders
(`engaging-scripts/`), the Chroma collections, the benchmark loaders (`qatfd/benchmarks/`),
and skunk's retrieval tools (`skunk/src/skunk/search_agent/search_tools.py`).

## The three levels

Every corpus is modeled with (up to) three levels; adjacent levels may coincide.

1. **Chunk** — the atomic embedded unit; one Chroma row. The row id is the `chunk_id`
   (duplicated in metadata for `$nin` filterability); the `documents` column is the chunk's
   text; `element_id` is an int ordering the chunk within its retrieval unit (a 0-based
   index, or any monotonic stand-in such as a byte offset).
2. **Retrieval unit** — skunk's **"document"**, metadata key **`doc_id`**. The thing the
   SearchAgent reads whole (`read_document`, via the benchmark's `document_map`), tracks in
   its pruned/seen state, and returns in its final `{"doc_ids": [...]}` answer — and
   therefore the granularity `AnswerOutput.retrieved_doc_ids` and retrieval recall operate
   on. Chosen per corpus at index-build time by the `ElementAdapter`s in
   `engaging-scripts/create_vector_db.py`.
3. **Source group** — the natural parent of the retrieval unit (a monthly bulletin, an SEC
   filing). Never stored; always *derived* from the `doc_id` string or chunk metadata
   (helpers in `qatfd/keys.py`), and used only for coarser recall metrics.

**Invariant** tying the write side to the read side: the `doc_id` written into Chroma, the
keys of the benchmark's `document_map`, and the ID scheme its `recall_metrics` consumes must
all agree. Each benchmark's loader rebuilds its `document_map` from the same artifacts the
vectors were built from, so `read_document` returns exactly the text that was indexed.

## The six benchmarks

| Benchmark | Corpus | Chunk (embedded) | Retrieval unit (`doc_id`) | Source group | Recall metrics |
|---|---|---|---|---|---|
| OfficeQA | ~697 monthly US Treasury Bulletin PDFs (1939–2025), cleaned per page | page element (paragraph/table; headers/footers/figures dropped) | **page** `YYYY_MM_page` | monthly bulletin `YYYY_MM` (`keys.officeqa_doc`) | `page_recall` (unit), `doc_recall` (source) |
| BrowseComp-Plus | Tevatron curated web-document corpus (~100k docs / ~900k elements) | ~1024-token passage (merged/split by our code) | **web document** `docid` | = unit | `gold_doc_recall`, `evidence_doc_recall` (two gold sets, both unit-level) |
| TREC-BioGen | 26.8M PubMed abstracts (2025 Pyserini collection) | whole abstract (one chunk per doc) | **PMID** | = unit | `doc_recall` (unit) |
| FinanceBench | 368 SEC filing PDFs (10-K/10-Q/8-K/earnings) | page element (text/table/figure, LLM-extracted) | **page** `{doc_name}::p{n}` (zero-indexed; `keys.financebench_page_key`) | filing `doc_name` (`keys.financebench_doc`) | `page_recall` (unit), `doc_recall` (source) |
| QAMPARI | QAMPARI's released chunked Wikipedia, 25.9M ~100-word passages | released passage, embedded as-is; chunk_id `{page_id}__{n}`, element_id = `n` | **Wikipedia article** (numeric `page_id`) | = unit | `doc_recall` at the article level, joining on normalized TITLE (gold proofs cite title-slug URLs; `title` lives in each chunk's metadata) |
| FreshStack | Per-topic recent software docs (test = langchain, 49,514 chunks; dev = laravel, 52,351) | released corpus record, embedded as-is; chunk_id = corpus `_id` `{repo}/{path}_{start}_{end}`, element_id = start_byte | **source file** `file_id` (`keys.freshstack_file_id`) | = unit | `file_recall` (unit; gold `_id`s collapsed to files) |

Notes:

- **OfficeQA / FinanceBench deliberately retrieve at the page level** — their gold evidence
  is page-granular, and pages are the right reading unit for long filings/bulletins. The
  bulletin/filing is the source group, reported as the coarser `doc_recall`.
- **QAMPARI / FreshStack were promoted on 2026-07-09** from chunk-as-unit to article/file-as-
  unit (this made `read_document` return a real document instead of re-showing the ~100-word
  chunk search already surfaced, and made pruning exclude meaningful amounts of material).
  Existing collections are migrated **in place** by `scripts/migrate_doc_ids.py` — a
  metadata-only rewrite (`Collection.update`), no re-embedding or HNSW rebuild; the new
  `doc_id` (and FreshStack's `element_id`) derive from the row id itself, and the old
  `doc_id` equals the still-stored `chunk_id`, so it is reversible. Fresh builds get the new
  schema directly from the updated adapters in `engaging-scripts/create_vector_db.py`.
  **Recall numbers from runs before/after this date are not comparable for these two
  benchmarks** (QAMPARI's article-level `doc_recall` is defined identically, but the
  retrieval behavior changed; FreshStack's exact-chunk `doc_recall` no longer exists — a
  file-level `file_recall` is the primary metric, and exact-chunk recall would have to be
  recomputed offline from `traces/<qid>.jsonl` chunk_ids).
- **FreshStack's gold is chunk-granular** (a nugget's `relevant_corpus_ids` are corpus
  `_id`s). With files as the unit, `file_recall` counts a gold file as found when it was
  retrieved — more forgiving than exact-chunk recall. `recall_metrics` collapses the
  *retrieved* side through `file_id` too (a no-op on file ids), so the same code reproduces
  the historical `file_recall` when replayed over pre-migration report rows.
- **Recall metric names stay benchmark-specific** (`page_recall`, `file_recall`, ...) —
  deliberately not normalized to generic unit/source names; the table above is the mapping.

## Mapping a new corpus

1. Pick the retrieval unit: what should the agent read whole, prune, and cite? (Rule of
   thumb: the largest coherent unit that comfortably fits a context window and matches the
   granularity of the benchmark's gold labels — or is collapsible to it.)
2. Write a `compute_*_embeddings.py` that emits `embeddings_*.npz` + `metadata*.json`
   (`unique_element_id -> per-element metadata`), and an `ElementAdapter` in
   `create_vector_db.py` returning `(doc_id, chunk_text, extras)` with an ordering
   `element_id` in the extras. Register it in `BENCHMARK_ADAPTERS`.
3. Write the `Benchmark` subclass: `document_map` keyed by `doc_id` (rebuilt from the same
   artifacts the vectors came from — in RAM when small, lazily from Chroma when not), gold
   ids in the same scheme (or collapsible to it), and `recall_metrics` at whatever
   granularities are meaningful. Put any key-format helpers in `qatfd/keys.py`.

The one skunk-side term to keep straight: in the search layer, **"document" always means the
retrieval unit** (skunk is corpus-agnostic and never sees the source-group level). The
operator pipeline's `AnnotatedValue.source_stem` is a different identity — the source
document's filename stem — which is exactly why it is not called `doc_id`.
