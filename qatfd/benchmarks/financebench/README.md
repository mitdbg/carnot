# financebench

## Tracked in git (small metadata)
- `financebench_splits.json` — dev/test split definition
- `README.md` — this file

## Not tracked — fetch / regenerate locally
A "complete" checkout of this directory also contains:

- `financebench_open_source.jsonl` — questions/answers corpus (from
  https://github.com/patronus-ai/financebench)
- `pdfs/` — ~368 source filings (10-K / 10-Q PDFs), e.g. `3M_2015_10K.pdf`,
  `ADOBE_2022_10K.pdf`, `AES_2016_10K.pdf`, …
- `financebench-element-embeddings/`
  - `metadata_rank0.json`, `metadata_rank1.json` — per-shard embedding metadata (2 shards)
- `chromadb/` — built Chroma vector index (`chroma.sqlite3` + one HNSW segment dir with `*.bin` / `index_metadata.pickle`)

These are large and are produced by the ingest/index pipeline; they are intentionally gitignored.
