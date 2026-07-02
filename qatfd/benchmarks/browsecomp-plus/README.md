# browsecomp-plus

## Tracked in git (small metadata)
- `browsecomp_plus_splits.json` — dev/test split definition
- `prompts.yaml` — benchmark prompt templates
- `README.md` — this file

## Not tracked — fetch / regenerate locally
A "complete" checkout of this directory also contains:

- `browsecomp_plus_decrypted.jsonl` — decrypted questions/answers corpus
- `browsecomp-plus-element-embeddings/`
  - `metadata_rank0.json` … `metadata_rank3.json` — per-shard embedding metadata (4 shards)
- `chromadb/` — built Chroma vector index (`chroma.sqlite3` + one HNSW segment dir with `*.bin` / `index_metadata.pickle`)

These are large and are produced by the ingest/index pipeline; they are intentionally gitignored.
