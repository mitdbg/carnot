# freshstack

## Tracked in git (small metadata)
- `freshstack_splits.json` — dev/test split definition
- `README.md` — this file
- `langchain/README.md`, `laravel/README.md` — per-topic layouts

## Structure
This benchmark is split by topic, one subdirectory each:
- `langchain/`
- `laravel/`

## Not tracked — fetch / regenerate locally
Each topic subdirectory contains (all gitignored):
- `corpus.jsonl` — document corpus for the topic
- `queries.jsonl` — queries for the topic
- `chromadb/` — built Chroma vector index (`chroma.sqlite3` + one HNSW segment dir with `*.bin` / `index_metadata.pickle`)

These are large and are produced by the ingest/index pipeline; they are intentionally gitignored.
