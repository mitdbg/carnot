# qampari

## Tracked in git (small metadata)
- `qampari_splits.json` — dev/test split definition
- `README.md` — this file

## Not tracked — fetch / regenerate locally
A "complete" checkout of this directory also contains:

- `test_data.jsonl` — full test questions/answers corpus
- `dev_data_sample50.jsonl` — 50-question dev sample
- `chromadb/` — built Chroma vector index (`chroma.sqlite3`, a
  `.qwen-qampari-0.6b.built_npz.txt` marker, + one HNSW segment dir with
  `*.bin` / `index_metadata.pickle`)

These are large and are produced by the ingest/index pipeline; they are intentionally gitignored.
