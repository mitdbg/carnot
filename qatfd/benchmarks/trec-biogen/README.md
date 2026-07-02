# trec-biogen

## Tracked in git (small metadata)
- `trec_biogen_splits.json` — dev/test split definition
- `README.md` — this file

## Not tracked — fetch / regenerate locally
A "complete" checkout of this directory also contains:

- `2025_task_a.json` — TREC BioGen 2025 Task A topics/questions
- `baseline_labels.json` — baseline relevance labels
- `chromadb/` — built Chroma vector index, split across four rounds:
  - `r0/`, `r1/`, `r2/`, `r3/`, each with `chroma.sqlite3`, a `.qwen-biogen-0.6b_r*.built_npz.txt` marker, and one HNSW segment dir (`*.bin` / `index_metadata.pickle`)

These are large and are produced by the ingest/index pipeline; they are intentionally gitignored.
