# officeqa

## Tracked in git (small metadata)
- `officeqa_splits.json` — dev/test split definition
- `prompts.yaml` — benchmark prompt templates
- `README.md` — this file

## Not tracked — fetch / regenerate locally
A "complete" checkout of this directory also contains:

- `officeqa_pro.csv` — questions/answers table
- `treasury_bulletin_pdfs/` — ~697 source Treasury Bulletin PDFs
- `treasury_bulletins_cleaned/` — cleaned/normalized bulletin text
- `chromadb/` — built Chroma vector index (`chroma.sqlite3` + one HNSW segment dir with `*.bin` / `index_metadata.pickle`)

These are large and are produced by the ingest/index pipeline; they are intentionally gitignored.
