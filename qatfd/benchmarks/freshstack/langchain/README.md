# freshstack / langchain

Topic subdirectory of the freshstack benchmark. Only this README is tracked in
git; a "complete" checkout also contains (all gitignored):

- `corpus.jsonl` — document corpus for the langchain topic
- `queries.jsonl` — queries for the langchain topic
- `chromadb/` — built Chroma vector index (`chroma.sqlite3` + one HNSW segment dir with `*.bin` / `index_metadata.pickle`)
