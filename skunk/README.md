# Skunk

Grounded document QA tooling for the competition. **Maintained code:** `retrieval/` only.

## Layout

```text
skunk/
├── README.md
├── .venv/                 Python dependencies
├── _old/                  Archived vendored clones (officeqa, rlm)
└── retrieval/
    ├── run.sh             Offline + eval one-liner
    ├── README.md          Full retrieval docs
    ├── scripts/           Maintenance scripts
    └── skunk_retrieval/   Python package
```

## Quick start

```bash
cd skunk/retrieval
export OPENROUTER_API_KEY='...'
bash run.sh
```

See [retrieval/README.md](retrieval/README.md) for CLI, env vars, and new-corpus workflow.

## Cleanup status

| Area | Done |
|------|------|
| `retrieval/skunk_retrieval/` | Active pipeline; legacy BM25 in `_old/` |
| `skunk/_old/` | Vendored `officeqa/`, `rlm/` moved out of the way |
| `/tmp/officeqa` legacy | Run `retrieval/scripts/archive_legacy_tmp.sh` to move old bm25/results |
