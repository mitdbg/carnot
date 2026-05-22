"""Offline corpus-prep scripts for the search agent.

NOT imported by the runtime path. These are run manually to produce
the artifacts the agent depends on (cleaned per-page text +
ChromaDB index):

- page_cleaner.py: render PDF page → LLM reorders elements → write
  cleaned `.txt` per page; emits `clean_page_map.json`.
- compute_element_embeddings.py: embed page elements.
- create_vector_db.py: load embeddings into ChromaDB.
- exp_vector_db_recall.py: recall sanity experiment.
- harness.py: teammate's standalone CLI harness (superseded by
  skunk's `eval/eval_e2e.py` at runtime).
"""
