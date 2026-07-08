"""Offline corpus-prep scripts for the search agent.

NOT imported by the runtime path. These are run manually to produce
the artifacts the agent depends on (cleaned per-page text +
ChromaDB index):

- page_cleaner.py: render PDF page → LLM reorders elements → write
  cleaned `.txt` per page; emits `clean_page_map.json`.
- compute_officeqa_element_embeddings.py /
  compute_browsecomp_plus_element_embeddings.py: embed page elements.
- create_vector_db.py: load embeddings into ChromaDB.
- export_chroma_collection.py: dump a collection back to npz.
- table_corrector.py: LLM cleanup of parsed table blocks (bbox/render
  helpers also reused by dais/ and the table-corrections viewer).
- tinker_backend.py: Tinker sampling backend for the datagen RL harness.
"""
