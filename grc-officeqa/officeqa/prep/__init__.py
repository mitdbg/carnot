"""Offline OfficeQA corpus-prep scripts. NOT imported by the runtime path.

- page_cleaner.py: render PDF page → LLM reorders elements → write cleaned
  `.txt` per page; emits `clean_page_map.json`.
- compute_officeqa_element_embeddings.py: embed page elements (feeds the
  chroma build — see qatfd's create_vector_db script, `--benchmark officeqa`).
- table_corrector.py: LLM cleanup of parsed table blocks (bbox/render helpers
  also reused by the table-corrections viewer).
"""
