"""Offline corpus-prep machinery for the search agent. NOT imported by the
runtime path.

- create_vector_db.py: the generic, resumable npz→ChromaDB loader
  (`run_build` + the `ElementAdapter` seam). Corpus-specific adapters and the
  CLI live with the benchmarks (qatfd `engaging-scripts/create_vector_db.py`;
  the OfficeQA embedding/cleaning scripts live in grc-officeqa/officeqa/prep/).
"""
