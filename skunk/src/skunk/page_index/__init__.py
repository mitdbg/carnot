"""Page-index: an offline-built catalog + the retriever that queries it.

A flat set of modules for one corpus (Treasury Bulletin):

  Build (offline) — `pipeline.py` sequences the phases: `catalog.py` (parse),
    `summarize.py` (LLM enrich), then `l1_harvest.py` → `placer.py` → `merger.py`
    (chapters), with `text_norm.py` shared by the last two. Produces the artifact
    under `artifact/page_index_old/` (slim `catalog/` + `concept_tree.json`).

  Query (per-query) — `query.py`: the `PageIndexRetriever` (loads the
    artifact, then ToC pick → year filter → semantic filter).

  Shared — `data_model.py` (catalog row, concept tree, artifact layout),
    `period.py` (period grammar).
"""
