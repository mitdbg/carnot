"""Page-index: an offline-built catalog + page store over the corpus.

The query/retrieval path that once read this artifact (`query.py`'s `PageIndexRetriever`)
has been removed; retrieval is now the search-agent backend (`skunk.search_agent`). What
remains here is the offline BUILD and the read-time page STORE that `extract.py` reads
through. Every catalog row is a single physical page (nothing is merged at build time).

Page store — `store.py`: the artifact's source of truth for page CONTENT, with two
  thread-safe access paths — `text(ref)` (read from `pages/<bulletin>.json`) and
  `image(ref)` (rendered on demand at 200 DPI, cached to `renders/`). The build writes each
  page's text (plus a figure note) under its page key, so the store stays a dumb
  page→text map. `extract.py` reads text and vision-tier images through `PageStore`,
  so the request path never touches the corpus parsed-JSON/PDFs directly.

Data model — `data_model.py`: the on-disk artifact schema (catalog row,
  era-keyed concept tree, filename layout) shared by build and the page store.

Build (offline) — `pipeline.py`: the end-to-end build pipeline
  (scan → continuation_check → notes_link → toc → reconstruct → place → catalog →
  page_store → era_merge), writing each artifact under one build folder. No page is ever
  merged: a header-less `is_continuation` page is resolved at READ time by fetching its
  predecessor chain (`continuation_chain`), and `continuation_check` only prunes over-long
  chains the scan mis-flagged. Domain logic lives alongside: `scan.py` (page scan),
  `continuation.py` (chain pruning), `notes_link.py` (data→footnote linking), `toc_index.py`
  (per-issue ToC extraction, reconstruction for ToC-less issues, placement), and `eras.py`
  (era segmentation + per-era canonical chapter build → the concept tree).
"""
