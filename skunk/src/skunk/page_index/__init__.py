"""Page-index: an offline-built catalog + page store + the retriever that queries them.

Query (per-query) — `query.py`: the `PageIndexRetriever` (loads the
  artifact, then ToC pick → year filter → semantic filter). Every catalog row is a
  single physical page (nothing is merged at build time), so a kept row maps
  straight to the page extract reads / the vision tier renders.

Page store — `store.py`: the artifact's source of truth for page CONTENT, with two
  thread-safe access paths — `text(ref)` (read from `pages/<bulletin>.json`) and
  `image(ref)` (rendered on demand at 200 DPI, cached to `renders/`). The build writes each
  page's text (plus a figure note) under its page key, so the store stays a dumb
  page→text map. `extract.py` reads text and vision-tier images through `PageStore`,
  so the request path never touches the corpus parsed-JSON/PDFs directly.

Data model — `data_model.py`: the on-disk artifact schema (catalog row,
  era-keyed concept tree, filename layout) shared by build and query.

Build (offline) — `pipeline.py`: the end-to-end build pipeline
  (scan → toc → reconstruct → place → catalog → page_store → era_merge), writing
  each artifact under one build folder. There is deliberately no continuation-merge
  pass — the scan's `is_continuation` flag is metadata only (see the pipeline module
  docstring for why). Domain logic lives alongside: `scan.py` (page scan),
  `toc_index.py` (per-issue ToC extraction, reconstruction for ToC-less issues,
  placement), and `eras.py` (era segmentation + per-era canonical chapter build →
  the concept tree).
"""
