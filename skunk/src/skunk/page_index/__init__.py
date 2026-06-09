"""Page-index: an offline-built catalog + page store + the retriever that queries them.

Query (per-query) — `query.py`: the `PageIndexRetriever` (loads the
  artifact, then ToC pick → year filter → semantic filter), expanding each kept
  anchor into its member refs (anchor + folded continuation run) so extract reads
  the merged text once and the vision tier renders every page of the table.

Page store — `store.py`: the artifact's source of truth for page CONTENT, with two
  thread-safe access paths — `text(ref)` (read from `pages/<bulletin>.json`) and
  `image(ref)` (rendered on demand at 200 DPI, cached to `renders/`). The build writes each
  anchor's already-merged text (its + folded continuation pages' text, plus a figure note)
  under the anchor's page key, so the store stays a dumb page→text map. `extract.py` reads
  text and vision-tier images through `PageStore`, so the request path never touches the
  corpus parsed-JSON/PDFs directly.

Data model — `data_model.py`: the on-disk artifact schema (catalog row,
  era-keyed concept tree, filename layout) shared by build and query.

Build (offline) — `pipeline.py`: the end-to-end build pipeline
  (scan → merge_continuations → toc → reconstruct → place → catalog → page_store →
  era_merge), writing each artifact under one build folder. Domain logic lives
  alongside: `scan.py` (page scan + `merge_continuations`, which folds each
  continuation page into the previous content page), `toc_index.py` (per-issue
  ToC extraction, reconstruction for ToC-less issues, placement), and `eras.py`
  (era segmentation + per-era canonical chapter build → the concept tree).
"""
