"""officeqa — the Grounded Reasoning Cup / OfficeQA application layer over the skunk library.

Everything Treasury-Bulletin-specific lives here: the corpus accessors (`corpus`),
the app config (`config.SkunkConfig`), and the offline prep scripts (`prep/`). The
eval harness is a sibling directory (`../eval/`).

NOTE: the `page_index/` build pipeline (scans, continuation/notes structure, page
renders, `PageStore`) is now orphaned — it fed the retired `PageContentStore`/extract
seam and is no longer used at query time (retrieve → compute reads page text from the
search-agent corpus directly). It is kept for now pending its own removal.
"""
