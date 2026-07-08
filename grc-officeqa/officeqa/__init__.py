"""officeqa — the Grounded Reasoning Cup / OfficeQA application layer over the skunk library.

Everything Treasury-Bulletin-specific lives here: the corpus accessors (`corpus`),
the page-index build pipeline and page store (`page_index`), the skunk
`PageContentStore` implementation (`page_store`), the app config
(`config.SkunkConfig`), and the offline prep scripts (`prep/`). The eval harness
is a sibling directory (`../eval/`).
"""
