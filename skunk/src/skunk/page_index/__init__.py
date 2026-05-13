"""Page-index build and load utilities.

Offline pipeline that produces a per-page catalog over the Treasury Bulletin
corpus. Output lives at cache/page_index/{YYYY-MM}.jsonl, one row per page.
See ARCHITECTURE.md §"The page index lives in retrieve" for design intent.
"""
