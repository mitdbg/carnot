# Skunk TODO
- Human-in-the-loop for visual-understanding (chart-read) questions. We should write a mechanism that has the LLM identify visual understanding questions, and then surfaces a popup for the user to answer
- Data page cleaning -- define a pre-competition pipeline that cleans and prepares the data for a corpus before competition
- Clean up PageIndex build and search process, possibly use LLM-extracted dates on date mask for increased quality.
- Integrate PageIndex + sem_filter retrieval agent
- Investigate: revisions and how to deal with them (with sem filter this should actually be easy)
- Critique system redesign: previous critique system was removed because they caused failures over pedantic issues but did not catch meaningful bugs. We should probably come up with new critique subsystem.
- Integrate with Databricks grading harness
- PageIndex retrieve era pruning (latency): `_eras_for_branch` now returns ALL eras (for
  publication skew + revisions), which costs one ToC pick per era (~9× fan-out). Safe asymmetric
  narrowing: skip eras whose span ENDS before the branch period/as_of starts — an era published
  entirely before a statistic's date cannot report that (later) data, while current+later eras
  must stay (skew forward + revisions restate old data in later issues). Keeps the recall win,
  cuts the impossible-era picks.
- PageIndex retrieve `as_of` shortcut: a branch with `as_of` set pins a single bulletin (an
  as-of/point-in-time snapshot). The retriever currently still runs the full parallel per-era ToC
  pick and then year-filters on publication month (`ref.month`). Instead, when `as_of` is set,
  skip ToC pick entirely and just return all of that bulletin's pages directly (it's one issue —
  the picker adds latency/cost and risk for no narrowing benefit). Saves ~9 ToC-pick LLM calls per
  as_of branch.
- PageIndex stray buckets: the per-era chapter finalize (`eras.py` `_finalize_era`) leaves a tail of tiny singleton buckets — OCR garble or form/front-matter the LLM skipped (~0.1–0.6% of corpus pages), effectively unreachable by the picker. Either fold each into the nearest canonical chapter (string/LLM similarity), or route them all to one corpus-wide look-aside bucket the picker always scans, so no page is orphaned.
- PageIndex build: single cross-stage render queue. Page rasterization is CPU-bound (PyMuPDF, GIL-serialized), and today the `prerender` stage only starts after the whole `scan` stage finishes. Instead, run one render queue (a worker pool draining a shared queue) that enqueues a page the moment the text `scan` flags it (`has_unparsed_graphics` / `parse_broken`), so rendering into the `renders/` cache overlaps the still-running scan LLM calls — images are warm by the time `vision_rescan` reads them. Pipelines the CPU-bound render against the I/O-bound scan instead of running them as serial stages.
- LLM per-request timeout for runaway Gemini "thinking". A single call occasionally runs its thinking trace away for minutes (a known Gemini soft-limit issue — the API has no mid-thought interrupt and no max-thinking-time; `thinking_level`/`thinking_budget` are soft and can overflow). With `asyncio.gather` over a batch (e.g. the planner replanning 101 UIDs), one stuck call blocks the WHOLE batch — observed as UID0149 hanging the v3 replan at 100/101 (and the earlier opaque 6-min "stall"). `make_genai_client` (`common.py`) builds `genai.Client(api_key=...)` with NO request timeout, so a runaway/hung request never raises → never hits the retry path. Fix: set a per-request timeout (`genai.Client(http_options=types.HttpOptions(timeout=ms))`, env-configurable e.g. `SKUNK_LLM_TIMEOUT_S`, ~120–180s) so a runaway raises `TimeoutException` → already caught by `_is_retryable` → retried (runaway is sporadic, so the retry completes normally). Secondary: consider dropping the planner `effort` medium→low to lower runaway probability (soft, and verify it keeps the as_of/period wins). Added progress prints to `eval_retrieve._build_plans` so "slow vs stuck" is now visible.

