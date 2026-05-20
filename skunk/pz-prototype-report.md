# Palimpzest-style semantic filter — prototype report

End-to-end retrieval pipeline with a two-stage Palimpzest-style
semantic cascade layered on top of the production page-index
retrieve path. Measured on 8 dev UIDs (the smallest post-date-filter
candidate sets that still contain in-index goldens), against the
final page-index build at `cache/page_index_final/`.

- **Eval scope**: 8 dev UIDs (test set excluded per `CLAUDE.md`)
- **Sample UIDs**: UID0097, UID0010, UID0009, UID0086, UID0201,
  UID0019, UID0123, UID0032 — picked by smallest post-date-filter
  candidate set among UIDs whose gold survived the prior page-index
  filter (so the filter actually has a recall-preservation target)
- **Retrieve model**: `google/gemini-3-flash-preview` (OpenRouter)
- **Filter model**: `google/gemini-2.5-flash` (OpenRouter)
- **Page-index build**: `cache/page_index_final/` (9 chapters /
  77,788 pages / 697 bulletins, built 2026-05-18)
- **Harness**: `eval/sem_filter_experiment.py`

## Pipeline

1. **L1 chapter pick** — one LLM call per `RetrieveBranch` against
   the concept-tree chapter list. Selects the most likely chapter
   given `(key, period)`.
2. **Date filter** — deterministic: keep page iff its `dates`
   field parses to ISO intervals that overlap the period's
   intervals. Pages with no parseable dates fall through (kept).
3. **Coarse semantic filter (Stage A)** — per-branch LLM call over
   batches of 20 pages. Each page sees only METADATA (titles,
   column headers, sampled row labels, dates, keywords). Bool-only
   output (no justification). Prompt requires precise subject /
   time / granularity match.
4. **Fine semantic filter (Stage B)** — per-branch LLM call over
   batches of 20 pages, run only on Stage-A survivors. Each page
   sees its FULL plain text plus metadata. Output requires a
   `value_quote` (literal substring of the page text showing a row
   + cell that anchors the answer), a one-sentence justification,
   then the boolean. Reject unless `value_quote` is non-empty and
   the cell matches the entity in `key` for `period` at the right
   granularity.
5. **Union across branches** — UID-level kept set is the union of
   per-branch fine survivors (mirrors production retrieve's branch
   union).

Coarse survivors are persisted per-branch in `results.jsonl` so the
fine prompt can be iterated via `--coarse-cache-from` without
re-paying for the coarse stage.

## Per-UID metrics

Numbers below are from `eval/reports/sem_filter/results.jsonl`.
`retr` is the post-date-filter (pre-semantic) candidate count;
`coarse` is post-Stage-A; `post` is post-Stage-B.

| UID | br | gold | retr | coarse | post | hits | recall | precision | $retr | $coarse | $fine | $total | wall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| UID0097 | 2 | 1 | 43  | 2  | **2**  | 1 | 100% | 50%  | $0.0011 | $0.0179 | $0.0039 | $0.023 | 9.2s |
| UID0010 | 1 | 1 | 60  | 5  | 3   | 0 | **0%** | 0% | $0.0005 | $0.0117 | $0.0046 | $0.017 | 5.7s |
| UID0009 | 2 | 1 | 85  | 4  | **2**  | 1 | 100% | 50%  | $0.0011 | $0.0282 | $0.0036 | $0.033 | 20.2s |
| UID0086 | 2 | 1 | 138 | 4  | 3   | 1 | 100% | 33%  | $0.0011 | $0.0545 | $0.0042 | $0.060 | 10.3s |
| UID0201 | 2 | 1 | 147 | 11 | **1**  | 1 | 100% | **100%** | $0.0011 | $0.0562 | $0.0191 | $0.076 | 14.7s |
| UID0019 | 4 | 2 | 149 | 35 | 8   | 2 | 100% | 25%  | $0.0021 | $0.1201 | $0.0584 | $0.181 | 37.0s |
| UID0123 | 2 | 1 | 165 | 13 | 13  | 1 | 100% | **8%**   | $0.0011 | $0.0612 | $0.0273 | $0.090 | 21.0s |
| UID0032 | 2 | 1 | 197 | 10 | 10  | 1 | 100% | **10%**  | $0.0011 | $0.0696 | $0.0329 | $0.104 | 19.1s |
| **TOTAL** | **17** | **9** | **984** | **84** | **42** | **8** | **88.9%** | **19.0%** | **$0.0090** | **$0.4193** | **$0.1540** | **$0.582** | **137.2 s** |

## Headline aggregates

| metric | value |
|---|---|
| Micro-recall (sum_hits / sum_in_idx_gold) | **88.9%** (8 / 9) |
| Micro-precision (sum_hits / sum_final) | **19.0%** (8 / 42) |
| Macro-recall (mean per-UID) | 87.5% |
| Macro-precision (mean per-UID) | 34.5% |
| Mean candidate set per UID (post-date-filter → final) | **123 → 5.2** pages |
| Total cost across 8 UIDs | **$0.582** (~$0.073/UID) |
| Cost share | retrieve $0.009 (1.6%) · coarse $0.419 (72.1%) · fine $0.154 (26.4%) |
| Mean wall-clock per UID (sequential) | 17.2 s |

## Observations

- **Recall is bounded by the coarse stage.** The single recall
  failure (UID0010) is dropped in coarse, before fine sees the
  page. Fine is recall-preserving on the 8 UIDs.
  - UID0010 asks for "Foreign Exchange and Securities investments
    in Japanese Yen, March 31, 2025." Gold is page `(2025-06, 76)`,
    titled "TABLE ESF-1—Balances …". The catalog's
    `row_labels_sample` for that page only contains the first 12
    row labels and the "Japanese yen" row sits further down — so
    the coarse stage's metadata-only view doesn't surface the
    entity, and the strict coarse prompt rejects on subject
    mismatch.

- **Precision is dragged down by two outliers** (UID0123 and
  UID0032 — both customs / tariff queries). On those UIDs the
  fine stage kept every coarse survivor (13/13 and 10/10), giving
  precision of 8% and 10% respectively. These are the natural
  targets for the next round of fine-prompt iteration; coarse
  survivors are cached.

- **Per-UID cost is dominated by coarse (72%)**. Coarse uses
  page metadata but ships every survivor through (mean ~123
  pages → ~8 batches of 20), whereas fine only sees Stage-A
  survivors (mean ~10 → 1 batch). The atypical economics here:
  the *cheap* stage runs on more pages but its per-batch
  prompt is dense enough that it still dominates spend.

- **Fine stage is materially filtering on 4 of 8 UIDs**:
  UID0201 (11→1), UID0019 (35→8), UID0086 (4→3), UID0009 (4→2).
  On UID0097 the coarse stage already cut to 2 so fine had
  nothing more to do. On UID0123 / UID0032 fine accepted
  everything — the prompt isn't tight enough for customs-table
  pages where multiple bulletin months print closely-related
  data.

- **Retrieve cost is negligible** at this scale (~1.6% of total
  spend). The chapter-pick LLM call averages ~$0.0005 per branch.

## The two prompts

### Coarse (Stage A) — metadata only, bool output

```
You decide, for EACH page in a batch, whether the page reports
values for a SPECIFIC retrieve target: a `key` (the data concept) and
a `period` (the time window). Each batch is up to ~hundreds of
candidate pages from the U.S. Treasury Bulletin.

Input shape (the user message):
  - `key`: short NL phrase describing the data to find
    (e.g. "national defense expenditures",
     "weekly average discount rate for new 91-day bills").
  - `period`: time window — `CY1940`, `FY1991..FY1995`, `2007-01-01`,
    etc. May also be a comma-enumerated list of points.
  - `pages`: a JSON array, each entry
    {"id": <int>, "bulletin": "<YYYY-MM>", "page": <int>,
     "titles": [...], "column_headers": [...],
     "row_labels_sample": [...], "dates": [...], "keywords": [...]}.

Output: a SINGLE JSON object (no prose, no fences, no justification):

  {"decisions": [{"id": <int>, "relevant": <true|false>}, ...]}

One entry per input page, in the same order. Use the input `id`.
Do NOT include a `reason` field — just the boolean.

## Decision rule — be PRECISE

Default is **relevant=false**. Mark `true` only when the page's
column headers, row labels, title, AND time signal together indicate
that the page reports the named `key` for a time inside (or that is
a plausible retrospective reporting issue for) `period`.

Check three things per page:

1. **Subject match.** The page's title / column headers / keywords
   must name the SUBJECT in `key`. Topic-adjacent is not enough —
   "Capital movements" is different from "Foreign currency
   positions" even though both involve money flowing between
   countries. If `key` names a specific security, program, or
   account, the page must mention that specific entity.

2. **Time match.** The page's `dates` field (or its bulletin month
   for retrospective tables) must intersect `period`. A page whose
   only dates are years before / after `period` is not relevant
   unless its bulletin month is the conventional issue that reports
   data for `period` (typically the month immediately after the
   period closes).

3. **Granularity match.** A monthly table cannot answer a daily-
   resolution `key`; a country-level aggregate cannot answer a state-
   level `key`; an annual snapshot cannot answer a weekly-series
   question. If the page reports a coarser or finer breakdown than
   `key` implies, reject.

Always reject pages that are tables of contents, chapter covers,
indexes, introductory prose, or pure boilerplate (no numeric tables).
```

### Fine (Stage B) — full page text, value_quote then bool

```
You are STAGE B of a two-stage cascading filter. Stage A pruned
obvious metadata mismatches; you now see each page's full plain text
plus structural anchors. Your job is to emit `relevant=true` ONLY
when you can point at a concrete numeric value or row in the page
text that IS the answer to the question, or IS one of the input
values the answer is built from.

The downstream extractor will read the page next. We want the final
candidate set to be ≤ 2× the size of the actual gold set per query.
If you cannot quote a specific value cell from the page text that
matches the retrieve target, REJECT.

Input (user message):
  - `key`: short NL phrase describing the data to find
    (e.g. "adjusted price for 2-3/8% U.S Treasury Inflation-Protected
    Security").
  - `period`: time window — `CY1940`, `FY1991..FY1995`, `2007-01-01`,
    or a comma-enumerated list.
  - `pages`: JSON array, each entry
    {"id": <int>, "bulletin": "<YYYY-MM>", "page": <int>,
     "titles": [...], "column_headers": [...], "dates": [...],
     "text": "<page plain text, may be truncated>"}.

Output: a SINGLE JSON object (no prose, no fences):

  {"decisions": [
     {"id": <int>,
      "value_quote": "<EXACT substring you copied from the page
                      `text` showing the value cell: the row label
                      + the value, e.g. 'Japanese yen ... 12,345' or
                      '2-3/8% TIPS—01/15/17-A ... 99.342280'. Empty
                      string if no such cell exists.>",
      "justification": "<one sentence, ≤25 words, explaining how
                        `value_quote` ties to `key` + `period`.>",
      "relevant": <true|false>},
     ...
  ]}

ORDER MATTERS inside each decision: emit `value_quote`, then
`justification`, then `relevant`.

## Decision rule — REJECT by default

Mark `true` ONLY when ALL of the following hold:

1. **`value_quote` is non-empty AND is a real substring of the
   page `text`.** You must literally copy a row + cell from the
   text. No paraphrasing. No "the page contains values like ...".
   If you cannot find such a substring, `value_quote=""` and
   `relevant=false`.

2. **The quoted cell is for the SPECIFIC entity in `key`.** Same
   currency, same country, same program, same security, same fund,
   same rate series. Topic-adjacent does not qualify ("Federal
   Disability Insurance" ≠ "Old-Age and Survivors Insurance";
   "Japanese yen" ≠ "Swiss franc"; "Capital movements" ≠ "Foreign
   currency positions" unless the row itself shows the specific
   entity).

3. **The cell's time signal matches `period`.** A date next to the
   value, the column header, the row's reporting date, the
   bulletin month for retrospective tables — any of these must
   land inside `period`. A page that has the right subject but
   only for a different time window must be rejected.

4. **The granularity matches `key`.** Annual roll-ups cannot
   answer monthly questions; country aggregates cannot answer
   line-item questions. If the cell is at a coarser or finer
   resolution than `key` asks for, reject.

If any one of (1)–(4) fails, `relevant=false`.

Pages whose text is a table of contents, chapter cover, section
index, front-matter prose, or boilerplate are ALWAYS rejected with
`value_quote=""`.

## Calibration

You should reject the vast majority of pages. A typical batch of
20 will likely yield 0–2 `relevant=true`. If you find yourself
marking >5 of 20 as true, re-read your `value_quote`s — most
likely you're keeping topic-adjacent pages whose quoted cell isn't
actually for the entity in `key`.
```

## Artifacts

- **Harness**: `eval/sem_filter_experiment.py`
- **Raw per-UID results**: `eval/reports/sem_filter/results.jsonl`
  (includes `coarse_survivors` per branch for follow-up runs)
- **Per-batch logs**: `eval/reports/sem_filter/run.log`
- **Replay command** (skip coarse, re-run fine only with the same
  per-branch coarse survivors):
  ```
  python -m eval.sem_filter_experiment \
      --uids UID0097,UID0010,UID0009,UID0086,UID0201,UID0019,UID0123,UID0032 \
      --coarse-cache-from eval/reports/sem_filter/results.jsonl \
      --filter-workers 32 --batch-size 20 --max-page-chars 12000 \
      --output-dir eval/reports/sem_filter_v2
  ```

## What's next

- **Tighten fine prompt against UID0123 / UID0032** customs cases.
  Both have multiple bulletin months printing the same table; the
  current prompt's three checks all pass, so the LLM keeps
  everything. A fourth check that forces the value_quote to belong
  to a row whose date matches `period` exactly (not just the
  bulletin month) should help.
- **Coarse stage cost optimization**. Coarse is 72% of total spend
  but its per-page work is purely metadata classification. A
  smaller / cheaper model for coarse (e.g. Gemini Haiku) would
  cut total cost ~half with limited recall risk; coarse already
  rejects only on subject/time/granularity, and Haiku handles
  those signals well.
- **Address UID0010 (Japanese-yen-via-ESF) recall miss**. Either
  surface more catalog metadata to coarse (bigger
  `row_labels_sample` truncation), or let coarse fall back to
  full-text on low-confidence rejects.
