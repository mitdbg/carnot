# BM25 rerank on page-index retrieval — negative result

**Date**: 2026-05-17
**Eval**: `eval/eval_retrieve.py` on dev (101 UIDs, held-out 32 excluded per `eval/test_set_uids.json`)
**Model**: `gemini-3-flash-preview` via direct Gemini API (`SKUNK_USE_DIRECT_GEMINI=1`)
**Catalog**: `cache/page_index_v4/catalog/` (697 bulletins, Stage-1 rebuild), tree at `data/page_index/concept_tree.json`

## TL;DR

Hybrid BM25 with dominance-bypass at the recommended `threshold=2.0` cost **3.5 pp recall** for only **4 % page reduction**. Stricter thresholds (5.0) never fire bypass at all. There is no setting between these that produces a useful trade-off on this corpus. **Recommendation: keep the scaffold disabled (`bm25_enabled=False`) and ship nothing.** The two new modules can be deleted in ~30 LoC; the wiring is all flag-guarded.

## Numbers

| Run | Micro-recall | Mean pages/UID | Wall | Bypass fires |
|---|---|---|---|---|
| Baseline (`chapter-year`) | **96.0 %** | 3,817 | 318 s | n/a |
| BM25 `top_k=20 threshold=2.0` | **92.5 %** (−3.5 pp) | 3,659 (−4 %) | 29 s* | 5 / 101 UIDs (5 %) |
| BM25 `top_k=20 threshold=5.0` | 96.0 % | 3,817 | 28 s* | 0 / 101 UIDs |

*BM25 runs reused the cached plans from the baseline (`cache/retrieve_bench_plans.jsonl`), so they skip the planner LLM call.

## Diagnosis: why dominance-bypass fails

Bypass fired on 5 UIDs at threshold=2.0. **4 of those 5** lost the golden page(s) outright:

| UID | branch chapter | year-filtered | top1 / median20 | golden BM25 rank | catastrophic? |
|---|---|---|---|---|---|
| UID0056 | Profile of the Economy | 2040 | 49.8 / 15.7 (3.18) | golden score **0.0** (rank None / 1297) | ✗ |
| UID0101 | Federal Fiscal Operations | 503, 487 | 44.2 / 12.6 (3.51) | golden ranks **3611, 3624** / 9656 | ✗ |
| UID0113 | Federal Debt | 5300, 4877 | 32.3 / 12.8 (2.54) | golden rank **11757** / 23889 | ✗ |
| UID0136 | Federal Debt | 7647 | 51.4 / 24.3 (2.12) | golden ranks **41, None, 118** / 25071 | ✗ (2 of 3 lost) |
| UID0114 | (preserved) | 3075 | (in bypass band) | golden in top-K | ✓ |

The other ~0.5 pp recall delta is **LLM nondeterminism**, not BM25: UID0058 had a different L1 chapter pick across baseline and BM25 runs at temperature 0.0, producing **disjoint** candidate sets (721 base-only vs 876 bm-only). Cross-run A/B noise; not attributable to either retriever.

## Why BM25 picks the wrong page

BM25 picks pages whose **title / column-headers / row-headers** lexically match the question. On Treasury Bulletin, the right answer often lives somewhere else:

- **Retrospective summaries** in a later bulletin (`UID0056`: question asks 1950–1990, answer table is in **1991-09 p.30**; BM25 went for `1992-03 p.38` which has a distinct "saving rate" caption but is the wrong reporting period).
- **Annual roundups / cumulative tables** whose titles are general ("Selected Economic Indicators", "Statement of Public Debt") and whose column headers happen to match — these score poorly on BM25 because the lexically-distinctive table on a different page outscores them.
- **Footnote / breakout tables** with terse titles (`UID0136`: October bulletins reporting on September 91-day bill auctions don't have "91-day" prominently in their titles — BM25 found mid-1960s bulletins with stronger lexical match instead).

This is the same failure mode that hurt embeddings (`data/page_index/README.md`), arrived at from the other direction. Embeddings fail because of semantic clustering: similar tables collide. BM25 fails because the *lexically obvious* page is often not the answer page — the answer lives in a less prominent table whose discriminating tokens are in the row/column structure, not the title. Year-filter survives because it uses `(min_year, max_year)` parsed from page content, which catches retrospective tables that lexical / semantic scoring can't.

## Conclusion

The dominance-bypass rule is structurally incompatible with this corpus. There is no threshold in the `[2.0, 5.0]` band that buys cost reduction without catastrophic recall loss; outside that band bypass either never fires (no-op) or fires too liberally (worse). Reordering without truncation is a no-op against the eval metric.

## Follow-up: BM25 as a pure top-K hard-cap (no bypass)

Run via `eval/bm25_topk_sweep.py` (post-hoc K sweep from one LLM L1 pick per branch — no per-K LLM cost, no per-K nondeterminism). Combined across branches per UID:

```
cut     micro_recall    macro_recall    mean_pred/UID
all              96.0 %  -               3817            ← baseline
BM25∞            95.0 %  -               -               ← BM25 ceiling (every page with score > 0)
K=100            35.8 %  39.3 %          72
K=250            54.2 %  56.6 %          172
K=500            71.1 %  74.5 %          329
K=1000           81.6 %  83.1 %          615
```

Two observations:

1. **BM25-∞ ceiling = 95.0 %.** Two of 201 dev goldens (UID0056 `1991-09 p.30`, UID0136 `1954-10 p.11`) have BM25 score = 0 across the year-filtered chapter — their indexed text contains zero tokens from the question. No K can recover them. So even an infinite BM25 budget caps at 95.0 % recall, 1 pp below the no-rerank baseline (96.0 %).

2. **The K curve is shallow.** At K=1000 (6× compression from baseline) we still lose 14 pp recall. To match the baseline 96 % we'd need K well above 3000 — which is most of the candidate set. Sample failure UIDs:
   - UID0101 — golden ranks 3611 and 3624 of 9656; need K ≈ 3700 to recover.
   - UID0113 — golden rank 11757 of 23889; need K ≈ 12000.
   - UID0136 — 3 goldens at ranks 41, ∅, 118; one unrecoverable.

For comparison against the prior embedding finding (`data/page_index/README.md`: *"subtractive at any K ≤ 1000 against this corpus + blob shape (cost 3 pp recall to halve page count)"*): at K=1000 BM25 loses **14 pp** while compressing ~6×; embeddings lost ~3 pp at ~2× compression. **BM25 is strictly worse than embeddings at comparable cost reduction.**

The user's hypothesis going in — "embeddings fail because keywords are semantically clustered; BM25's lexical scoring should sidestep that" — turns out to be inverted on this corpus. The failure mode is not semantic clustering between similar tables; it is that **golden pages frequently do not lexically match the question** at the title/header/keyword level. The right table lives in a retrospective summary, annual roundup, or generically-titled cross-section whose surface text uses different terminology than the question. Embeddings recover some of this via semantic generalization; BM25 cannot.

## Removal recipe

If kept disabled long-term, delete the scaffold in one PR:

1. `rm src/skunk/page_index/bm25.py src/skunk/page_index/bm25_runtime.py`
2. In `src/skunk/retrieve.py`: drop the BM25 import, `_bm25_cache`, `_bm25_chapter_lock`, `_get_bm25_index`, and the `if ctx.config.bm25_enabled:` block.
3. In `src/skunk/config.py`: drop `bm25_enabled`, `bm25_top_k`, `bm25_dominance_threshold` (and their `SKUNK_BM25_*` env vars).
4. In `src/skunk/page_index/retrieve_probe.py`: `picked_chapters` field is harmless; can stay.
5. In `eval/eval_retrieve.py`: drop the `chapter-year-bm25` choice, the `_bm25_get` helper, and the `--bm25-*` CLI flags.

The direct-Gemini fallback in `src/skunk/common.py` (`_call_gemini_direct` + `use_direct_gemini` config) is independent of BM25 and worth keeping — it's how this experiment ran without OpenRouter credits.

## What might work next (out of scope here)

- **No bypass, hard cap** (e.g. always return BM25 top-200). Doesn't depend on a confidence signal; trades recall for cost in a predictable way. Would need a recall/K curve to find the knee.
- **Reranker that scores row/column headers separately**, with a question-aware weight, so retrospective tables with generic titles but matching column structure aren't outscored by lexically-named topic tables.
- **Decoupled extract pre-filter**: keep retrieve's recall ceiling, push BM25/heuristics into `extract` to decide which of the 1,800 candidate pages to render first. The downstream `extract` operator can short-circuit on success, so an ordering signal there pays off without needing to truncate.
