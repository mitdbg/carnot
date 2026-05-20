# Compute experiment — full dev-set eval (unified trace, refreshed)

**Trace source**: `eval/traces/aggregated_20260519/` — 101 dev-set UIDs (test-set filtered). The aggregated directory keeps the **last** trace seen for each UID across the baseline + lookup-fix + replan-rescue passes (latest reroll wins).
**Run window**: 2026-05-18 → 2026-05-19. Newest traces in the directory land at 14:58 on 2026-05-19.
**Model**: `google/gemini-3-flash-preview` via OpenRouter for every non-search call; direct Gemini API `gemini-3-flash-preview` for the Google-Search-grounded `lookup_external` path.

---

## Grading methodology

For each UID, the predicted answer is compared to the gold answer by **scale-tolerant relative difference**:

1. Extract numbers from both pred and gold (first-number for scalars, element-wise for `[a, b]` lists).
2. Compute the relative %Δ across a set of scale factors `k ∈ {0, ±1, ±2, ±3, ±6, ±9, ±12}` — i.e. `min_k |pred · 10^k − gold| / |gold|`. The best `k` and resulting %Δ are reported.
3. This is the "millions of dollars" tolerance: if the gold is `57,615.04` (implicitly millions) and pred is `57,615,040,000`, the `k = −6` match drops the Δ to 0% and the answer is graded exact. Unicode minus is normalized; sign-flip is detected separately when magnitudes match within 0.5% but signs differ.

Buckets (cumulative; lower buckets count toward higher ones):

| Bucket | Definition |
|---|---|
| **Exact** | scale-normalized Δ = 0.0% |
| **Within 0.1%** | Δ ≤ 0.1% |
| **Within 1.0%** | Δ ≤ 1.0% |
| **Within 5.0%** | Δ ≤ 5.0% |

Outside those: `wrong_sign` (magnitude match, sign flipped), `wrong` (Δ > 5%), `no_answer` (pred is None or non-numeric).

In this trace, scale-tolerance doesn't flip any UID — UID0228, the previously-flagged 10⁶× case, is now emitting `57,615.04` directly (compute fix landed in this batch). The methodology is in place so future scale shifts grade correctly.

---

## Numerical accuracy

Cumulative buckets (101 UIDs):

| Threshold | Count | Share of 101 | Share of answered (92) |
|---|---:|---:|---:|
| **Exact (Δ = 0.0%)** | **58** | 57.4% | 63.0% |
| **Within 0.1%** | **61** | 60.4% | 66.3% |
| **Within 1.0%** | **64** | 63.4% | 69.6% |
| **Within 5.0%** | **70** | 69.3% | 76.1% |

Exclusive buckets (each row counts UIDs in that band only):

| Bucket | Count | Share of 101 |
|---|---:|---:|
| Exact (Δ = 0.0%) | 58 | 57.4% |
| (0%, 0.1%] | 3 | 3.0% |
| (0.1%, 1.0%] | 3 | 3.0% |
| (1.0%, 5.0%] | 6 | 5.9% |
| Wrong-sign | 1 | 1.0% |
| Wrong (>5%) | 21 | 20.8% |
| No answer | 9 | 8.9% |
| **Total** | **101** | |

**Headline**: **64/101 = 63.4% within 1%** and **70/101 = 69.3% within 5%** on the dev set. 58 UIDs (57.4%) match the gold exactly (after scale normalization); only 3 more squeeze in under 0.1%, then a steeper climb of 6 UIDs between 1% and 5%.

### UIDs in each bucket

**Within 0.1% but not exact** (3 UIDs):
- UID0010 (Δ=0.0013%), UID0214 (Δ=0.0303%), UID0225 (Δ=0.0794%)

**Within 1.0% but not within 0.1%** (3 UIDs):
- UID0007 (Δ=0.106%), UID0084 (Δ=0.129%), UID0226 (Δ=0.129%)

**Within 5.0% but not within 1.0%** (6 UIDs):

| UID | Pred | Gold | Δ% |
|---|---|---|---:|
| UID0018 | `80.378` | `81.406` | 1.26 |
| UID0027 | `2960` | `3069` | 3.55 |
| UID0044 | `1501` | `1461` | 2.74 |
| UID0096 | `0.377` | `0.388` | 2.84 |
| UID0172 | `378238.08` | `372507.20` | 1.54 |
| UID0205 | `0.97` | `0.98` | 1.02 |

### Wrong, wrong-sign, and no-answer UIDs

| UID | Pred | Gold | Cat | Δ% (best scale) |
|---|---|---|---|---:|
| UID0029 | `0.67717` | `0.88525` | wrong | 23.5 |
| UID0030 | `15` | `18` | wrong | 16.7 |
| UID0042 | `-1.172` | `1.172` | wrong-sign | 0.0 |
| UID0053 | `None` | `23.587` | no-answer | — |
| UID0056 | `None` | `1973` | no-answer | — |
| UID0062 | `6758` | `6,244` | wrong | 8.2 |
| UID0069 | `6.14` | `-18.51%` | wrong | sign + magnitude |
| UID0071 | `None` | `0.900544` | no-answer | — |
| UID0083 | `None` | `$2,760.44` | no-answer | — |
| UID0102 | `53.70` | `57.50` | wrong | 6.6 |
| UID0110 | `[2017, 2.80]` | `[2017, 0.69]` | wrong | 90.0 (k=−1) |
| UID0113 | `-3.85` | `17.69` | wrong | sign + magnitude |
| UID0114 | `0.38` | `0.35` | wrong | 8.6 |
| UID0122 | `0.442` | `0.953` | wrong | 53.6 |
| UID0140 | `None` | `907,654` | no-answer | — |
| UID0148 | `[38427, 2401.69]` | `[28, 2444.28]` | wrong | 99.9 (k=−3) |
| UID0150 | `[208.09, -0.95]` | `[191.85, -18.39]` | wrong | 94.8 |
| UID0154 | `7` | `12` | wrong | 41.7 |
| UID0165 | `1568` | `4928` | wrong | 68.2 |
| UID0174 | `-3.147` | `−3.524` | wrong | 10.7 |
| UID0175 | `0.00164` | `0.00137` | wrong | 19.7 |
| UID0177 | `None` | `236.7` | no-answer | — |
| UID0188 | `22882.79` | `2051.51` | wrong | 11.5 (k=−1) |
| UID0193 | `None` | `3.9970` | no-answer | — |
| UID0201 | `[0.006, 'surplus']` | `[0.012, surplus]` | wrong | 50.0 |
| UID0213 | `1906.4` | `-550.3` | wrong | sign + magnitude |
| UID0220 | `165.0` | `27` | wrong | 38.9 (k=−1) |
| UID0223 | `None` | `16.78` | no-answer | — |
| UID0243 | `628.855` | `264.632` | wrong | 76.2 (k=−1) |
| UID0244 | `None` | `504.12` | no-answer | — |
| UID0245 | `0.000` | `-0.113` | wrong | underflow |

Several wrong UIDs show their best fit at non-zero `k` (UID0110 at k=−1, UID0148 at k=−3, UID0188 at k=−1, UID0220 at k=−1, UID0243 at k=−1) but none drop under 5% even at the best scale, so they remain wrong. These are partial-scale errors — closer at k≠0 than at k=0, but still off enough that scale alone doesn't explain them.

---

## Failure-mode analysis

Grouped by **the stage where the failure originated**, not where the trace ended. Covers all 43 UIDs that aren't 100% exact — including the 12 small-drift cases (≤5% but non-zero Δ).

### Summary

Root-cause attribution after inspecting each non-exact trace (question, plan, extract outputs, lookup values, compute code). **The "compute failure" framing is partly misleading — a third of these are upstream bugs.**

| Stage of root cause | Minor drift (≤5%) | Wrong / wrong-sign (>5%) | No-answer | Total | Share of 43 non-exact |
|---|---:|---:|---:|---:|---:|
| **Extract** — missing / duplicate / vision miscount / overlapping ranges | 1 | 8 | 2 | 11 | 25.6% |
| **Lookup_external** — missing / ungrounded | 0 | 0 | 3 | 3 | 7.0% |
| **Planner** — wrong formula or missing qualifier in task | 0 | 4 | 0 | 4 | 9.3% |
| **Compute** — wrong methodology / formula | 1 | 9 | 0 | 10 | 23.3% |
| **Compute** — scale / unit / qualifier ignored | 0 | 2 | 0 | 2 | 4.7% |
| **Compute** — precision / rounding | 7 | 0 | 0 | 7 | 16.3% |
| **Compute** — codegen budget exhausted (FAILED) | 0 | 0 | 1 | 1 | 2.3% |
| **Compute** — returned None (OK but no value) | 0 | 0 | 3 | 3 | 7.0% |
| **Question ambiguity** | 1 | 1 | 0 | 2 | 4.7% |

Attribution shifts vs the prior framing:
- **Extract is responsible for 11 (26%)** of non-exact outcomes, not 2. Six of those are *with-answer* failures where extract returned bad data and compute silently propagated it.
- **Planner accounts for 4 (9%)** — formula bugs or missing qualifiers in the plan task. These dominate the biggest %-off outliers (UID0148 137139%, UID0220 511%, UID0201 50%) because compute faithfully implements a wrong formula.
- **Compute genuinely owns 23 (53%)** — wrong methodology (10), precision drift (7), scale/qualifier (2), no-answer paths (4). Down from the 38 the prior table implied.
- **Question ambiguity is real** (2 UIDs) — answers are defensible under different interpretations.

### 1. Extract-stage failures — 2 UIDs (6.5%)

Outcome `FAILED — [extract] no relevant values found across tiers`. Page text exists, but neither parsed-table tier nor vision tier found a row matching the lookup key. Replanner triggered in both cases; the second extract pass also returned nothing.

| UID | Question topic | Diagnosis |
|---|---|---|
| UID0053 | public-debt securities held by public, FY1994 (replan-triggered) | Extract on the replanned FY1994 branch came back empty; page didn't carry that row |
| UID0083 | national defense FY2024 monthly figures | Extract found nothing; planner spent 5.4 min on this UID (longest planner time) |

**When it happens**: at extract — the operator decides "no relevant values" after running both tiers. Diagnosis is upstream: the golden page either (a) wasn't retrieved (under `--golden` this shouldn't happen unless the planner emitted the wrong key) or (b) the table on the page uses phrasing the extract LLM didn't match.

### 2. Lookup_external-stage failures — 3 UIDs (9.7%)

Lookup didn't return a usable value. Pipeline either ended at the lookup or at a follow-up replanner without re-entering compute.

| UID | Lookup target | Failure |
|---|---|---|
| UID0193 | USD/GBP FX rate, CY1941 | `[lookup_external] Ungrounded reply (no grounding_titles)` — Google Search path gave a reply without grounding citations |
| UID0223 | Germany nominal GDP 1996 (replanner-triggered) | Pipeline ended at replanner step; second lookup wasn't re-attempted before time/iteration budget exhausted |
| UID0244 | USD/CAD FX rate, Dec 1959 (replanner-triggered) | Replanner re-issued lookup; second attempt finished but compute didn't re-fire with the new value |

**When it happens**: at `lookup_external` for UID0193, and at the **handoff** between replanner and the next compute pass for UID0223 and UID0244. The replanner identifies a missing value, issues a fresh lookup branch, but the orchestrator doesn't always cycle back through compute with the updated `prev`. That's a control-flow bug, not a lookup-content bug.

### 3. Compute — codegen budget exhausted — 1 UID (3.2%)

Outcome `FAILED — [compute] no successful exec within budget`. Codegen kept producing programs that raised at exec time; the retry budget ran out.

| UID | Last error | Diagnosis |
|---|---|---|
| UID0177 | `Could not find scalar for description parts: ['Nonmarketable', 'February 28, 1950']` after 9 codegen attempts | Extract delivered `Total marketable` for 1950-02-28; the question asked for **Nonmarketable**. Codegen kept rewriting the lookup expression, never realizing the row was missing from `prev` — should have emitted the `{"missing": [...]}` JSON instead |

**When it happens**: inside compute, in the codegen → exec → retry loop. Same `KeyError` / `description parts not found` error fires repeatedly because retry has no signal that the data fundamentally isn't there.

### 4. Compute — returned None (OK outcome) — 3 UIDs (9.7%)

Compute finished cleanly (no FAILED), but the final answer is `None`. Codegen exec budget didn't trip, but `result` ended up unset / None.

| UID | n_codegen | Question | Diagnosis |
|---|---:|---|---|
| UID0056 | 6 | Personal saving rate calculation (1987–1989) | Replanner fired; compute exited OK but no number emitted |
| UID0071 | 11 | TIPS price std-dev with index ratios (2007) | High codegen-attempt count; likely a critique that pruned every program before one set `result` |
| UID0140 | 6 | Total surplus/deficit comparison (2008) | 23 extract values delivered, compute didn't converge on a single answer |

**When it happens**: in compute, on a path where the model emitted code that ran without exception but didn't assign `result` (or assigned None). The exit-OK + no-value path needs an explicit "did codegen actually produce a value?" check, otherwise these reach the user as silent no-answers.

### 5–6. Non-correct with answer (34 UIDs) — per-UID root-cause telemetry

The 22 wrong + 12 minor-drift UIDs are NOT homogeneous "compute screwed up". I inspected every trace (question, plan, extract outputs, lookup values, compute code) and attributed a root-cause stage to each. The result is below — the **majority of "compute" failures are actually upstream (planner emitted a wrong formula, or extract returned bad data)**.

#### Per-UID attribution

| UID | Cat | Δ% | Root cause stage | Specific diagnosis |
|---|---|---:|---|---|
| UID0029 | wrong | 23.5 | **Extract** | 21 extracted entries with 11 *duplicate descriptions but different values* (e.g., 3 entries labeled "Aa yields 1963" with three different value-vectors). Compute couldn't disambiguate. |
| UID0030 | wrong | 16.7 | **Extract (vision)** | Vision extract returned scalar `15` for "count local maxima"; gold is 18. Vision under-counted. |
| UID0042 | wrong-sign | 0.0 | **Planner** | Plan task says "calculate the Zipf exponent" — qualifier `"Zipf exponent is the slope of the log-size vs log-rank regression"` doesn't mention the convention that Zipf is *reported as positive*. Compute emitted the raw slope `-1.172`; gold `1.172`. |
| UID0062 | wrong | 8.2 | **Extract → Compute selection** | Extract returned two FY1947 rows: `"...FY1947"` and `"...FY1947 (revised reporting)"`. Question explicitly asks for the *pre-Air-Force-split* reporting structure. Compute picked the wrong row of the two. |
| UID0069 | wrong | 133 | **Compute methodology** | Question asks for 95% Expected Shortfall via historical method. Compute returned `6.14` (looks like a mean or VaR), gold `-18.51%`. ES formula wasn't applied. |
| UID0102 | wrong | 6.6 | **Compute methodology** | Extract returned both 12 monthly values AND quarterly aggregates (Q1, Q2, Q3). H Spread (IQR Type 7) requires computing percentiles from the 12 monthly values. Compute likely used the quarterly aggregates as if they were Q1/Q3 directly. |
| UID0110 | wrong | 90 (k=−1) | **Compute methodology** | "Geometric mean of quarterly growth rates" — must convert percent change to growth factor `(1+r/100)`, geomean, then back. Compute geomeaned percentages directly. Right year, wrong magnitude. |
| UID0113 | wrong | 122 | **Extract incomplete** | Question needs 1980 + 1981 redemption rates. Extract returned `"Redemptions Total, CY1981"` and `"Amount outstanding, CY1980/1981"` but **no `Redemptions Total, CY1980`**. Compute filled in with available data → wrong sign + wrong magnitude. |
| UID0114 | wrong | 8.6 | **Extract overlap → Compute drift** | Extract returned one CY1999–CY2002 monthly series AND four yearly monthly series (overlapping). Compute ran 14 codegen attempts (19.7 min) on the duplicated data and landed at 0.38 vs gold 0.35. |
| UID0122 | wrong | 53.6 | **Compute methodology** | ESF foreign-exchange share with CPI deflation — multi-step computation. Plan task is correct; extract + lookup both clean. Compute emitted `0.442` vs gold `0.953`. Suspected wrong aggregation order (share-of-mean vs mean-of-shares). |
| UID0148 | wrong | 137139 | **Planner** | Question: "how many ... exceeded 2400M". Plan task says "Calculate the **total amount** outstanding (sum)". Planner misread "how many" as "total amount". First answer element is the sum (38427) instead of the count (28). |
| UID0150 | wrong | 95 | **Extract failure** | Bond prices extract OK (5 entries). USD/DEM FX extract step FAILED with `[extract] no relevant values found`. Compute proceeded without FX rates and improvised — wrong answer. |
| UID0154 | wrong | 41.7 | **Question ambiguity** | Question says "Using only **exactly 2 sources** ... how many months from Feb 1977–Jan 1979 had Treasury bills > $20B". Compute counted months where the survey-snapshot value exceeded threshold (= 7 by snapshots); gold interprets as "how many months in the range" using interpolation/continuity (= 12). Both defensible. |
| UID0165 | wrong | 68.2 | **Compute methodology** | 1% lower-tail VaR + currency conversion. Extract + lookup clean. Compute ran 8 codegen attempts + replanner. Methodology error in VaR calculation. |
| UID0174 | wrong | 10.7 | **Compute methodology** | Extract clean (Jan + Mar 1960 values for both series). Pred `-3.147`, gold `-3.524`. Same sign, slightly different formula — likely midpoint-elasticity variant. |
| UID0175 | wrong | 19.7 | **Compute methodology** | Plan qualifier explicitly says `"sample variance"`. With n=6, sample/population ratio = n/(n-1) = 6/5 = 1.2 — and 0.00137 × 1.2 = 0.00164 = pred exactly. **Compute used population variance, ignoring the qualifier.** |
| UID0188 | wrong | 1015 | **Lookup → Compute scale** | Initial lookup of 1938 silver price errored (JSON parse); replan recovered it. Pred `22882.79` ≈ 10× gold `2051.51`. Scale/unit error in compute — likely treating statutory rate per ounce vs per thousand-ounces inconsistently. |
| UID0201 | wrong | 50 | **Planner** | For 2 values, Gini = `|A−B|/(A+B)`. Plan task says `Gini = |A − B| / (2 × (A + B))` — that formula is off by a factor of 2. Pred 0.006 is exactly half of gold 0.012. **Pure planner formula bug.** |
| UID0213 | wrong | 446 | **Compute methodology + sign** | Extract step initially errored (Dec 1946 missing), replan recovered it. CPI lookups OK. Compute emitted `1906.4` vs gold `-550.3` — both sign and magnitude off (3.5× magnitude). Inflation-adjust formula likely inverted. |
| UID0220 | wrong | 511 | **Planner** | Question asks "absolute **percent** difference". Plan task says "**Absolute difference**" — dropped the "percent". Pred `165.0` is the raw difference in $M; gold `27` is the percent difference. **Pure planner qualifier-drop.** |
| UID0243 | wrong | 137 | **Compute methodology** | FY1960/1961/1962 debt + CPI lookups OK. Pred 628.855 / gold 264.632 = 2.376. Likely compute compared inflation-adjusted FY1961 vs **nominal** FY1960, instead of both adjusted to 1962 dollars. |
| UID0245 | wrong | underflow | **Extract** | Extract step [3] for Aug 1982 yield returned **2 duplicate entries** (same description, possibly same value). With duplicates of 1982 standing in for both 1981 and 1982, the symmetric growth rate `(B−A)/((A+B)/2) = 0`. Pred `0.000`, gold `-0.113`. |
| UID0007 | within_1 | 0.106 | **Compute precision** | Geom mean of 80 months 1942-03–1948-10. Extract returned one clean vector. Drift is likely a single month included/excluded at an endpoint. |
| UID0010 | within_0_1 | 0.0013 | **Compute precision** | JPY conversion. FX rate 149.92 (presumed rounded). Drift is rounding of FX rate (gold may have used 149.92 with one more digit). |
| UID0018 | within_5 | 1.26 | **Extract overlap** | Extract returned 4 entries: 1984 monthly, 1985 monthly, 1986-01..1987-03, AND 1986-01..1986-12. The two 1986 ranges overlap — 1986 months double-counted in the geomean if not deduped. |
| UID0027 | within_5 | 3.55 | **Extract dup → Compute** | Extract step [9] returned 18 entries for Aa yields with multiple "1963 (alternate)"-style duplicates. Compute identified the max-spread month, but in the duplicated data the wrong month surfaces as max. |
| UID0044 | within_5 | 2.74 | **Question ambiguity** | "Between the 3rd Thursday and 4th Wednesday in Jan 1939". The 4th Wed is Jan 25; "week ended Jan 25" appears in the extract. Compute returned 1501; gold 1461. Both are defensible depending on whether the boundary week is fully included. |
| UID0084 | within_1 | 0.129 | **Compute methodology** | OLS forecast FY1968 from FY1961–1967 interest charges. Extract returned a single vector including FY1968 — compute may have included FY1968 in the regression rather than holding it out. |
| UID0096 | within_5 | 2.84 | **Compute methodology** | Centered MA of customs duty rate FY1939–1941. Extract returned 6 entries (3 duties + 3 dutiable-imports values). Compute computed mean of `duty[i]/imports[i]` ratios; gold likely computed `mean(duty) / mean(imports)`. Different aggregation order. |
| UID0172 | within_5 | 1.54 | **Compute rounding** | UK liabilities × GBP/USD rate. Plan qualifier says "round FX rate to hundredths before multiplication". FX is `0.6559...`; rounded to 0.66. Compute may have rounded to 0.65 or used unrounded 0.6559 → 1.5% drift. |
| UID0205 | within_5 | 1.02 | **Compute precision** | 17 years of FY receipts/expenditures × annual FX, then Pearson. Lots of floating-point steps; 1% drift over 17 years of multiplied values is consistent with intermediate rounding. |
| UID0214 | within_0_1 | 0.03 | **Compute precision** | Currency in circulation × (1 + CPI YoY/100). Lookup gave CPI rate `5.932...` — gold likely used 5.93 or different precision. |
| UID0225 | within_0_1 | 0.08 | **Compute precision / extract noise** | Extract returned 2 entries; one is labeled "26-week" (question only asks 13-week). Even so, compute averaged correctly. Final rounding diff (12.59 vs 12.6). |
| UID0226 | within_1 | 0.129 | **Compute precision** | Std dev of 3 values. With n=3, sample-vs-population stdev ratio = √(n/(n−1)) = √1.5 ≈ 1.22, which would be 22% off — not 0.13%. So compute used the right variant; this is straight numerical drift. |

#### Summary by root cause

| Stage of root cause | Count | UIDs |
|---|---:|---|
| **Extract — duplicate / mislabeled entries** | 4 | UID0029, UID0027, UID0245, UID0114 |
| **Extract — incomplete (missing required value)** | 2 | UID0113, UID0150 |
| **Extract — vision miscount** | 1 | UID0030 |
| **Extract — overlapping ranges in series** | 1 | UID0018 |
| **Extract — ambiguous multiple rows** | 1 | UID0062 |
| Extract subtotal | **9** | |
| **Planner — wrong formula in task** | 3 | UID0148, UID0201, UID0220 |
| **Planner — missing convention qualifier** | 1 | UID0042 |
| Planner subtotal | **4** | |
| **Compute — wrong methodology / formula** | 9 | UID0069, UID0084, UID0096, UID0102, UID0110, UID0122, UID0165, UID0174, UID0213, UID0243 (10 — UID0084 is within-1) |
| **Compute — scale / unit conversion** | 1 | UID0188 |
| **Compute — qualifier ignored (sample vs population)** | 1 | UID0175 |
| **Compute — precision / rounding** | 7 | UID0007, UID0010, UID0172, UID0205, UID0214, UID0225, UID0226 |
| Compute subtotal | **19** | |
| **Question ambiguity** | 2 | UID0044, UID0154 |

So of the 34 non-correct-with-answer UIDs: **9 (26%) are upstream extract bugs**, **4 (12%) are planner bugs**, **19 (56%) are genuine compute bugs**, and **2 (6%) are question-ambiguity cases**. The "compute failure" framing in earlier reports was hiding that **roughly a third (13/34) of these losses would be fixed by improving extract or planner**, not by touching compute at all.

#### Notable upstream patterns worth fixing first

- **Extract duplicate descriptions are common and silently corrupt downstream compute**. UID0029 (23.5% off), UID0027 (3.6%), UID0245 (underflow), UID0114 (8.6%) all have the same root signature: extract returned multiple entries with identical or near-identical descriptions but different values. The extract dedup step is supposed to handle this; it's not catching value-conflict duplicates. A verifier that flags "same description, different values" would catch all 4.
- **Planner-formula bugs are dramatic** when they fire. UID0148 (137139% off), UID0201 (50% off, exactly 2×), UID0220 (511% off) are all cases where the model wrote a *mathematically wrong* task description and compute faithfully implemented the wrong formula. These don't show up in compute logs as errors — compute produces an answer, it's just answering the wrong question. A planner-output sanity check (does the task formula match the question's verb tense and operator?) could catch UID0148 (sum-vs-count) and UID0220 (missing "percent").
- **Qualifier-drop is a real category**. UID0175 (sample → population variance), UID0042 (Zipf positive), UID0220 (missed "percent") all have correct-looking plans where one explicit instruction was lost between plan and code. The codegen prompt asks for every qualifier to "appear as an explicit operation, not a comment" — but the qualifier isn't enforced.

---

## Runtime

Per-UID wall = sum of step durations (retrieve ≈ 0 under `--golden`). Stage time is summed across operator invocations within a UID (some UIDs run extract / compute more than once after the replanner fires).

| Stage | Active in N UIDs | Total time | Mean (active) | Median (active) | p90 (active) | Max |
|---|---:|---:|---:|---:|---:|---:|
| **Wall (total)** | 101 | 4.17 h | 2.48 min | 57.0 s | 6.5 min | 21.8 min |
| planner | 101 | 39.0 min | 23.2 s | 11.5 s | 32.3 s | 5.4 min |
| extract | 101 | 46.7 min | 27.7 s | 11.2 s | 41.0 s | 11.2 min |
| lookup_external | 19 | 29.3 min | 1.54 min | 1.10 min | 3.20 min | 12.8 min |
| compute | 101 | 2.15 h | 1.28 min | 22.9 s | 4.30 min | 19.7 min |
| replanner | 14 | 6.4 min | 27.3 s | 24.0 s | 47.0 s | 1.69 min |

### Share of total wall, by stage

| Stage | Total time | Share |
|---|---:|---:|
| planner | 39.0 min | 15.6% |
| extract | 46.7 min | 18.6% |
| lookup_external | 29.3 min | 11.7% |
| compute | 2.15 h | 51.5% |
| replanner | 6.4 min | 2.5% |
| **Total** | **4.17 h** | 100% |

Compute is just over half of total wall — exactly where the failure-mode share concentrates. Long-tail compute UIDs (UID0114 at 19.7 min, UID0203 at 12.8 min lookup, UID0083 at 5.4 min planner) are the ones to attack for both runtime and accuracy.

### Top-10 longest UIDs (wall)

| UID | Wall | Planner | Extract | Lookup | Compute | Replan | Cat |
|---|---:|---:|---:|---:|---:|---:|---|
| UID0114 | 21.8 min | 12.2 s | 1.29 min | 0 s | 19.7 min | 38.0 s | wrong |
| UID0203 | 18.0 min | 25.4 s | 16.3 s | 12.8 min | 4.55 min | 0 s | exact |
| UID0083 | 17.7 min | 5.44 min | 11.2 min | 0 s | 31.3 s | 27.3 s | no-answer |
| UID0071 | 15.0 min | 22.7 s | 1.38 min | 0 s | 12.4 min | 49.4 s | no-answer |
| UID0027 | 11.5 min | 11.5 s | 52.5 s | 0 s | 10.1 min | 21.3 s | within 5% |
| UID0188 | 10.4 min | 41.7 s | 52.8 s | 2.43 min | 5.88 min | 32.7 s | wrong |
| UID0028 | 8.5 min | 17.2 s | 1.18 min | 0 s | 7.05 min | 0 s | exact |
| UID0172 | 8.4 min | 29.2 s | 21.4 s | 1.53 min | 5.33 min | 39.8 s | within 5% |
| UID0140 | 7.9 min | 15.5 s | 1.35 min | 0 s | 5.49 min | 50.2 s | no-answer |
| UID0009 | 6.4 min | 5.02 min | 11.2 s | 0 s | 1.24 min | 0 s | exact |

UID0114's 19.7-min compute / 14-codegen run is the single biggest opportunity to both shorten the tail and recover an accuracy point — it currently lands wrong by 8.6% after burning that budget.

---

## Replanner activity

14 of 101 UIDs invoked the replanner at least once. Outcomes:

| Replanner outcome | Count | UIDs |
|---|---:|---|
| Exact / within 1% | 1 | UID0214 (within 0.1%) |
| Within 5% | 2 | UID0027, UID0172 |
| Wrong | 5 | UID0114, UID0165, UID0188, UID0213, UID0223 |
| No answer | 6 | UID0053, UID0056, UID0071, UID0083, UID0140, UID0177 |

Replanner is a low-yield rescue path: 1/14 = 7% recovered to within 1%, 3/14 = 21% to within 5%. The handoff bug (§2) accounts for several of the no-answers — replanner ran but compute never re-fired with the new data.

---

## Action items

Re-ranked by root-cause attribution. Going after the upstream bugs first because they're cheaper to fix and they affect downstream silently.

**Tier 1 — upstream fixes (extract + planner). ~13 UIDs of recovery potential.**

1. **Add a value-conflict check to extract dedup** — UID0029 (23.5% off), UID0027 (3.6%), UID0245 (underflow), UID0114 (8.6%) all stem from extract returning multiple entries with the same / near-identical description but different values. The existing dedup catches description duplicates; it doesn't catch description-match + value-mismatch. Add a verifier that flags or merges these. **~4 UIDs**.
2. **Plan-task formula sanity check** — UID0148 (137139%, "how many" → "total amount"), UID0201 (50%, wrong Gini formula), UID0220 (511%, dropped "percent"). The planner is writing math that doesn't match the question's verb. A second-pass check ("does the task formula's output type match the question's interrogative?") would catch all three. **~3 UIDs**.
3. **Qualifier enforcement in codegen** — UID0175 (sample vs population variance), UID0042 (Zipf convention). Every qualifier in the plan is supposed to "appear as an explicit operation, not a comment". It doesn't. Either tighten the codegen prompt or add a static check that each qualifier maps to a code statement. **~2 UIDs**.
4. **Vision extract reliability for chart-reading questions** — UID0030 (15 vs 18 maxima). Only 1 UID hit this in the trace, but it's an open category whenever the question is about a visual element. Low priority unless more chart questions land.
5. **Extract robustness for overlapping series ranges** — UID0018, UID0114. When extract returns both a wide-range vector and several narrower-range vectors covering the same period, downstream compute double-counts unless explicitly told. **~1–2 UIDs**.

**Tier 2 — control-flow bugs (the no-answer paths). ~5 UIDs.**

6. **Compute's "OK + result=None" exit path** — UID0056, UID0071, UID0140, plus the underflow case UID0245. Add a check that `result` is set before declaring success. Converts silent no-answers into FAILEDs (so replanner has a signal to fire).
7. **Replanner → compute re-entry** — UID0223, UID0244 land no-answer because after replanner fires a fresh lookup, compute doesn't re-invoke. Control-flow bug.
8. **Recognize repeat-failure missing-data signal** — UID0177. Codegen's exec keeps failing with the same `description not found` error; the model should emit `{"missing": [...]}` instead of retrying.

**Tier 3 — genuine compute bugs (the hard residue). ~12 UIDs but each needs case-specific work.**

9. **Compute methodology errors** — UID0069 (ES), UID0102 (IQR on aggregates), UID0110 (geomean on % vs factors), UID0122, UID0165, UID0174, UID0213, UID0243. These are the cases where the plan was reasonable, the data was correct, and compute applied the wrong formula. The fix is per-UID prompt examples + maybe a "what does the textbook formula for X look like?" reference panel.
10. **Compute scale/conversion** — UID0188 (silver-stock 10× off). One UID; worth pulling the code and inspecting.
11. **Cap compute codegen iterations + wall-clock** — UID0114 (19.7 min, 14 attempts), UID0071 (12.4 min, 11). No accuracy cost (these are already losses); just bounds the tail latency.
12. **Compute precision drift (≤5%)** — UID0007, UID0010, UID0172, UID0205, UID0214, UID0225, UID0226. These are the ones that close the gap between "within 5%" and "exact". Hard to ship as one change; needs prompt nudges around (a) range endpoints, (b) FX-rounding policy, (c) intermediate precision.

**Out of scope** (question ambiguity, 2 UIDs): UID0044 (date-range boundary), UID0154 (interpretation of "2 sources"). Both answers are defensible. Could be addressed by clarifying the benchmark questions, not by changing the pipeline.

---

## Caveats

- **Grading is automated, with scale tolerance**. First-number for scalars, element-wise for lists. Scale tolerance over `k ∈ {0, ±1, ±2, ±3, ±6, ±9, ±12}` handles "answer × 10⁶ vs answer in millions" cleanly. Doesn't capture semantic equivalence ("surplus" vs `-0.012`).
- **Aggregated trace = last-seen trace per UID**. The directory consolidates baseline + lookup-fix + replan-rescue runs by keeping the most recent trace for each UID. Earlier traces under different prompts / effort levels live in the sibling `lookup_fix_*` and `replan_rescue_*` directories.
- **Dev-only numbers**. The 32 held-out test UIDs in `eval/test_set_uids.json` were not touched. Any generalization claim later needs a separate measurement on the test set.
