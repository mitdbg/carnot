# DSL — formal specification

The OfficeQA pipeline is encoded as a **flat Plan** of data-gathering branches that feed an implicit `compute()` step at the end. The planner emits this as text (or JSON); the orchestrator walks the branches and dispatches subagent calls.

## Plan shape

Every plan is exactly one of:

```
single retrieve branch:    retrieve(...) --> extract(...) --> compute()
single lookup branch:      lookup_external(...) --> compute()
parallel of branches:      [ branch ; branch ; ... ] --> compute()
```

`compute()` is always at the end. It's a structural part of every plan, but takes no arguments. The grammar does not support nested parallels, chains that skip `extract` after `retrieve`, or `compute` anywhere other than the terminator — the type system makes those unrepresentable.

## Branches

A branch is one of two shapes:

```
RetrieveBranch = retrieve(concept, period[, source_bulletin]) --> extract([visual_only])
LookupBranch   = lookup_external(nl)
```

### `RetrieveBranch`

| field | type | required | description |
|---|---|---|---|
| `concept` | str | yes | domain tag(s) the page must report on |
| `period` | period string | yes | date span the question targets (see grammar below) |
| `source_bulletin` | `'YYYY-MM'` | no | pin to one specific issue |
| `visual_only` | bool | no | set on the trailing `extract(...)`; skips Tiers 1-2 and goes directly to vision |

The `visual_only` flag is part of the branch (it threads through to the extract step) and is only meaningful when the question targets charts/figures/scanned images.

### `LookupBranch`

| field | type | required | description |
|---|---|---|---|
| `nl` | str | yes | natural-language description of the external data to look up |

Use for any factual data that can't be extracted from the bulletin corpus: CPI-U, exchange rates, event dates, bureau names, etc.

## Surface forms

Both text DSL and JSON are accepted.

### Text DSL

```
retrieve(concept='national_defense', period='CY1940') --> extract() --> compute()
[ retrieve(concept='a', period='CY1940') --> extract() ; lookup_external(nl='CPI-U for 1953') ] --> compute()
```

String arguments use **single quotes** (`'...'`) — keeps plans JSON-safe when wrapped.

### JSON

```json
{
  "branches": [
    {"kind": "retrieve", "concept": "national_defense", "period": "CY1940"},
    {"kind": "lookup_external", "nl": "USD/JPY exchange rate on 2025-03-31"}
  ],
  "global_constraints": []
}
```

A simple chain has one element in `branches`. A parallel has two or more.

## Inter-op types

```
PageRef         { month: 'YYYY-MM', page: int, file_path: str }
DocHandle       { refs: [PageRef], desc: str }
NamedEntry      { unit: str, quote: str, dims: {...},
                  kind: 'scalar' | 'vector' | 'table',
                  index_name?: str,         # vector only
                  row_name?: str, col_name?: str  # table only
                }
TypedValue      { value: dict[str, Any], desc: str, meta: dict[str, NamedEntry] }
FormattedString { text: str }
```

`PageRef.page` is the **1-based PDF page index** — the only page-number meaning used in the codebase. It matches `source_docs?page=N` from the Fraser benchmark URLs.

`TypedValue.value` is always a dict. For results from `lookup_external`, the single unnamed value lives under key `""`. For results from `extract`, keys are snake_case names disambiguating the extracted entries (e.g. `national_defense_cy1940`). The `meta` dict is keyed identically.

Each entry has one of three **kinds**, with payload shapes enforced by `TypedValue.__post_init__`:

- `kind='scalar'` — `value[k]` is `int | float | str`. Single facts, sibling cells distinguished by `dims`.
- `kind='vector'` — `value[k]` is a flat dict `{index_label: scalar}`, with `meta[k].index_name` naming the varying dim (e.g. `'month'`). Used for 1-D series; `dims` is shared across all cells.
- `kind='table'` — `value[k]` is a 2-level dict `{row_label: {col_label: scalar}}` with consistent column keys across rows, and `meta[k].row_name` / `meta[k].col_name` naming the two varying dims.

**No nesting beyond these shapes.** Vector cells and table cells must be primitive scalars (int/float/str). A vector-of-vectors, table-with-list-cells, or 3-level dict is forbidden and will be dropped by the extract parser; if it ever reaches `TypedValue.__post_init__` (a bug in extract), construction fails loudly with `ValueError` rather than corrupting compute downstream.

**Quorum is all-or-nothing per entry.** When extract runs N samples, samples are bucketed by the canonical form of the *whole* entry (name + kind + axis names + every cell value + unit + dims). Only buckets meeting `extract_quorum` survive. There is no per-cell voting — a single-cell disagreement keeps both variants in separate buckets, and neither is kept unless quorum is hit on the exact match.

## Period grammar

Used by `RetrieveBranch.period`.

```
period      ::= point | range | enumeration
point       ::= "CY"YYYY | "FY"YYYY | "Q"[1-4]"-"YYYY | YYYY"-"MM | YYYY"-"MM"-"DD
range       ::= point ".." point                                  # inclusive
enumeration ::= point ("," point)+                                # multi-point list
```

Examples: `CY1940`, `FY1981`, `Q3-1982`, `2025-03-31`, `CY1940..CY1949`, `1991-06,1996-06,2001-06`.

## Operators

### `retrieve(concept, period, source_bulletin?) → DocHandle`

Looks up matching pages via the retrieve subagent's internal page index (filtered by concept + period + optional source_bulletin pin), returns a `DocHandle` whose `PageRef`s are fully specified.

**Failure.** `StepFailed("retrieve", "no candidates after pre-filter")`.

### `extract(visual_only?) → TypedValue`

| arg | type | req | description |
|---|---|---|---|
| `visual_only` | bool | no | Skip Tiers 1-2; vision-only |

Per-page tier dispatch:
1. Tier 1 — JSON-parsed page elements via `skunk.common.parsed_json.get_text_for_pdf_page`.
2. Tier 2 — PyMuPDF native text per PDF page (`cache/pages/{month}/p{NNN}.txt`).
3. Tier 3 — PNG render at 300 dpi sent to vision LLM.

Returns a `TypedValue` whose `.value` is a dict of named entries — each one a `scalar`, `vector`, or `table` per the kind contract above, anchored by a verbatim quote on the page.

**Failure.** `StepFailed("extract", "no relevant values found across tiers")`.

### `lookup_external(nl) → TypedValue`

Single Gemini call. Returns a `TypedValue` with `value={"": <result>}` and a single `meta[""]` entry carrying the inferred unit.

### `compute() → FormattedString`  *(implicit terminator)*

No arguments. Reads `ctx.question` and `prev` (the data-phase output: either one `TypedValue` for a single-branch plan or `list[TypedValue]` for a parallel), then runs:

1. **Plan + codegen** (Gemini): emits `CODE\n<python>` or `MISSING:<reason>`.
2. **Exec** (sandbox): runs the code with `prev` in scope; available imports are numpy, pandas, statsmodels, math.
3. **Verifier** (Gemini): checks the output's *form* (precision, unit, percent vs decimal, comma rules, list bracketing) against the question.

Up to 3 attempts per question; each retry sees the complete history of prior failures.

**MissingData recovery.** If compute reports `MISSING:<reason>`, the orchestrator catches it as a `MissingData` exception and runs a single recovery round: the recovery planner emits one supplemental `Branch` (typically a `LookupBranch`), the orchestrator executes it, augments `prev`, and retries compute. If compute still reports MISSING after recovery, the question fails.

**Failure modes.** `StepFailed("compute", ...)` after all attempts fail or recovery declines/exhausts.

## Validation

The type system enforces plan shape (head/tail/branch composition). The validator only checks period strings against the period grammar.

## Worked examples

**UID0001** — Total US national defense expenditures for CY1940:
```
retrieve(concept='national_defense_expenditure', period='CY1940')
  --> extract()
  --> compute()
```

**UID0004** — Absolute pct change between CY1940 and CY1953:
```
[
  retrieve(concept='national_defense_expenditure', period='CY1940') --> extract();
  retrieve(concept='national_defense_expenditure', period='CY1953') --> extract()
]
  --> compute()
```

**UID0029** — Bulletin pinning:
```
retrieve(concept='bond_yields', period='CY1960..CY1969', source_bulletin='1970-06')
  --> extract()
  --> compute()
```

**UID0030** — Visual-only extraction on a chart page:
```
retrieve(concept='line_plots_on_page', period='1990-09', source_bulletin='1990-09')
  --> extract(visual_only=True)
  --> compute()
```

**UID0055** — Parallel external lookups:
```
[
  lookup_external(nl='year WWII ended');
  lookup_external(nl='year Korean War started')
]
  --> compute()
```

**UID0010** — Mixed parallel (retrieve + external lookup):
```
[
  retrieve(concept='fx_investments', period='2025-03', source_bulletin='2025-03') --> extract();
  lookup_external(nl='USD/JPY exchange rate on 2025-03-31')
]
  --> compute()
```

The plan cache lives at `data/dsl_planning_pass.csv` and is regenerated by `tools/plan_dsl_with_gemini.py` (or lazily by `skunk.run` on first execution per UID). Each row is `uid, question, plan_text` and is loaded via `skunk.run.load_plan_cache`.
