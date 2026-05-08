# DSL — formal specification

The OfficeQA pipeline is encoded as a **DSL plan**: a left-to-right chain of operators with optional parallel branches. The planner emits this as text; the orchestrator parses it into an AST and walks it.

## Surface grammar

```
chain      ::= step ("-->" step)*
step       ::= op_call | parallel
parallel   ::= "[" chain (";" chain)+ "]"
op_call    ::= name "(" args? ")"
args       ::= kv ("," kv)*
kv         ::= ident "=" value
value      ::= literal | list | string         # strings ALWAYS use single quotes
list       ::= "[" value ("," value)* "]"
```

### Quoting rule (load-bearing)

All string argument values use **single quotes** (`'...'`). This keeps plans JSON-safe when the planner output is wrapped in JSON. When a Python literal inside `compute(code='...')` itself needs a quoted string, use `"..."` for the inner literal — the outer single quotes never collide.

### Per-op annotations

Each `op_call` may carry two pieces of metadata orthogonal to its `args`:
- `concepts` — natural-language tags for the domain entities involved
- `constraints` — natural-language rules the subagent must obey ("use nominal dollars", "round to nearest hundredth")

JSON serialization is the canonical interchange format; the text grammar is sugar for human authoring and few-shots.

## Types

```
PageRef         { file_path: str, year: int, month: "YYYY-MM", page: int, pdf_page: int? }
DocHandle       { refs: [PageRef], desc: str }                # non-empty refs invariant
TypedValue      { value: Any, dtype: str, desc: str }         # dtype: scalar:<unit>, list[<inner>], df, …
FormattedString { text: str, desc: str }
Output          = DocHandle | TypedValue | FormattedString
```

`PageRef.page` is the **bulletin printed page number** (canonical; matches `source_docs?page=N`).
`PageRef.pdf_page` is the 1-based PDF page index used for cache file access and rendering.
The translation between the two lives in `prep/page_map.py`.

`dtype` is advisory but inspected by `compute` and `format` for unit-checking. Reserved tags: `scalar:<unit>`, `list[<inner>]`, `tuple[<inner>,...]`, `df`, `unknown`.

## Period grammar

Used by `retrieve.period`.

```
period      ::= point | range | enumeration
point       ::= "CY"YYYY | "FY"YYYY | "Q"[1-4]"-"YYYY | YYYY"-"MM | YYYY"-"MM"-"DD
range       ::= point ".." point                                  # inclusive
enumeration ::= point ("," point)+                                # multi-point list
```

Examples: `CY1940`, `FY1981`, `Q3-1982`, `2025-03-31`, `CY1940..CY1949`, `1991-06,1996-06,2001-06`.

## Operators

### `retrieve(concept, period, source_bulletin?) → DocHandle`  *(chain-head only)*

| arg | type | req | description |
|---|---|---|---|
| `concept` | str \| list[str] | yes | domain tag(s) the page must report on |
| `period` | period-string | yes | date span the question targets |
| `source_bulletin` | `'YYYY-MM'` | no | pin to one specific issue when the question explicitly names a bulletin |

**Behavior.** Looks up matching pages via the retrieve subagent's internal page index (filtered by concept + period + optional source_bulletin pin), returns a `DocHandle` whose `PageRef`s are fully specified (`page` = bulletin printed page number, `pdf_page` = PDF index).

**Failure.** `StepFailed("retrieve", "no candidates after pre-filter")`.

### `extract(concept, mode?) → TypedValue`

| arg | type | req | description |
|---|---|---|---|
| `concept` | str | yes | the value(s) to read |
| `mode` | `'value'` \| `'list'` \| `'table'` | no, default `'value'` | scalar / list of scalars / whole DataFrame |

**Pre.** `prev` is `DocHandle`. **Post.** `TypedValue` (`dtype` reflects mode).

**Behavior.** Per-page tier dispatch:
1. Tier 1 — load `cache/tables/{month}/p{page}-*.csv`, send DataFrames as CSV to LLM.
2. Tier 2 — slice the page's OCR text, send to LLM.
3. Tier 3 — render page (PyMuPDF, 300 dpi), send to vision LLM.

First tier with a confident answer wins.

**Failure.** `StepFailed("extract", "value not found across tiers")`.

### `read_visual(concept) → TypedValue`

| arg | type | req | description |
|---|---|---|---|
| `concept` | str | yes | what to read from the chart/figure |

Same I/O as `extract` minus `mode`. Always Tier 3 vision. Used for charts and figures.

### `lookup_external(resource, **params) → TypedValue`  *(chain-head capable)*

| arg | type | req | description |
|---|---|---|---|
| `resource` | str | yes | `cpi_u` \| `fx_rate` \| `gdp` \| `event_year` \| `event_date` |
| `**params` | per-resource kwargs | varies | e.g. `currency='JPY'`, `date='2025-03-31'`, `event='korean_war_start'` |

Cache-first CSV read, falls back to live API (BLS / FRED / exchangerate.host) when `cache_only=False`. The `event_year` / `event_date` resources resolve knowledge-bound dates that the planner can't pin from the question text alone.

### `compute(code) → TypedValue`

| arg | type | req | description |
|---|---|---|---|
| `code` | str | yes | Python body that reads `prev` and sets `result = ...`. Sandboxed: numpy, pandas, statsmodels, math available. |

**Pre.** `prev` is `TypedValue` or `list[TypedValue]` (from a parallel branch). **Post.** `TypedValue`.

This is the only transformation op. Reducers (sum, mean, geo_mean, std, …) and named formulas (abs_diff, cagr, …) are all one-liners in `code`. Sandboxed exec, retried up to 2× on exception.

### `format(precision?, unit?, layout?) → FormattedString`  *(chain terminator)*

Flat keyword args (no nested dict — keeps the surface JSON-safe). At most 3 keys:

| arg | type | req | description |
|---|---|---|---|
| `precision` | int | no | decimal places |
| `unit` | str | no | `'usd'` \| `'usd_millions'` \| `'pct'` \| `'count'` \| `'decimal'` — implies suffix and comma rules |
| `layout` | str | no | `'scalar'` (default) \| `'bracket_list'` for `[a, b]` answers |

Deterministic — no LLM call.

## Composition rules

1. **Chain typing.** In `op_a --> op_b`, `OutputType(op_a) ∈ AcceptedInputs(op_b)`. Validator checks this statically.
2. **Parallel typing.** `[c1; c2; ...] --> op_next` requires each `c_i` to terminate in the same `Output` type `T`, and `op_next` to accept `list[T]`.
3. **Chain head.** Must be `retrieve` or `lookup_external` (the only ops that take no input).
4. **Chain tail.** Must be `format` for an answer-producing chain. Sub-chains feeding a parallel may end in any type.
5. **Concepts/constraints scope.** `OpNode.constraints` apply to that op only; `ChainNode.global_constraints` apply to every op in the chain.
6. **Determinism budget.** `format` is fully deterministic. `compute` is deterministic per-call but its `code` is LLM-generated. `retrieve` / `extract` / `read_visual` / `lookup_external` involve LLMs at runtime.

## Worked examples

All examples use single-quoted string args and flat `format(...)` kwargs.

**UID0001** — Total US national defense expenditures for CY1940:
```
retrieve(concept='national_defense_expenditure', period='CY1940')
  --> extract(concept='total national defense expenditures, annual', mode='value')
  --> format(unit='usd_millions', precision=0)
```

**UID0004** — Absolute pct change in CY1953 vs CY1940 monthly national defense:
```
[
  retrieve(concept='national_defense_expenditure', period='CY1940')
    --> extract(concept='monthly national defense expenditures', mode='list')
    --> compute(code='result = sum(prev.value)');
  retrieve(concept='national_defense_expenditure', period='CY1953')
    --> extract(concept='monthly national defense expenditures', mode='list')
    --> compute(code='result = sum(prev.value)')
]
  --> compute(code='a, b = prev[0].value, prev[1].value; result = abs((b - a) / a) * 100')
  --> format(unit='pct', precision=2)
```

**UID0029** — "According to the bulletin published in June 1970, average yield spread CY1960-69":
```
retrieve(concept='bond_yields', period='CY1960..CY1969', source_bulletin='1970-06')
  --> extract(concept='Aa corporate vs Treasury yield spread, monthly', mode='list')
  --> compute(code='vs = prev.value; result = sum(vs) / len(vs)')
  --> format(precision=5)
```

**UID0030** — Local maxima on line plots, page 5 of Sept 1990 bulletin:
```
retrieve(concept='line_plots_on_page', period='1990-09', source_bulletin='1990-09')
  --> read_visual(concept='count of local maxima across all line plots on the page')
  --> format(unit='count')
```

**UID0035** — Benford first-digit count on a whole table (`mode='table'`):
```
retrieve(concept='receipts_table', period='1980-05', source_bulletin='1980-05')
  --> extract(concept='receipts data table on page 41', mode='table')
  --> compute(code='df = prev.value; result = sum(1 for v in df.values.flatten() if str(v).startswith("1"))')
  --> format(unit='count')
```

**UID0055** — WWII end → Korean War start (event_year resolution):
```
[
  lookup_external(resource='event_year', event='wwii_end');
  lookup_external(resource='event_year', event='korean_war_start')
]
  --> compute(code='y1, y2 = prev[0].value, prev[1].value; result = (f"CY{y1}", f"CY{y2}")')
  --> retrieve(concept='moody_aaa_corporate_bond_yield', period='prev')
  --> extract(concept='annual avg Moody Aaa bond yield, both years', mode='list')
  --> compute(code='vs = prev.value; result = abs(vs[1] - vs[0])')
  --> format(unit='pct', precision=1)
```
`period='prev'` is a sentinel: the period is bound from the upstream value at execution time.

## Validation

`data/dsl_planning_pass.csv` contains a validated plan for every question in `data/officeqa_pro.csv` (133 / 133 = 100% parse + validate, generated 2026-05-08 with `tools/plan_dsl_with_gemini.py`). Use it as the planner's few-shot pool.
