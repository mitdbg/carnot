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
PageRef         { file_path: str, year: int, month: "YYYY-MM", page: int }
DocHandle       { refs: [PageRef], desc: str }                # non-empty refs invariant
TypedValue      { value: Any, dtype: str, desc: str }         # dtype: scalar:<unit>, list[<inner>], df, …
FormattedString { text: str, desc: str }
Output          = DocHandle | TypedValue | FormattedString
```

`PageRef.page` is the **1-based PDF page index** — the only page-number meaning used in the codebase. It matches `source_docs?page=N` from the Fraser benchmark URLs. The bulletin's printed-page footer is recoverable via `skunk.common.parsed_json.get_printed_page(ref, ctx)` for trace/prompt enrichment but is never used as a lookup key.

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

**Behavior.** Looks up matching pages via the retrieve subagent's internal page index (filtered by concept + period + optional source_bulletin pin), returns a `DocHandle` whose `PageRef`s are fully specified (`page` is the 1-based PDF page index).

**Failure.** `StepFailed("retrieve", "no candidates after pre-filter")`.

### `extract() → TypedValue`

No arguments.

**Pre.** `prev` is `DocHandle`. **Post.** `TypedValue` with `dtype='named'` whose `.value` is a dict mapping snake_case names to scalars/lists/tables relevant to `ctx.question`.

**Behavior.** Per-page tier dispatch — first tier that returns a non-empty dict wins:
1. Tier 1 — JSON-parsed page elements via `skunk.common.parsed_json.get_text_for_pdf_page`.
2. Tier 2 — PyMuPDF native text per PDF page (`cache/pages/{month}/p{NNN}.txt`).
3. Tier 3 — PNG render (PyMuPDF, 300 dpi) sent to vision LLM.

The agent reads the page and the user's question, then emits every value/list/table that could plausibly be needed to answer it, with disambiguating names (e.g. `national_defense_cy1940`, `national_defense_fy1940`, `jpy_holdings_mar_2025`). It does not pre-compute or aggregate.

**Failure.** `StepFailed("extract", "no relevant values found across tiers")`.

### `read_visual() → TypedValue`

No arguments. Same output shape as `extract`, but always Tier 3 vision. Used for charts and figures.

### `lookup_external(nl) → TypedValue`  *(chain-head capable)*

| arg | type | req | description |
|---|---|---|---|
| `nl` | str | yes | Natural language description of the external data to look up |

Single Gemini call. Returns a typed value — the subagent infers the appropriate `dtype` (e.g. `scalar:year`, `scalar:fx_rate`, `scalar:cpi`, `list[scalar:fx_rate]`). Strings are also supported (named entities, place names) with `dtype='text'`. Use for any factual data that can't be extracted from the bulletin corpus: CPI-U, exchange rates, event dates, bureau names, etc.

### `compute() → FormattedString`  *(chain terminator)*

No arguments. Compute is the only chain terminator and produces the final answer string.

**Pre.** `prev` is `TypedValue` (typically `dtype='named'`) or `list[TypedValue]` (parallel branches). **Post.** `FormattedString`.

**Behavior.** Reads `ctx.question` and `prev`, then runs:
1. **Plan + codegen** (Gemini): emits either `CODE\n<python>` that assigns a string to `result`, OR `MISSING:<reason>` if extracted values are insufficient.
2. **Exec** (sandbox): runs the code with `prev` in scope; sandbox imports are numpy/pandas/statsmodels/math.
3. **Verifier** (Gemini): checks the output's *form* (precision, unit, percent vs decimal, comma rules, list bracketing) against the question. PASS → return; FAIL with reason → retry.

Up to 3 attempts; each retry sees feedback from the prior failure (exec exception or verifier rejection). Compute owns formatting in addition to derivation — there is no separate format op.

**Failure.** `StepFailed("compute", ...)` for `MISSING:<reason>` (insufficient input) or after all attempts fail.

## Composition rules

1. **Chain typing.** In `op_a --> op_b`, `OutputType(op_a) ∈ AcceptedInputs(op_b)`. Validator checks this statically.
2. **Parallel typing.** `[c1; c2; ...] --> op_next` requires each `c_i` to terminate in the same `Output` type `T`, and `op_next` to accept `list[T]`.
3. **Chain head.** Must be `retrieve` or `lookup_external` (the only ops that take no input).
4. **Chain tail.** Must be `compute` for an answer-producing chain — it produces the final string. Sub-chains feeding a parallel may end in any type.
5. **Concepts/constraints scope.** `OpNode.constraints` apply to that op only; `ChainNode.global_constraints` apply to every op in the chain.
6. **Determinism budget.** `compute` makes 2–3 Gemini calls per attempt (codegen + verifier); all other ops (`retrieve`, `extract`, `read_visual`, `lookup_external`) also involve LLM calls at runtime. Temperature is 0 throughout.

## Worked examples

`extract()` and `compute()` take no arguments. The agent reads `ctx.question` and decides what to do.

**UID0001** — Total US national defense expenditures for CY1940:
```
retrieve(concept='national_defense_expenditure', period='CY1940')
  --> extract()
  --> compute()
```

**UID0004** — Absolute pct change in CY1953 vs CY1940 monthly national defense:
```
[
  retrieve(concept='national_defense_expenditure', period='CY1940') --> extract();
  retrieve(concept='national_defense_expenditure', period='CY1953') --> extract()
]
  --> compute()
```

**UID0029** — "According to the bulletin published in June 1970, average yield spread CY1960-69":
```
retrieve(concept='bond_yields', period='CY1960..CY1969', source_bulletin='1970-06')
  --> extract()
  --> compute()
```

**UID0030** — Local maxima on line plots, page 5 of Sept 1990 bulletin:
```
retrieve(concept='line_plots_on_page', period='1990-09', source_bulletin='1990-09')
  --> read_visual()
  --> compute()
```

**UID0035** — Benford first-digit count on a whole table:
```
retrieve(concept='receipts_table', period='1980-05', source_bulletin='1980-05')
  --> extract()
  --> compute()
```

**UID0055** — Year WWII ended and Korean War started:
```
[
  lookup_external(nl='year that WWII ended');
  lookup_external(nl='year the Korean War started')
]
  --> compute()
```

## Validation

The plan cache lives at `data/dsl_planning_pass.csv` and is regenerated by `tools/plan_dsl_with_gemini.py` (or lazily by `skunk.run` on first execution per UID). Each row is `uid, question, plan_text` and is loaded via `skunk.run.load_plan_cache`.
