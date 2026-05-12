# DSL — formal specification

The OfficeQA pipeline is encoded as a **Plan** whose top-level structure is a chain of one or more `compute` nodes terminating in a final aggregator. The planner emits this as text (or JSON); the orchestrator walks the chain and dispatches subagent calls.

## Plan shape

A plan is one of:

```
Flat (one compute node):
  single retrieve branch:    retrieve(...) --> extract(...) --> compute()
  single lookup branch:      lookup_external(...) --> compute()
  parallel of branches:      [ branch ; branch ; ... ] --> compute()

Decomposed (parallel sub-computes + final aggregator):
  [
    [ branches_1 ] --> compute(task='...');
    [ branches_2 ] --> compute(task='...')
  ] --> compute()
```

The trailing `compute()` is always the **final aggregator** (no args). Each inner `[branches] --> compute(task='...')` is an **intermediate compute** that consumes its own data branches and emits `list[AnnotatedValue]` for the aggregator. Decomposition is useful when the question contains two or more distinct calculations whose outputs are themselves operands for a later step.

Plan depth is bounded by `SkunkConfig.max_compute_depth` (default `2`):
  - depth 1 = legacy flat (one compute node).
  - depth 2 = one intermediate layer + a final aggregator.

The grammar does not support chains that skip `extract` after `retrieve`, nesting intermediate computes inside other intermediate computes, or `compute` anywhere other than as the chain terminator — the AST makes those unrepresentable.

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

Flat (back-compat shape — what the planner emits today):

```json
{
  "branches": [
    {"kind": "retrieve", "concept": "national_defense", "period": "CY1940"},
    {"kind": "lookup_external", "nl": "USD/JPY exchange rate on 2025-03-31"}
  ]
}
```

`from_dict` rewrites this to `Plan(computes=[ComputeNode(branches=[...], final=True)])`.

Decomposed (explicit chain — what Phase 2 planner can emit):

```json
{
  "computes": [
    {"task": "Sum 1940 monthly defense expenditures",
     "branches": [{"kind": "retrieve", "concept": "national_defense", "period": "CY1940"}]},
    {"task": "Sum 1953 monthly defense expenditures",
     "branches": [{"kind": "retrieve", "concept": "national_defense", "period": "CY1953"}]}
  ]
}
```

`from_dict` appends an empty-branches final aggregator if the input doesn't include one explicitly. `to_dict` round-trips back to the legacy `{"branches": [...]}` shape for a single-compute plan, or emits `{"computes": [...]}` for a decomposed plan.

A simple chain has one element in `branches`. A parallel has two or more.

## Inter-op types

```
PageRef         { month: 'YYYY-MM', page: int, file_path: str }
                  __post_init__ requires month when page is set.
DocHandle       { refs: [PageRef], desc: str }
AnnotatedValue  { description: str, value: Any, unit: str,
                  kind: 'scalar' | 'vector' | 'table',
                  index_name?: str,          # vector only
                  row_name?: str, col_name?: str  # table only
                }
FormattedString { text: str }
```

`PageRef.page` is the **1-based PDF page index** — the only page-number meaning used in the codebase. It matches `source_docs?page=N` from the Fraser benchmark URLs.

`AnnotatedValue.description` is the free-text label that uniquely identifies the datum — series name, period, sub-category, unit hint, etc. Both `extract` and `lookup_external` return `list[AnnotatedValue]`.

Each entry has one of three **kinds**, with payload shapes enforced in the extract parser before an `AnnotatedValue` is constructed:

- `kind='scalar'` — `value` is `int | float | str`. Single facts.
- `kind='vector'` — `value` is a flat dict `{index_label: scalar}`, with `index_name` naming the varying dim (e.g. `'month'`). Used for 1-D series.
- `kind='table'` — `value` is a 2-level dict `{row_label: {col_label: scalar}}` with consistent column keys across rows, and `row_name` / `col_name` naming the two varying dims.

**No nesting beyond these shapes.** Vector cells and table cells must be primitive scalars (int/float/str). A vector-of-vectors, table-with-list-cells, or 3-level dict is rejected by the extract parser and dropped before reaching compute.

**Quorum is all-or-nothing per entry.** When extract runs N samples, samples are bucketed by the canonical form of the *whole* entry (name + kind + axis names + every cell value + unit + dims). Only buckets meeting `extract_quorum` survive. There is no per-cell voting — a single-cell disagreement keeps both variants in separate buckets, and neither is kept unless quorum is hit on the exact match.

## Period grammar

Used by `RetrieveBranch.period`.

```
period      ::= point | range | enumeration
point       ::= "CY"YYYY | "FY"YYYY | "Q"[1-4]"-"YYYY | YYYY"-"MM | YYYY"-"MM"-"DD | YYYY
range       ::= point ".." point                                  # inclusive
enumeration ::= point ("," point)+                                # multi-point list
```

Examples: `CY1940`, `FY1981`, `Q3-1982`, `2025-03-31`, `CY1940..CY1949`, `1991-06,1996-06,2001-06`.

## Operators

### `retrieve(concept, period, source_bulletin?) → DocHandle`

Looks up matching pages via the retrieve subagent's internal page index (filtered by concept + period + optional source_bulletin pin), returns a `DocHandle` whose `PageRef`s are fully specified.

**Failure.** `StepFailed("retrieve", "no candidates after pre-filter")`.

### `extract(visual_only?) → list[AnnotatedValue]`

| arg | type | req | description |
|---|---|---|---|
| `visual_only` | bool | no | Skip Tiers 1-2; vision-only |

Per-page tier dispatch:
1. Tier 1 — JSON-parsed page elements (structured text + HTML tables).
2. Tier 2 — PyMuPDF native text per PDF page (`cache/pages/{month}/p{NNN}.txt`).
3. Tier 3 — PNG render at 300 dpi sent to vision LLM.

Returns a list of `AnnotatedValue` entries — each one a `scalar`, `vector`, or `table` per the kind contract above.

**Failure.** `StepFailed("extract", "no relevant values found across tiers")`.

### `lookup_external(nl) → list[AnnotatedValue]`

Single Gemini call. Returns a list with one `AnnotatedValue` carrying the looked-up value, an inferred unit, and an inferred kind.

### `compute(task?, final?) → FormattedString | list[AnnotatedValue]`

Compute has two modes, switched by `op.args["final"]` (default `True` when the orchestrator omits the arg, preserving legacy behavior):

**Final mode (`final=True`)** — chain terminator. Reads `ctx.question` (or `op.args["task"]` if planner-supplied) and `prev` (`list[AnnotatedValue]`). Runs:

1. **Codegen** (Gemini): emits `CODE\n<python>` or `MISSING:<reason>`.
2. **Exec** (sandbox): runs the code with `prev` in scope; available imports are numpy, pandas, statsmodels, math.
3. **Self-critique** (Gemini): checks the output's *form* (precision, unit, percent vs decimal, comma rules, list bracketing) against the question. ACCEPT ships; REVISE triggers one more codegen pass, shipped unconditionally.

Returns `FormattedString` (the answer the user sees).

**Intermediate mode (`final=False`)** — invoked by the orchestrator on each non-final ComputeNode in a decomposed plan. Reads `op.args["task"]` and `prev`. Codegen prompt is tuned for raw values: `result` is set to a Python scalar/dict/dict-of-dict (or an explicit `list[AnnotatedValue]`), with optional `result_unit` / `result_kind` / `result_description` / `result_index_name` / `result_row_name` / `result_col_name` to refine the wrapping. The runtime wraps `result` into one `AnnotatedValue` (or passes through an explicit list). No self-critique.

Returns `list[AnnotatedValue]` for the downstream aggregator.

**MissingData recovery (final only).** If the final compute reports `MISSING:<reason>`, the orchestrator catches it as a `MissingData` exception and runs a single recovery round: the recovery planner emits one supplemental `Branch` (typically a `LookupBranch`), the orchestrator executes it, augments `prev`, and retries compute. Intermediate computes do NOT get recovery — a missing intermediate is logged and its slot is skipped.

**Failure modes.** `StepFailed("compute", ...)` after all attempts fail or recovery declines/exhausts (final), or codegen retry budget exhausts (intermediate).

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

**Decomposed** — Two parallel sub-computes feed a final aggregator. Each sub-compute is scoped to its own sub-task, so its codegen prompt sees a narrower question and a smaller `prev`:

```
[
  [ retrieve(concept='national_defense', period='CY1940') --> extract() ] --> compute(task='Sum the 1940 monthly defense expenditures');
  [ retrieve(concept='national_defense', period='CY1953') --> extract() ] --> compute(task='Sum the 1953 monthly defense expenditures')
] --> compute()
```

The plan cache lives at `data/dsl_planning_pass.csv` and is regenerated lazily by `skunk.run` on first execution per UID. Each row is `uid, question, plan_text` and is loaded via `skunk.run.load_plan_cache`.
