# OCR Table Validation Feasibility Report

## Goal

The feasibility study asks whether a multimodal LLM can compare an extracted
table image with its OCR table text and detect PDF-to-JSON OCR mistakes. The
first benchmark uses synthetic numeric typos so accuracy can be measured
against known injected errors.

## Data Baseline

Local baseline from `extracted_tables/tables.json`:

- 750 table rows.
- 750 measured table PNGs.
- OCR text length in characters:
  - min: 208
  - median: 2,580
  - mean: 3,464.3
  - p90: 7,588
  - max: 13,771
- Image size:
  - median: 1.627 MP
  - p90: 3.824 MP
  - max: 4.57 MP

Estimated prompt size is about 2.4k median tokens or 2.6k mean tokens per
table, before exact provider-reported usage.

## Model And Cost

The pilot defaults to:

```text
gemini/gemini-3.5-flash
```

Earlier rough estimates put Gemini-style review in the low cents per dozen
tables range, but exact cost should come from provider usage metadata rather
than static estimates.

The Gemini SDK exposes usage metadata after each call:

```text
prompt_token_count
candidates_token_count
total_token_count
thoughts_token_count
cached_content_token_count
```

We added a non-breaking usage-returning path to `scripts/tools/llm_wrapper.py`
so the feasibility runner can record actual token usage and convert it to cost.
Existing wrapper methods that return only strings are left intact.

## Pilot Design

The runner is:

```text
scripts/run_ocr_feasibility_pilot.py
```

For each table:

1. Read table metadata from sample `extracted_tables/tables.json`.
2. Resolve the symlinked table crop.
3. For corrupted cases, inject numeric digit substitutions into `parsed_ocr`.
4. For clean controls, keep `parsed_ocr` unchanged.
5. Send `(image, OCR HTML)` to the model through `LLMWrapper`.
6. Parse the JSON verdict.
7. Score table-level detection and rough correction recovery.

The script uses parallel cached calls from `scripts/tools/llm_wrapper.py`.

## Output Schema

Each result row includes:

```text
table_id
bulletin
page
split
mutations
raw_response
parsed_response
score
```

The summary reports:

```text
true_positives
false_negatives
false_positives
true_negatives
uncertain
unparseable
table_recall
table_precision
clean_false_positive_rate
mutation_recovery_rate
```

The runner now also records parse diagnostics and salvages a top-level verdict
from truncated raw JSON when possible.

## Observed Results

```text
model: gemini/gemini-3.1-pro-preview
output_format: JSON
elapsed_seconds: 275.5540289878845
seconds_per_request: 1.83702685991923
consumption: 150 requests; token usage not reported
cost: not reported
precision: 0.7092198581560284
recall: 1.0
true_positives: 100
false_positives: 41
false_negatives: 0
true_negatives: 9
```

```text
model: gemini/gemini-3.5-flash
output_format: JSON
elapsed_seconds: 163.44785237312317
seconds_per_request: 1.0896523491541545
consumption: 150 requests; token usage not reported
cost: not reported
precision: 0.7058823529411765
recall: 0.96
true_positives: 96
false_positives: 40
false_negatives: 4
true_negatives: 7
```

## Interpretation

The model is highly sensitive to injected errors, but it over-flags clean
controls. The first run's original `unparseable` count was caused mainly by
truncated JSON output. After salvaging top-level verdicts, the main problem is
false positives, not recall.

## Recommendations

Next run:

```bash
python scripts/run_ocr_feasibility_pilot.py \
  --output-dir cleaning_v2 \
  --model gemini/gemini-3.5-flash \
  --n-corrupt 100 \
  --n-clean 50 \
  --max-output-tokens 4096
```

Use exact usage metadata from the new wrapper path to compute actual spend.

Prompt changes to test:

- Ask for one suspect cell maximum.
- Tell the model to prefer `clean` unless a specific numeric mismatch is
  clearly visible.
- Require `uncertain` when the crop is hard to read.
- Remove long explanations to reduce truncation.

Evaluation target:

- Keep recall high.
- Reduce clean false positive rate substantially.
- Keep unparseable responses under 5%.