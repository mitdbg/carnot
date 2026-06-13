# Full-Corpus OCR Cleaning Design

## Goal

Run an LLM validation and correction stage over the full extracted Treasury
Bulletin table corpus. Each task is one extracted table image plus its parsed
OCR table text. The stage should identify likely OCR mistakes, then produce
corrections for tables that appear corrupt.

The run must be cache-aware, resumable, and auditable because it will process a
large number of table images.

## Inputs

Full-corpus input:

```text
/orcd/home/002/gerarvit/orcd/scratch/officeqa/extracted_tables/tables.json
```

Dev-sample input:

```text
/orcd/home/002/gerarvit/orcd/scratch/officeqa/ocr_cleaning/dev_samples/extracted_tables/tables.json
```

Each table row should provide:

```text
id
bulletin
page
table_index
image_path or image_link
parsed_ocr
```

## Outputs

Default full-run output path:

```text
/orcd/home/002/gerarvit/orcd/scratch/officeqa/ocr_cleaning/full_run/
```

Expected output files:

```text
verdicts.jsonl
correction_tables.jsonl
corrections.jsonl
raw_responses.jsonl
summary.json
run_state.json
```

Do not store full prompts in outputs. Store lightweight provenance only:

```text
model
prompt_version
stage
response_format
timestamp or run_id
usage
cache_hit
estimated_cost_usd
```

## CLI Shape

Full run:

```bash
python scripts/run_full_ocr_cleaning.py \
  --output-dir /orcd/home/002/gerarvit/orcd/scratch/officeqa/ocr_cleaning/full_run \
  --model gemini/gemini-3.1-pro-preview \
  --max-workers 20 \
  --checkpoint-interval 5000
```

Dev mode:

```bash
python scripts/run_full_ocr_cleaning.py --dev
```

`--dev` should switch input defaults to the 27-bulletin dev sample and use a
dev output directory. Normal mode should use the full extracted-table corpus.

## Model

Use:

```text
gemini/gemini-3.1-pro-preview
```

The script should call the model through:

```text
scripts/tools/llm_wrapper.py
```

The wrapper provides:

- request caching
- parallel execution
- retry handling
- rate limiting
- usage metadata from Gemini responses

## Two-Pass Flow

### Pass 1: Verdict

Pass 1 runs over every table. It is a cheap detector pass.

Prompt output contract:

```text
C|<confidence>
E|<confidence>
U|<confidence>
```

Labels:

```text
C = clean: visible numeric OCR values match the image
E = error: at least one visible numeric OCR value appears wrong
U = uncertain: image is unreadable, cropped badly, or cannot be checked
```

Output file:

```text
verdicts.jsonl
```

One row per table:

```json
{
  "table_id": "1982_07_84_1",
  "bulletin": "1982-07",
  "page": 84,
  "table_index": 1,
  "verdict": "E",
  "confidence": 0.91,
  "raw_response": "E|0.91",
  "parse_status": "ok",
  "model": "gemini/gemini-3.1-pro-preview",
  "prompt_version": "verdict_v1",
  "usage": {},
  "estimated_cost_usd": 0.0,
  "cache_hit": false
}
```

### Pass 2: Corrections

Pass 2 runs only for tables where Pass 1 returns `E`.

Pass 2 must double check the table. It should not blindly assume Pass 1 was
right. It can return clean or uncertain if the suspected table is not actually
corrupt.

Correction output should support multiple corrections per table.

Suggested compact response protocol:

```text
E|<confidence>
X|<ocr_text>|<corrected_text>|<row_hint>|<column_hint>
X|<ocr_text>|<corrected_text>|<row_hint>|<column_hint>
```

If Pass 2 rejects the error:

```text
C|<confidence>
```

If Pass 2 cannot verify:

```text
U|<confidence>
```

Table-level Pass 2 output:

```text
correction_tables.jsonl
```

One row per table sent to Pass 2:

```json
{
  "table_id": "1982_07_84_1",
  "status": "confirmed_error",
  "confidence": 0.94,
  "correction_count": 2,
  "model": "gemini/gemini-3.1-pro-preview",
  "prompt_version": "correction_v1",
  "usage": {},
  "cache_hit": false
}
```

Correction-level output:

```text
corrections.jsonl
```

One row per correction:

```json
{
  "table_id": "1982_07_84_1",
  "correction_index": 0,
  "ocr_text": "42.72",
  "corrected_text": "12.72",
  "row_hint": "Series E and H",
  "column_hint": "Accrued discount",
  "confidence": 0.94
}
```

## Raw Responses

Store every raw model response in:

```text
raw_responses.jsonl
```

One row per LLM call:

```json
{
  "table_id": "1982_07_84_1",
  "stage": "verdict",
  "raw_response": "E|0.91",
  "model": "gemini/gemini-3.1-pro-preview",
  "prompt_version": "verdict_v1",
  "usage": {},
  "estimated_cost_usd": 0.0,
  "cache_hit": false
}
```

Again: do not store prompt text.

## Resume Logic

Resume should use both output files and the wrapper cache.

On startup:

1. Load `verdicts.jsonl` and collect completed Pass 1 table IDs.
2. Load `correction_tables.jsonl` and collect completed Pass 2 table IDs.
3. Skip Pass 1 for tables already present in `verdicts.jsonl`.
4. Queue Pass 2 only for Pass 1 `E` rows not already present in
   `correction_tables.jsonl`.
5. Keep `LLMWrapper` cache enabled as a billing backstop.

If duplicate rows exist, the resume loader should keep the latest row for each
`(stage, table_id)` pair.

## Checkpointing

The runner should pool all table tasks and checkpoint periodically.

Default checkpoint interval:

```text
5000 completed LLM calls
```

API batches can be smaller than the checkpoint interval to avoid loading
thousands of table images into memory at once. If `--batch-size` is omitted,
the runner should choose a memory-safe default derived from `--max-workers`.

At each checkpoint:

- append buffered output rows to disk
- flush file handles
- call `wrapper.flush_cache(wait=True)`
- update `run_state.json`

Also flush at normal exit and on keyboard interrupt where possible.

## Summary

Write/update:

```text
summary.json
```

Include:

```text
total_tables
pass1_completed
pass1_clean
pass1_error
pass1_uncertain
pass1_unparseable
pass2_completed
pass2_confirmed_error
pass2_rejected_clean
pass2_uncertain
total_corrections
total_usage
non_cached_usage
estimated_cost_usd
cache_hits
cache_misses
elapsed_seconds
```

Usage should come from Gemini response metadata:

```text
prompt_token_count
candidates_token_count
total_token_count
thoughts_token_count
cached_content_token_count
```

## Prompt Versions

Hardcode prompt templates in the script and version them:

```text
verdict_v1
correction_v1
```

Output rows should store only the prompt version, not the full prompt.

## Implementation Notes

Prefer a new full-corpus runner instead of extending the feasibility pilot:

```text
scripts/run_full_ocr_cleaning.py
```

Use:

```text
LLMWrapper.batch_call_llm_vision_with_usage(...)
```

Use the shared wrapper cache by default. Allow override with `--cache-path`.

Keep outputs append-only and resume-safe. The script should be able to restart
after a crash without losing paid work or reprocessing completed table IDs.
