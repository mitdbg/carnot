# OCR Table Validation Feasibility Study

## Purpose

This feasibility study evaluates whether a multimodal LLM can review an
extracted table crop plus its parsed OCR table text and reliably identify OCR
mistakes. The target workflow is a quality-control pass for the PDF-to-JSON
extraction path, starting with the 27-bulletin OCR-cleaning dev sample.

The first pilot should focus on synthetic numeric typos injected into
`parsed_ocr`. This gives a measurable benchmark before using the model on
unknown real OCR errors.

## Data Baseline

Local baseline from `extracted_tables/tables.json`:

- 750 extracted table rows.
- 750 table PNGs measured.
- OCR text length, in characters:
  - Minimum: 208
  - Median: 2,580
  - Mean: 3,464.3
  - P90: 7,588
  - Maximum: 13,771
- Table image size, in megapixels:
  - Median: 1.627 MP
  - P90: 3.824 MP
  - Maximum: 4.57 MP

Estimated prompt size per table:

- Median input: about 2.4k tokens.
- Mean input: about 2.6k tokens.
- Output budget: about 300 tokens for a compact structured verdict.

The 27-bulletin dev sample will likely contain about 2,000 table rows if it has
roughly the same density as the local 1939 dump.

## Cost Estimate

Gemini 3.5 Flash is the requested default model for the pilot. As of
2026-06-13, Google's official Gemini API pricing page lists Gemini 3.5 Flash
standard paid pricing at $1.50 per 1M input tokens and $9.00 per 1M output
tokens. The same page lists batch pricing at $0.75 per 1M input tokens and
$4.50 per 1M output tokens.

Using the mean estimate of 2.6k input tokens and 300 output tokens per table:

- Standard Gemini 3.5 Flash: about $0.0066 per table.
- Batch/Flex Gemini 3.5 Flash: about $0.0033 per table.
- 750-table local dump, standard: about $4.95.
- 2,025-table dev sample, standard: about $13.37.
- 2,025-table dev sample, batch/flex: about $6.68.

These are order-of-magnitude estimates. The real bill depends on actual image
tokenization, output length, retries, provider tier, and whether batch/flex
inference is used. Pricing should be checked at run time against the official
Gemini API pricing page.

Sources:

- Gemini API pricing: https://ai.google.dev/gemini-api/docs/pricing
- Gemini token counting and media tokenization: https://ai.google.dev/gemini-api/docs/tokens

## Time Estimate

Sequential execution:

- 3-10 seconds per table is a reasonable first estimate.
- About 2,000 tables would take roughly 1.5-6 hours sequentially.

Parallel execution:

- With 20 workers, the 2,000-table sample is likely in the 5-25 minute range.
- Provider rate limits, retry behavior, and image upload overhead will dominate.
- The existing `scripts/tools/llm_wrapper.py` has parallel batch calls, request
  caching, retry handling, and rate-limit state, so the pilot should use it
  rather than implementing new API plumbing.

## Accuracy Unknowns

Accuracy is the binding feasibility question. It should not be assumed from
model quality or low cost.

Likely easy cases:

- Isolated digit substitutions in clear numeric cells.
- Large visual discrepancies where one number is obviously different.
- Tables with simple row and column structure.

Likely hard cases:

- Dense scanned tables with small type.
- Row or column misalignment.
- Superscripts, footnotes, and ditto marks.
- Negative signs, decimal points, and comma placement.
- Multi-line cells and split headers.
- Cases where the OCR text is internally plausible but visually wrong.

A binary prompt such as "are there any mistakes?" is too weak. It will hide
false negatives and will not produce enough detail to score corrections.

## Pilot Design

Use the sampled `extracted_tables/tables.json` and table PNG symlinks generated
by `scripts/build_ocr_cleaning_sample.py`.

For each selected table:

1. Read the original `parsed_ocr`.
2. Inject one or more numeric character typos into numeric-looking tokens.
3. Send the table image plus corrupted OCR table text to Gemini 3.5 Flash.
4. Ask the model to identify and correct any OCR mistakes.
5. Score the model response against the known injected changes.

Initial typo classes:

- `1` <-> `4`
- `5` <-> `8`
- `0` <-> `8`
- `3` <-> `8`
- `6` <-> `8`
- `2` <-> `7`

Later typo classes:

- Dropped or inserted commas.
- Dropped or inserted decimal points.
- Dropped or inserted minus signs.
- Extra or missing leading/trailing zeroes.

The first pilot should inject numeric character substitutions only. That keeps
the benchmark simple and gives clear expected answers.

## Recommended Prompt Shape

Use a strict JSON output schema:

```json
{
  "verdict": "clean|has_errors|uncertain",
  "confidence": 0.0,
  "suspect_cells": [
    {
      "ocr_text": "string from corrupted OCR",
      "corrected_text": "string read from image",
      "row_hint": "nearby row label or empty",
      "column_hint": "nearby column label or empty",
      "reason": "brief evidence"
    }
  ]
}
```

Prompt requirements:

- Compare the rendered table image against the OCR HTML.
- Focus on numeric OCR mistakes.
- Do not flag formatting differences unless they change numeric content.
- Return `clean` only if no numeric discrepancy is visible.
- Return `uncertain` when the image is unreadable or the cell cannot be located.
- Keep the output compact and valid JSON.

## Metrics

Table-level metrics:

- Recall: fraction of corrupted tables flagged as `has_errors`.
- Precision: fraction of flagged tables that really contain injected errors.
- False positive rate on clean controls.
- Uncertain rate.

Cell/string-level metrics:

- Injected typo detection rate.
- Correct original value recovery rate.
- Exact correction accuracy.
- Over-correction rate, where the model changes clean values.

Recommended evaluation set:

- 100-200 corrupted tables.
- 50-100 clean control tables.
- Stratify across decades, issue months, image quality, and table length.

## Go/No-Go Metrics

A practical first-pass threshold:

- At least 90% table-level recall on injected numeric typos.
- At most 10% false positive rate on clean controls.
- At least 80% exact correction accuracy for flagged injected cells.
- Low enough `uncertain` rate that the workflow does not simply defer most work
  to humans.

If Gemini 3.5 Flash misses subtle errors but catches obvious ones, use it as a
triage model. Route `uncertain` cases and high-value tables to a stronger model
or to manual review.

## Recommended Implementation

Build a script that:

- Reads the dev-sample `extracted_tables/tables.json`.
- Samples table rows deterministically with a fixed random seed.
- Creates corrupted OCR variants by substituting likely-confused numeric
  characters in numeric-looking tokens.
- Sends `(prompt, image_bytes)` requests through
  `scripts/tools/llm_wrapper.py`.
- Uses `LLMWrapper.batch_call_llm_vision(...)` for parallel cached calls.
- Defaults to `gemini/gemini-3.5-flash`.
- Writes JSONL records with source table metadata, injected mutations, raw model
  response, parsed response, and scoring fields.
- Writes a summary JSON with recall, precision, false-positive rate,
  correction accuracy, elapsed time, and model/config metadata.

This should be run first on a small smoke sample, then scaled to the full
27-bulletin dev set once prompts and scoring are stable.
