# OCR Cleaning Steps

## Corpus

The cleaning work targets the U.S. Treasury Bulletin corpus. Each source PDF is
one bulletin issue, named by publication month, for example
`treasury_bulletin_1940_06.pdf`. The cleaning task compares source PDF/table
images against parsed JSON and extracted table OCR.

## Sample

The OCR-cleaning dev sample contains 27 bulletins:

- 1940s: 1940, 1941, 1942
- 1960s: 1960, 1961, 1962
- 1980s: 1980, 1981, 1982
- Issues per year: June, July, August

This is:

```text
3 decades x 3 years x 3 adjacent issues = 27 bulletins
```

## Server Paths

The current scripts use fixed ORCD paths:

```text
/orcd/home/002/gerarvit/orcd/scratch/officeqa/
```

Inputs:

```text
treasury_bulletin_pdfs_no_ocr/
treasury_bulletins_parsed/jsons/
extracted_tables/tables.json
extracted_tables/images/
```

Generated sample:

```text
/orcd/home/002/gerarvit/orcd/scratch/officeqa/ocr_cleaning/dev_samples/
```

Pilot outputs:

```text
/orcd/home/002/gerarvit/orcd/scratch/officeqa/ocr_cleaning/feasibility_pilot/
```

## Build The Sample

Run:

```bash
cd /home/gerardo/carnot/skunk
source .envrc
python scripts/build_ocr_cleaning_sample.py
```

The builder creates:

```text
manifest.csv
manifest.json
pdfs/
parsed_json/
extracted_tables/tables.json
extracted_tables/images/
```

Behavior:

- Symlinks selected source PDFs into `pdfs/`.
- Symlinks selected parsed JSON files into `parsed_json/`.
- Filters the full extracted-table manifest to only sampled bulletins.
- Writes the filtered table manifest to `extracted_tables/tables.json`.
- Symlinks matching extracted table PNGs into `extracted_tables/images/`.
- Refuses to overwrite non-symlink files in generated symlink directories.

## Run The Feasibility Pilot

Smoke test without LLM calls:

```bash
python scripts/run_ocr_feasibility_pilot.py \
  --n-corrupt 5 \
  --n-clean 3 \
  --dry-run
```

Run the pilot:

```bash
python scripts/run_ocr_feasibility_pilot.py \
  --n-corrupt 100 \
  --n-clean 50 \
  --max-workers 20
```

Useful explicit output path:

```bash
python scripts/run_ocr_feasibility_pilot.py \
  --output-dir cleaning \
  --model gemini/gemini-3.5-flash \
  --n-corrupt 100 \
  --n-clean 50 \
  --max-output-tokens 4096
```

The pilot writes:

```text
planned_requests.jsonl
results.jsonl
summary.json
llm_wrapper_cache.pckl
```

## Pilot Logic

For corrupted examples, the script injects synthetic numeric OCR mistakes into
`parsed_ocr`, then asks the vision model to compare the corrupted OCR HTML with
the table crop.

Current digit confusions:

- `1` <-> `4`
- `5` <-> `8`
- `0` <-> `8`
- `3` <-> `8`
- `6` <-> `8`
- `2` <-> `7`

Clean controls use the original OCR text unchanged.

## Debugging

Inspect result counts:

```bash
python - <<'PY'
import json
from collections import Counter
rows=[json.loads(line) for line in open("cleaning/results.jsonl") if line.strip()]
print(len(rows))
print(Counter(r["score"]["verdict"] for r in rows))
print(Counter(r["split"] for r in rows))
PY
```

Inspect unparseable responses:

```bash
python - <<'PY'
import json
for line in open("cleaning/results.jsonl"):
    r=json.loads(line)
    if r["score"].get("unparseable"):
        print(r["table_id"], r["split"])
        print(r.get("raw_response", "")[:2000])
        break
PY
```

The first real run showed many raw responses were truncated after beginning a
valid JSON object. The runner now raises the default output budget and salvages
top-level verdicts from raw response prefixes when possible.
