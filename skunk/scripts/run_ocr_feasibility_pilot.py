"""Run the OCR table validation feasibility pilot.

The pilot reads the sampled extracted-table manifest, injects synthetic numeric
OCR typos into `parsed_ocr`, asks a vision LLM to compare the corrupted OCR text
against the table crop, and scores table-level detection plus rough correction
recovery.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from copy import deepcopy
from io import BytesIO

from PIL import Image

from tools.llm_wrapper import LLMWrapper, parse_json_response


DATA_DIR = "/orcd/home/002/gerarvit/orcd/scratch/officeqa/"
DEFAULT_SAMPLE_ROOT = f"{DATA_DIR}/ocr_cleaning/dev_samples"
# DEFAULT_OUTPUT_DIR = f"{DATA_DIR}/ocr_cleaning/feasibility_pilot"
DEFAULT_OUTPUT_DIR = "cleaning"
DEFAULT_MODEL = "gemini/gemini-3.5-flash"
# DEFAULT_MODEL = "gemini/gemini-3.1-pro-preview"

TAG_RE = re.compile(r"(<[^>]+>)")
NUMBER_RE = re.compile(r"[+-]?(?:\d[\d,]*|\d*\.\d+)(?:\.\d+)?")
RAW_VERDICT_RE = re.compile(r'"verdict"\s*:\s*"([^"]+)"')
SINGLE_LINE_RE = re.compile(r"\b([CEU])\s*\|\s*(0(?:\.\d+)?|1(?:\.0+)?)\b")
SINGLE_LABEL_RE = re.compile(r"^\s*([CEU])\s*$")
DIGIT_CONFUSIONS = {
    "1": ["4"],
    "4": ["1"],
    "5": ["8"],
    "8": ["5", "0", "3", "6"],
    "0": ["8"],
    "3": ["8"],
    "6": ["8"],
    "2": ["7"],
    "7": ["2"],
}
MODEL_CARDS = {
    "gemini/gemini-3.5-flash": {
        "input_usd_per_million": 1.50,
        "output_usd_per_million": 9.00,
    },
    "gemini/gemini-3.1-pro-preview": {
        "input_usd_per_million": 1.25,
        "output_usd_per_million": 10.00,
    },
    "gemini/gemini-3.1-flash-lite": {
        "input_usd_per_million": 0.25,
        "output_usd_per_million": 1.50,
    },
    "gemini/gemini-2.5-flash": {
        "input_usd_per_million": 0.30,
        "output_usd_per_million": 2.50,
    },
    "gemini/gemini-2.5-flash-lite": {
        "input_usd_per_million": 0.10,
        "output_usd_per_million": 0.40,
    },
    "gemini/gemini-2.5-pro": {
        "input_usd_per_million": 1.25,
        "output_usd_per_million": 10.00,
    },
}


def resolve_image_path(table: dict, sample_root: str) -> str:
    image_link = table.get("image_link")
    if image_link:
        candidate = os.path.join(sample_root, image_link)
        if os.path.exists(candidate):
            return candidate

    image_path = table.get("image_path") or ""
    if os.path.exists(image_path):
        return image_path

    raise FileNotFoundError(f"table image not found for {table.get('id')}: {image_path}")


def numeric_mutation_sites(parsed_ocr: str) -> list[dict]:
    sites = []
    offset = 0
    for part in TAG_RE.split(parsed_ocr):
        start = offset
        offset += len(part)
        if not part or part.startswith("<"):
            continue

        for match in NUMBER_RE.finditer(part):
            token = match.group(0)
            for token_idx, char in enumerate(token):
                if char not in DIGIT_CONFUSIONS:
                    continue
                absolute_idx = start + match.start() + token_idx
                sites.append(
                    {
                        "absolute_idx": absolute_idx,
                        "token_start": start + match.start(),
                        "token_end": start + match.end(),
                        "original_char": char,
                        "replacement_options": DIGIT_CONFUSIONS[char],
                        "original_token": token,
                    }
                )
    return sites


def inject_numeric_typos(parsed_ocr: str, rng: random.Random, max_typos: int) -> tuple[str, list[dict]]:
    sites = numeric_mutation_sites(parsed_ocr)
    if not sites:
        return parsed_ocr, []

    rng.shuffle(sites)
    selected = []
    used_indexes = set()
    for site in sites:
        if site["absolute_idx"] in used_indexes:
            continue
        selected.append(site)
        used_indexes.add(site["absolute_idx"])
        if len(selected) >= max_typos:
            break

    chars = list(parsed_ocr)
    mutations = []
    for site in sorted(selected, key=lambda item: item["absolute_idx"]):
        replacement = rng.choice(site["replacement_options"])
        chars[site["absolute_idx"]] = replacement
        corrupted_token = (
            parsed_ocr[site["token_start"] : site["absolute_idx"]]
            + replacement
            + parsed_ocr[site["absolute_idx"] + 1 : site["token_end"]]
        )
        mutations.append(
            {
                "absolute_idx": site["absolute_idx"],
                "original_char": site["original_char"],
                "replacement_char": replacement,
                "original_token": site["original_token"],
                "corrupted_token": corrupted_token,
            }
        )

    return "".join(chars), mutations


def build_json_prompt(table: dict, corrupted_ocr: str) -> str:
    table_label = table.get("label") or ""
    bulletin = table.get("bulletin") or ""
    page = table.get("page")
    prompt = f"""
You are validating OCR extracted from a U.S. Treasury Bulletin table image.

Compare the table image to the OCR HTML below. Focus only on numeric OCR
mistakes, especially wrong digits inside numbers. Ignore harmless formatting
differences, HTML structure differences, whitespace, and line wrapping.

Bulletin: {bulletin}
Page: {page}
Table label: {table_label}

Return only valid JSON with this exact shape:
{{
  "verdict": "clean|has_errors|uncertain",
  "confidence": 0.0,
  "suspect_cells": [
    {{
      "ocr_text": "the wrong OCR number or short cell text",
      "corrected_text": "the number or short cell text visible in the image",
      "row_hint": "nearby row label if visible, else empty string",
      "column_hint": "nearby column label if visible, else empty string",
      "reason": "brief visual evidence"
    }}
  ]
}}

Use "has_errors" if any numeric OCR value disagrees with the image.
Use "clean" only if the numeric OCR values match the image.
Use "uncertain" if the crop is unreadable or the relevant value cannot be
located.

OCR HTML:
{corrupted_ocr}
""".strip()
    return prompt


def build_single_line_prompt(table: dict, corrupted_ocr: str) -> str:
    table_label = table.get("label") or ""
    bulletin = table.get("bulletin") or ""
    page = table.get("page")
    prompt = f"""
You are checking whether OCR text from a U.S. Treasury Bulletin table matches
the attached table image.

Task:
Compare the table image against the OCR HTML. Focus only on numeric OCR
mistakes.

Return exactly one line in this format:

<label>|<confidence>

Labels:
C = clean: visible numeric OCR values match the image
E = error: at least one visible numeric OCR value is wrong
U = uncertain: image is unreadable, cropped badly, or the value cannot be checked

Confidence:
A decimal from 0.00 to 1.00.

Rules:
- Return only the single line. No JSON. No markdown. No explanation.
- Ignore whitespace, HTML tags, row/column formatting, and line wrapping.
- Ignore spelling differences in labels unless they change a numeric value.
- Mark E only when a specific numeric mismatch is visible.
- Mark U if you cannot verify the numeric values from the image.
- Prefer C over E when the apparent difference is only formatting, commas, or
  HTML structure.
- Valid outputs look like C|0.94, E|0.88, or U|0.61.

Examples:

Example 1
OCR says: <td>12,678</td>
Image shows: 17,678
Output:
E|0.98

Example 2
OCR says: <td>42.72</td>
Image shows: 12.72
Output:
E|0.99

Example 3
OCR says: <td>1,234</td>
Image shows: 1,234
Output:
C|0.97

Example 4
OCR says: <td>1234</td>
Image shows: 1,234
Output:
C|0.93

Example 5
OCR says: <td>-20</td>
Image is too blurry to distinguish -20 from -70
Output:
U|0.72

Example 6
OCR says: <td>88.5</td>
Image shows: 85.5
Output:
E|0.96

Now check the attached table image against this OCR HTML.

Bulletin: {bulletin}
Page: {page}
Table label: {table_label}

OCR HTML:
{corrupted_ocr}
""".strip()
    return prompt


def build_prompt(table: dict, corrupted_ocr: str, response_format: str) -> str:
    if response_format == "json":
        return build_json_prompt(table, corrupted_ocr)
    return build_single_line_prompt(table, corrupted_ocr)


def parse_single_line_response(text: str) -> dict:
    raw = (text or "").strip()
    match = SINGLE_LINE_RE.search(raw)
    if match:
        label = match.group(1)
        confidence_text = match.group(2)
        verdict = {"C": "clean", "E": "has_errors", "U": "uncertain"}[label]
        first_line = raw.splitlines()[0].strip() if raw else ""
        return {
            "verdict": verdict,
            "confidence": float(confidence_text),
            "label": label,
            "format_ok": first_line == f"{label}|{confidence_text}",
            "format_recovered": first_line != f"{label}|{confidence_text}",
        }

    first_line = raw.splitlines()[0].strip() if raw else ""
    label_match = SINGLE_LABEL_RE.match(first_line)
    if label_match:
        label = label_match.group(1)
        verdict = {"C": "clean", "E": "has_errors", "U": "uncertain"}[label]
        return {
            "verdict": verdict,
            "confidence": None,
            "label": label,
            "format_ok": False,
            "format_recovered": True,
        }

    return {}


def parse_verdict(parsed_response: dict) -> str:
    verdict = str(parsed_response.get("verdict") or "").strip().lower()
    if verdict in {"clean", "has_errors", "uncertain"}:
        return verdict
    return "unparseable"


def parse_failure_reason(raw_response: str, parsed_response: dict) -> str | None:
    if parse_verdict(parsed_response) != "unparseable":
        return None
    raw = (raw_response or "").strip()
    if not raw:
        return "empty_raw_response"
    if not isinstance(parsed_response, dict) or not parsed_response:
        if raw.startswith("```"):
            return "json_parse_failed_fenced_or_markdown"
        if raw.startswith("{") or raw.startswith("["):
            return "json_parse_failed_json_like"
        return "json_parse_failed_prose"
    if "verdict" not in parsed_response:
        return "missing_verdict"
    return f"bad_verdict:{parsed_response.get('verdict')!r}"


def raw_verdict(raw_response: str) -> str:
    parsed_single_line = parse_single_line_response(raw_response)
    single_line_verdict = parse_verdict(parsed_single_line)
    if single_line_verdict != "unparseable":
        return single_line_verdict

    match = RAW_VERDICT_RE.search(raw_response or "")
    if not match:
        return "unparseable"
    verdict = match.group(1).strip().lower()
    if verdict in {"clean", "has_errors", "uncertain"}:
        return verdict
    return "unparseable"


def score_record(record: dict) -> dict:
    parsed = record.get("parsed_response") or {}
    raw = record.get("raw_response") or ""
    parsed_verdict = parse_verdict(parsed)
    raw_prefix_verdict = raw_verdict(raw)
    verdict = parsed_verdict
    salvaged_from_raw = False
    if verdict == "unparseable" and raw_prefix_verdict != "unparseable":
        verdict = raw_prefix_verdict
        salvaged_from_raw = True
    has_injected_error = bool(record["mutations"])
    flagged_error = verdict == "has_errors"
    response_text = json.dumps(parsed, ensure_ascii=False)

    recovered = 0
    for mutation in record["mutations"]:
        original_token = mutation["original_token"]
        corrupted_token = mutation["corrupted_token"]
        if original_token in response_text and corrupted_token in response_text:
            recovered += 1
        elif original_token in response_text and mutation["replacement_char"] in response_text:
            recovered += 1

    return {
        "verdict": verdict,
        "has_injected_error": has_injected_error,
        "flagged_error": flagged_error,
        "true_positive": has_injected_error and flagged_error,
        "false_negative": has_injected_error and not flagged_error,
        "false_positive": (not has_injected_error) and flagged_error,
        "true_negative": (not has_injected_error) and verdict == "clean",
        "uncertain": verdict == "uncertain",
        "unparseable": verdict == "unparseable",
        "parsed_verdict": parsed_verdict,
        "raw_prefix_verdict": raw_prefix_verdict,
        "salvaged_from_raw": salvaged_from_raw,
        "parse_failure_reason": parse_failure_reason(raw, parsed),
        "mutation_count": len(record["mutations"]),
        "recovered_mutation_count": recovered,
    }


def write_jsonl(path: str, records: list[dict]) -> None:
    with open(path, "w") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")


def read_image_as_jpeg_bytes(path: str) -> bytes:
    with Image.open(path) as image:
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()


def usage_value(usage: dict, key: str) -> int:
    value = usage.get(key)
    return int(value) if isinstance(value, int | float) else 0


def estimate_cost_usd(model: str, usage: dict) -> float | None:
    card = MODEL_CARDS.get(model)
    if not card:
        return None
    input_tokens = usage_value(usage, "input_tokens")
    output_tokens = usage_value(usage, "output_tokens")
    return (
        input_tokens / 1_000_000 * card["input_usd_per_million"]
        + output_tokens / 1_000_000 * card["output_usd_per_million"]
    )


def sum_usage(usages: list[dict]) -> dict:
    keys = [
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "thinking_tokens",
        "cache_input_tokens",
    ]
    return {key: sum(usage_value(usage, key) for usage in usages) for key in keys}


parser = argparse.ArgumentParser(description="Run OCR table validation feasibility pilot.")
parser.add_argument("--sample-root", default=DEFAULT_SAMPLE_ROOT)
parser.add_argument("--tables-json", default=None)
parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
parser.add_argument("--model", default=DEFAULT_MODEL)
parser.add_argument("--n-corrupt", type=int, default=100)
parser.add_argument("--n-clean", type=int, default=50)
parser.add_argument("--max-typos-per-table", type=int, default=1)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--response-format", choices=["single_line", "json"], default="single_line")
parser.add_argument("--max-output-tokens", type=int, default=10000)
parser.add_argument("--max-workers", type=int, default=int(os.environ.get("LLM_MAX_WORKERS", "20")))
parser.add_argument(
    "--max-requests-per-minute",
    type=int,
    default=int(os.environ.get("LLM_MAX_REQUESTS_PER_MINUTE", "600")),
)
parser.add_argument("--cache-path", default=None)
parser.add_argument("--no-cache", action="store_true")
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

tables_json = args.tables_json or os.path.join(args.sample_root, "extracted_tables", "tables.json")
with open(tables_json) as f:
    manifest = json.load(f)

rng = random.Random(args.seed)
eligible_tables = []
skipped = []
for table in manifest.get("tables", []):
    parsed_ocr = table.get("parsed_ocr") or ""
    sites = numeric_mutation_sites(parsed_ocr)
    if not sites:
        skipped.append({"id": table.get("id"), "reason": "no numeric typo sites"})
        continue
    try:
        image_path = resolve_image_path(table, args.sample_root)
    except FileNotFoundError as e:
        skipped.append({"id": table.get("id"), "reason": str(e)})
        continue
    row = deepcopy(table)
    row["_image_path"] = image_path
    row["_numeric_site_count"] = len(sites)
    eligible_tables.append(row)

if len(eligible_tables) < args.n_corrupt + args.n_clean:
    raise SystemExit(
        f"Need {args.n_corrupt + args.n_clean} eligible tables, found {len(eligible_tables)}"
    )

rng.shuffle(eligible_tables)
corrupt_tables = eligible_tables[: args.n_corrupt]
clean_tables = eligible_tables[args.n_corrupt : args.n_corrupt + args.n_clean]

records = []
for split, table_rows in [("corrupt", corrupt_tables), ("clean", clean_tables)]:
    for table in table_rows:
        parsed_ocr = table.get("parsed_ocr") or ""
        if split == "corrupt":
            corrupted_ocr, mutations = inject_numeric_typos(
                parsed_ocr,
                rng,
                max(1, args.max_typos_per_table),
            )
        else:
            corrupted_ocr = parsed_ocr
            mutations = []

        prompt = build_prompt(table, corrupted_ocr, args.response_format)
        records.append(
            {
                "table_id": table.get("id"),
                "bulletin": table.get("bulletin"),
                "page": table.get("page"),
                "table_index": table.get("table_index"),
                "label": table.get("label"),
                "split": split,
                "image_path": table["_image_path"],
                "numeric_site_count": table["_numeric_site_count"],
                "mutations": mutations,
                "response_format": args.response_format,
                "prompt": prompt
            }
        )

os.makedirs(args.output_dir, exist_ok=True)
planned_path = os.path.join(args.output_dir, "planned_requests.jsonl")
write_jsonl(planned_path, records)

if args.dry_run:
    summary = {
        "dry_run": True,
        "tables_json": tables_json,
        "output_dir": args.output_dir,
        "model": args.model,
        "response_format": args.response_format,
        "seed": args.seed,
        "planned_request_count": len(records),
        "corrupt_count": args.n_corrupt,
        "clean_count": args.n_clean,
        "skipped_count": len(skipped),
        "planned_requests_path": planned_path,
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"Dry run wrote {planned_path}")
    print(f"Wrote {summary_path}")
    sys.exit(0)

requests = []
for record in records:
    requests.append((record["prompt"], read_image_as_jpeg_bytes(record["image_path"])))

cache_path = args.cache_path

wrapper_kwargs = {
    "max_workers": args.max_workers,
    "max_requests_per_minute": args.max_requests_per_minute,
    "cache_enabled": not args.no_cache,
}
if cache_path is not None:
    wrapper_kwargs["cache_path"] = cache_path
wrapper = LLMWrapper(**wrapper_kwargs)

started_at = time.time()
response_payloads = wrapper.batch_call_llm_vision_with_usage(
    requests,
    max_tokens=args.max_output_tokens,
    model=args.model,
    desc="OCR feasibility pilot",
    show_progress=True,
    use_cache=not args.no_cache,
)
elapsed_seconds = time.time() - started_at
wrapper.flush_cache(wait=True)

scored_records = []
for record, response_payload in zip(records, response_payloads, strict=True):
    response = response_payload.get("text", "")
    if args.response_format == "json":
        parsed_response = parse_json_response(response)
    else:
        parsed_response = parse_single_line_response(response)
    usage = response_payload.get("usage") or {}
    record["raw_response"] = response
    record["parsed_response"] = parsed_response
    record["usage"] = usage
    record["cache_hit"] = bool(response_payload.get("cache_hit"))
    record["estimated_cost_usd"] = estimate_cost_usd(args.model, usage)
    record["score"] = score_record(record)
    del record["prompt"]
    scored_records.append(record)

results_path = os.path.join(args.output_dir, "results.jsonl")
write_jsonl(results_path, scored_records)

scores = [record["score"] for record in scored_records]
corrupt_scores = [score for score in scores if score["has_injected_error"]]
clean_scores = [score for score in scores if not score["has_injected_error"]]
true_positives = sum(score["true_positive"] for score in scores)
false_negatives = sum(score["false_negative"] for score in scores)
false_positives = sum(score["false_positive"] for score in scores)
true_negatives = sum(score["true_negative"] for score in scores)
uncertain = sum(score["uncertain"] for score in scores)
unparseable = sum(score["unparseable"] for score in scores)
salvaged_from_raw = sum(score["salvaged_from_raw"] for score in scores)
mutation_count = sum(score["mutation_count"] for score in scores)
recovered_mutation_count = sum(score["recovered_mutation_count"] for score in scores)
usages = [record.get("usage") or {} for record in scored_records]
total_usage = sum_usage(usages)
non_cached_usages = [
    record.get("usage") or {} for record in scored_records if not record.get("cache_hit")
]
non_cached_usage = sum_usage(non_cached_usages)
estimated_cost_values = [
    cost
    for record in scored_records
    for cost in [record.get("estimated_cost_usd")]
    if cost is not None
]
total_estimated_cost_usd = sum(estimated_cost_values) if estimated_cost_values else None
non_cached_estimated_cost_values = [
    cost
    for record in scored_records
    if not record.get("cache_hit")
    for cost in [record.get("estimated_cost_usd")]
    if cost is not None
]
non_cached_estimated_cost_usd = (
    sum(non_cached_estimated_cost_values) if non_cached_estimated_cost_values else None
)
cache_hits = sum(bool(record.get("cache_hit")) for record in scored_records)

summary = {
    "dry_run": False,
    "tables_json": tables_json,
    "output_dir": args.output_dir,
    "model": args.model,
    "response_format": args.response_format,
    "model_card": MODEL_CARDS.get(args.model),
    "seed": args.seed,
    "request_count": len(scored_records),
    "corrupt_count": len(corrupt_scores),
    "clean_count": len(clean_scores),
    "skipped_count": len(skipped),
    "elapsed_seconds": elapsed_seconds,
    "seconds_per_request": elapsed_seconds / len(scored_records) if scored_records else None,
    "cache_hits": cache_hits,
    "cache_misses": len(scored_records) - cache_hits,
    "total_usage": total_usage,
    "non_cached_usage": non_cached_usage,
    "workload_estimated_cost_usd": total_estimated_cost_usd,
    "workload_estimated_cost_per_request_usd": total_estimated_cost_usd / len(scored_records)
    if total_estimated_cost_usd is not None and scored_records
    else None,
    "non_cached_estimated_cost_usd": non_cached_estimated_cost_usd,
    "non_cached_estimated_cost_per_request_usd": non_cached_estimated_cost_usd
    / (len(scored_records) - cache_hits)
    if non_cached_estimated_cost_usd is not None and len(scored_records) - cache_hits
    else None,
    "true_positives": true_positives,
    "false_negatives": false_negatives,
    "false_positives": false_positives,
    "true_negatives": true_negatives,
    "uncertain": uncertain,
    "unparseable": unparseable,
    "salvaged_from_raw": salvaged_from_raw,
    "table_recall": true_positives / (true_positives + false_negatives)
    if true_positives + false_negatives
    else None,
    "table_precision": true_positives / (true_positives + false_positives)
    if true_positives + false_positives
    else None,
    "clean_false_positive_rate": false_positives / len(clean_scores) if clean_scores else None,
    "mutation_count": mutation_count,
    "recovered_mutation_count": recovered_mutation_count,
    "mutation_recovery_rate": recovered_mutation_count / mutation_count if mutation_count else None,
    "planned_requests_path": planned_path,
    "results_path": results_path,
    "cache_path": wrapper.cache_path if not args.no_cache else None,
}

summary_path = os.path.join(args.output_dir, "summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
    f.write("\n")

print(f"Wrote planned requests to {planned_path}")
print(f"Wrote results to {results_path}")
print(f"Wrote summary to {summary_path}")
print(json.dumps(summary, indent=2))
