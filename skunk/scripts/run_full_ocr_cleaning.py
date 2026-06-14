"""Run full-corpus LLM OCR cleaning over extracted Treasury Bulletin tables."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from io import BytesIO

from PIL import Image

from tools.llm_wrapper import LLMWrapper


DATA_DIR = "/orcd/home/002/gerarvit/orcd/scratch/officeqa"
FULL_TABLES_JSON = f"{DATA_DIR}/extracted_tables/tables.json"
DEV_TABLES_JSON = f"{DATA_DIR}/ocr_cleaning/dev_samples/extracted_tables/tables.json"
FULL_OUTPUT_DIR = f"{DATA_DIR}/ocr_cleaning/full_run"
DEV_OUTPUT_DIR = f"{DATA_DIR}/ocr_cleaning/dev_full_run"
DEFAULT_MODEL = "gemini/gemini-3.1-pro-preview"
VERDICT_PROMPT_VERSION = "verdict_v1"
CORRECTION_PROMPT_VERSION = "correction_v1"

VERDICT_RE = re.compile(r"\b([CEU])\s*\|\s*(0(?:\.\d+)?|1(?:\.0+)?)\b")
LABEL_ONLY_RE = re.compile(r"^\s*([CEU])\s*$")
CORRECTION_RE = re.compile(r"^X\|(.*)$")

MODEL_CARDS = {
    "gemini/gemini-3.1-pro-preview": {
        "input_usd_per_million": 1.25,
        "output_usd_per_million": 10.00,
    },
    "gemini/gemini-3.5-flash": {
        "input_usd_per_million": 1.50,
        "output_usd_per_million": 9.00,
    },
    "gemini/gemini-2.5-pro": {
        "input_usd_per_million": 1.25,
        "output_usd_per_million": 10.00,
    },
    "gemini/gemini-2.5-flash": {
        "input_usd_per_million": 0.30,
        "output_usd_per_million": 2.50,
    },
    "gemini/gemini-2.5-flash-lite": {
        "input_usd_per_million": 0.10,
        "output_usd_per_million": 0.40,
    },
}


def now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_jsonl_latest(path: str, key: str = "table_id") -> dict:
    latest = {}
    if not os.path.exists(path):
        return latest
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_key = row_table_id(row) if key == "table_id" else row.get(key)
            if row_key:
                latest[row_key] = row
    return latest


def row_table_id(row: dict) -> str | None:
    value = row.get("table_id") or row.get("id")
    return str(value) if value is not None else None


def stable_table_key(row: dict) -> tuple[str, str, str] | None:
    bulletin = row.get("bulletin")
    page = row.get("page")
    table_index = row.get("table_index")
    if bulletin is None or page is None or table_index is None:
        return None
    return str(bulletin), str(page), str(table_index)


def completed_lookup(rows: list[dict]) -> set[tuple]:
    lookup = set()
    for row in rows:
        table_id = row_table_id(row)
        if table_id:
            lookup.add(("table_id", table_id))
        key = stable_table_key(row)
        if key:
            lookup.add(("stable", *key))
    return lookup


def table_completion_keys(table: dict) -> list[tuple]:
    keys = []
    table_id = row_table_id(table)
    if table_id:
        keys.append(("table_id", table_id))
    key = stable_table_key(table)
    if key:
        keys.append(("stable", *key))
    return keys


def read_jsonl_file_stats(path: str) -> dict:
    stats = {
        "path": path,
        "exists": os.path.exists(path),
        "size_bytes": os.path.getsize(path) if os.path.exists(path) else 0,
        "lines": 0,
        "valid_json_rows": 0,
        "invalid_json_rows": 0,
        "rows_with_table_id": 0,
        "stage_counts": {},
    }
    if not os.path.exists(path):
        return stats
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            stats["lines"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats["invalid_json_rows"] += 1
                continue
            stats["valid_json_rows"] += 1
            if row_table_id(row):
                stats["rows_with_table_id"] += 1
            stage = row.get("stage")
            if stage:
                stats["stage_counts"][stage] = stats["stage_counts"].get(stage, 0) + 1
    return stats


def append_jsonl(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def recover_outputs_from_raw(output_dir: str) -> dict:
    raw_path = os.path.join(output_dir, "raw_responses.jsonl")
    verdicts_path = os.path.join(output_dir, "verdicts.jsonl")
    correction_tables_path = os.path.join(output_dir, "correction_tables.jsonl")
    corrections_path = os.path.join(output_dir, "corrections.jsonl")
    stats = {
        "raw_rows_scanned": 0,
        "verdict_rows_recovered": 0,
        "correction_table_rows_recovered": 0,
        "correction_rows_recovered": 0,
    }
    if not os.path.exists(raw_path):
        return stats

    raw_by_stage_and_id = {}
    with open(raw_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            stage = row.get("stage")
            table_id = row_table_id(row)
            if stage in ("verdict", "correction") and table_id:
                raw_by_stage_and_id[(stage, table_id)] = row
                stats["raw_rows_scanned"] += 1

    verdict_rows_by_id = read_jsonl_latest(verdicts_path)
    correction_rows_by_id = read_jsonl_latest(correction_tables_path)
    verdict_completed = completed_lookup(list(verdict_rows_by_id.values()))
    correction_completed = completed_lookup(list(correction_rows_by_id.values()))
    correction_keys = set()
    if os.path.exists(corrections_path):
        with open(corrections_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                table_id = row_table_id(row)
                correction_index = row.get("correction_index")
                if table_id and correction_index is not None:
                    correction_keys.add((table_id, correction_index))

    recovered_verdict_rows = []
    recovered_correction_table_rows = []
    recovered_correction_rows = []
    identity_keys = [
        "table_id",
        "bulletin",
        "year",
        "month",
        "page",
        "table_index",
        "label",
        "image_path",
    ]

    for (stage, table_id), raw_row in raw_by_stage_and_id.items():
        task = {"identity": {key: raw_row.get(key) for key in identity_keys}}
        raw_keys = table_completion_keys(raw_row)
        if stage == "verdict" and not any(key in verdict_completed for key in raw_keys):
            stage_row, _ = process_verdict_payload(task, raw_row)
            recovered_verdict_rows.append(stage_row)
            verdict_rows_by_id[table_id] = stage_row
            verdict_completed.update(table_completion_keys(stage_row))
        elif stage == "correction" and not any(key in correction_completed for key in raw_keys):
            stage_row, correction_rows = process_correction_payload(task, raw_row)
            recovered_correction_table_rows.append(stage_row)
            correction_rows_by_id[table_id] = stage_row
            correction_completed.update(table_completion_keys(stage_row))
            for correction_row in correction_rows:
                correction_key = (
                    correction_row.get("table_id"),
                    correction_row.get("correction_index"),
                )
                if correction_key not in correction_keys:
                    recovered_correction_rows.append(correction_row)
                    correction_keys.add(correction_key)

    append_jsonl(verdicts_path, recovered_verdict_rows)
    append_jsonl(correction_tables_path, recovered_correction_table_rows)
    append_jsonl(corrections_path, recovered_correction_rows)
    stats["verdict_rows_recovered"] = len(recovered_verdict_rows)
    stats["correction_table_rows_recovered"] = len(recovered_correction_table_rows)
    stats["correction_rows_recovered"] = len(recovered_correction_rows)
    return stats


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp_path, path)


def usage_value(usage: dict, key: str) -> int:
    value = usage.get(key)
    return int(value) if isinstance(value, (int, float)) else 0


def sum_usage(rows: list[dict]) -> dict:
    keys = [
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "thinking_tokens",
        "cache_input_tokens",
    ]
    totals = {key: 0 for key in keys}
    for row in rows:
        usage = row.get("usage") or {}
        for key in keys:
            totals[key] += usage_value(usage, key)
    return totals


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


def resolve_image_path(table: dict, tables_json: str) -> str:
    image_link = table.get("image_link")
    if image_link:
        sample_root = os.path.dirname(os.path.dirname(tables_json))
        candidate = os.path.join(sample_root, image_link)
        if os.path.exists(candidate):
            return candidate

    image_path = table.get("image_path") or ""
    if os.path.exists(image_path):
        return image_path

    output_dir = table.get("output_dir")
    if output_dir:
        candidate = os.path.join(output_dir, "images", f"{table.get('id')}.png")
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(f"image not found for table {table.get('id')}: {image_path}")


def read_image_as_jpeg_bytes(path: str) -> bytes:
    with Image.open(path) as image:
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()


def table_identity(table: dict, image_path: str) -> dict:
    return {
        "table_id": table.get("id"),
        "bulletin": table.get("bulletin"),
        "year": table.get("year"),
        "month": table.get("month"),
        "page": table.get("page"),
        "table_index": table.get("table_index"),
        "label": table.get("label"),
        "image_path": image_path,
    }


def build_verdict_prompt(table: dict) -> str:
    return f"""
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

Bulletin: {table.get("bulletin")}
Page: {table.get("page")}
Table label: {table.get("label") or ""}

OCR HTML:
{table.get("parsed_ocr") or ""}
""".strip()


def build_correction_prompt(table: dict) -> str:
    return f"""
You are double-checking a U.S. Treasury Bulletin table that a first-pass OCR
review marked as possibly corrupt.

Task:
Compare the table image against the OCR HTML. Focus only on numeric OCR
mistakes. First decide whether there is truly a visible numeric mismatch. If
there is no clear numeric mismatch, return clean. If the image is unreadable or
too cropped to verify, return uncertain.

Return one of these compact formats:

C|<confidence>
U|<confidence>
E|<confidence>
X|<ocr_text>|<corrected_text>|<row_hint>|<column_hint>
X|<ocr_text>|<corrected_text>|<row_hint>|<column_hint>

Rules:
- Return only the compact lines. No JSON. No markdown. No explanation.
- The first line must be C, U, or E with confidence.
- If the first line is E, include one X line per numeric correction.
- Multiple X lines are allowed.
- Do not invent corrections. Include only numeric mismatches visible in the image.
- Ignore whitespace, HTML tags, row/column formatting, and line wrapping.
- Prefer C over E when the apparent difference is only formatting, commas, or
  HTML structure.

Examples:

Example 1
OCR says: <td>42.72</td>
Image shows: 12.72
Output:
E|0.99
X|42.72|12.72||

Example 2
OCR says: <td>1,234</td>
Image shows: 1,234
Output:
C|0.97

Example 3
OCR says: <td>-20</td>
Image is too blurry to distinguish -20 from -70
Output:
U|0.72

Now double-check the attached table image against this OCR HTML.

Bulletin: {table.get("bulletin")}
Page: {table.get("page")}
Table label: {table.get("label") or ""}

OCR HTML:
{table.get("parsed_ocr") or ""}
""".strip()


def parse_label_response(raw_response: str) -> tuple[str, float | None, str, bool]:
    raw = (raw_response or "").strip()
    match = VERDICT_RE.search(raw)
    if match:
        label = match.group(1)
        confidence = float(match.group(2))
        first_line = raw.splitlines()[0].strip() if raw else ""
        return label, confidence, "ok" if first_line == f"{label}|{match.group(2)}" else "recovered", first_line != f"{label}|{match.group(2)}"
    first_line = raw.splitlines()[0].strip() if raw else ""
    label_match = LABEL_ONLY_RE.match(first_line)
    if label_match:
        return label_match.group(1), None, "recovered", True
    return "P", None, "unparseable", False


def label_to_verdict(label: str) -> str:
    return {
        "C": "clean",
        "E": "error",
        "U": "uncertain",
        "P": "unparseable",
    }.get(label, "unparseable")


def parse_corrections(raw_response: str) -> list[dict]:
    corrections = []
    for line in (raw_response or "").splitlines():
        line = line.strip()
        match = CORRECTION_RE.match(line)
        if not match:
            continue
        parts = match.group(1).split("|", 4)
        while len(parts) < 5:
            parts.append("")
        corrections.append(
            {
                "ocr_text": parts[0].strip(),
                "corrected_text": parts[1].strip(),
                "row_hint": parts[2].strip(),
                "column_hint": parts[3].strip(),
                "extra": parts[4].strip(),
            }
        )
    return corrections


def response_payload_to_common(row: dict, stage: str, prompt_version: str, model: str, payload: dict) -> dict:
    usage = payload.get("usage") or {}
    return {
        **row,
        "stage": stage,
        "model": model,
        "prompt_version": prompt_version,
        "raw_response": payload.get("text", ""),
        "usage": usage,
        "estimated_cost_usd": estimate_cost_usd(model, usage),
        "cache_hit": bool(payload.get("cache_hit")),
        "created_at": now_ts(),
    }


def call_batches(
    wrapper: LLMWrapper,
    tasks: list[dict],
    stage: str,
    prompt_version: str,
    model: str,
    max_output_tokens: int,
    batch_size: int,
    checkpoint_interval: int,
    use_cache: bool,
    output_dir: str,
    process_payload,
) -> int:
    processed = 0
    pending_raw_rows = []
    pending_stage_rows = []
    pending_correction_rows = []
    next_checkpoint = checkpoint_interval

    for start in range(0, len(tasks), batch_size):
        batch = tasks[start : start + batch_size]
        requests = []
        for task in batch:
            requests.append((task["prompt"], read_image_as_jpeg_bytes(task["image_path"])))

        payloads = wrapper.batch_call_llm_vision_with_usage(
            requests,
            max_tokens=max_output_tokens,
            model=model,
            desc=f"{stage} {start + 1}-{start + len(batch)}",
            show_progress=True,
            use_cache=use_cache,
        )

        for task, payload in zip(batch, payloads, strict=True):
            raw_row = response_payload_to_common(
                task["identity"], stage, prompt_version, model, payload
            )
            stage_row, correction_rows = process_payload(task, raw_row)
            pending_raw_rows.append(raw_row)
            pending_stage_rows.append(stage_row)
            pending_correction_rows.extend(correction_rows)
            processed += 1

        append_outputs(output_dir, stage, pending_raw_rows, pending_stage_rows, pending_correction_rows)
        pending_raw_rows.clear()
        pending_stage_rows.clear()
        pending_correction_rows.clear()

        if processed >= next_checkpoint or start + len(batch) >= len(tasks):
            wrapper.flush_cache(wait=True)
            write_run_state(output_dir, stage, processed, len(tasks))
            next_checkpoint += checkpoint_interval

    return processed


def append_outputs(
    output_dir: str,
    stage: str,
    raw_rows: list[dict],
    stage_rows: list[dict],
    correction_rows: list[dict],
) -> None:
    append_jsonl(os.path.join(output_dir, "raw_responses.jsonl"), raw_rows)
    if stage == "verdict":
        append_jsonl(os.path.join(output_dir, "verdicts.jsonl"), stage_rows)
    else:
        append_jsonl(os.path.join(output_dir, "correction_tables.jsonl"), stage_rows)
        append_jsonl(os.path.join(output_dir, "corrections.jsonl"), correction_rows)


def write_run_state(output_dir: str, stage: str, processed: int, total: int) -> None:
    write_json(
        os.path.join(output_dir, "run_state.json"),
        {
            "updated_at": now_ts(),
            "stage": stage,
            "processed_in_current_stage": processed,
            "total_in_current_stage": total,
        },
    )


def process_verdict_payload(task: dict, raw_row: dict) -> tuple[dict, list[dict]]:
    label, confidence, parse_status, format_recovered = parse_label_response(
        raw_row["raw_response"]
    )
    stage_row = {
        **task["identity"],
        "verdict": label,
        "verdict_label": label_to_verdict(label),
        "confidence": confidence,
        "parse_status": parse_status,
        "format_recovered": format_recovered,
        "model": raw_row["model"],
        "prompt_version": raw_row["prompt_version"],
        "usage": raw_row["usage"],
        "estimated_cost_usd": raw_row["estimated_cost_usd"],
        "cache_hit": raw_row["cache_hit"],
        "created_at": raw_row["created_at"],
    }
    return stage_row, []


def process_correction_payload(task: dict, raw_row: dict) -> tuple[dict, list[dict]]:
    label, confidence, parse_status, format_recovered = parse_label_response(
        raw_row["raw_response"]
    )
    corrections = parse_corrections(raw_row["raw_response"]) if label == "E" else []
    status = {
        "C": "rejected_clean",
        "E": "confirmed_error",
        "U": "uncertain",
        "P": "unparseable",
    }.get(label, "unparseable")
    stage_row = {
        **task["identity"],
        "status": status,
        "verdict": label,
        "confidence": confidence,
        "correction_count": len(corrections),
        "parse_status": parse_status,
        "format_recovered": format_recovered,
        "model": raw_row["model"],
        "prompt_version": raw_row["prompt_version"],
        "usage": raw_row["usage"],
        "estimated_cost_usd": raw_row["estimated_cost_usd"],
        "cache_hit": raw_row["cache_hit"],
        "created_at": raw_row["created_at"],
    }
    correction_rows = []
    for idx, correction in enumerate(corrections):
        correction_rows.append(
            {
                **task["identity"],
                "correction_index": idx,
                "ocr_text": correction["ocr_text"],
                "corrected_text": correction["corrected_text"],
                "row_hint": correction["row_hint"],
                "column_hint": correction["column_hint"],
                "extra": correction["extra"],
                "table_confidence": confidence,
                "model": raw_row["model"],
                "prompt_version": raw_row["prompt_version"],
                "created_at": raw_row["created_at"],
            }
        )
    return stage_row, correction_rows


def build_tasks(tables: list[dict], tables_json: str, completed: dict, stage: str) -> list[dict]:
    tasks = []
    completed_keys = completed_lookup(list(completed.values()))
    for table in tables:
        table_id = table.get("id")
        if not table_id or any(key in completed_keys for key in table_completion_keys(table)):
            continue
        parsed_ocr = table.get("parsed_ocr") or ""
        if not parsed_ocr.strip():
            continue
        try:
            image_path = resolve_image_path(table, tables_json)
        except FileNotFoundError:
            continue
        identity = table_identity(table, image_path)
        if stage == "verdict":
            prompt = build_verdict_prompt(table)
        else:
            prompt = build_correction_prompt(table)
        tasks.append(
            {
                "identity": identity,
                "image_path": image_path,
                "prompt": prompt,
            }
        )
    return tasks


def select_error_tables(tables: list[dict], verdict_rows_by_id: dict) -> list[dict]:
    error_ids = set()
    error_stable_keys = set()
    for row in verdict_rows_by_id.values():
        if row.get("verdict") != "E":
            continue
        table_id = row_table_id(row)
        if table_id:
            error_ids.add(table_id)
        key = stable_table_key(row)
        if key:
            error_stable_keys.add(key)
    selected = []
    for table in tables:
        table_id = row_table_id(table)
        key = stable_table_key(table)
        if (table_id and table_id in error_ids) or (key and key in error_stable_keys):
            selected.append(table)
    return selected


def load_tables(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data.get("tables", [])


def summarize(output_dir: str, total_tables: int, model: str) -> dict:
    verdict_rows = list(read_jsonl_latest(os.path.join(output_dir, "verdicts.jsonl")).values())
    correction_table_rows = list(
        read_jsonl_latest(os.path.join(output_dir, "correction_tables.jsonl")).values()
    )
    corrections = []
    corrections_path = os.path.join(output_dir, "corrections.jsonl")
    if os.path.exists(corrections_path):
        with open(corrections_path) as f:
            corrections = [json.loads(line) for line in f if line.strip()]

    raw_rows = []
    raw_path = os.path.join(output_dir, "raw_responses.jsonl")
    if os.path.exists(raw_path):
        with open(raw_path) as f:
            raw_rows = [json.loads(line) for line in f if line.strip()]

    costs = [
        row.get("estimated_cost_usd")
        for row in raw_rows
        if row.get("estimated_cost_usd") is not None
    ]
    non_cached_costs = [
        row.get("estimated_cost_usd")
        for row in raw_rows
        if not row.get("cache_hit") and row.get("estimated_cost_usd") is not None
    ]
    summary = {
        "updated_at": now_ts(),
        "model": model,
        "model_card": MODEL_CARDS.get(model),
        "total_tables": total_tables,
        "pass1_completed": len(verdict_rows),
        "pass1_clean": sum(row.get("verdict") == "C" for row in verdict_rows),
        "pass1_error": sum(row.get("verdict") == "E" for row in verdict_rows),
        "pass1_uncertain": sum(row.get("verdict") == "U" for row in verdict_rows),
        "pass1_unparseable": sum(row.get("verdict") == "P" for row in verdict_rows),
        "pass2_completed": len(correction_table_rows),
        "pass2_confirmed_error": sum(
            row.get("status") == "confirmed_error" for row in correction_table_rows
        ),
        "pass2_rejected_clean": sum(
            row.get("status") == "rejected_clean" for row in correction_table_rows
        ),
        "pass2_uncertain": sum(row.get("status") == "uncertain" for row in correction_table_rows),
        "pass2_unparseable": sum(row.get("status") == "unparseable" for row in correction_table_rows),
        "total_corrections": len(corrections),
        "raw_response_rows": len(raw_rows),
        "total_usage": sum_usage(raw_rows),
        "non_cached_usage": sum_usage([row for row in raw_rows if not row.get("cache_hit")]),
        "workload_estimated_cost_usd": sum(costs) if costs else None,
        "non_cached_estimated_cost_usd": sum(non_cached_costs) if non_cached_costs else None,
        "cache_hits": sum(bool(row.get("cache_hit")) for row in raw_rows),
        "cache_misses": sum(not bool(row.get("cache_hit")) for row in raw_rows),
    }
    write_json(os.path.join(output_dir, "summary.json"), summary)
    return summary


def resume_diagnostics(
    tables: list[dict],
    output_dir: str,
    verdict_rows_by_id: dict,
    correction_rows_by_id: dict,
    verdict_tasks: list[dict],
    correction_tasks: list[dict],
) -> dict:
    table_ids = {row_table_id(table) for table in tables if row_table_id(table)}
    table_stable_keys = {
        stable_table_key(table) for table in tables if stable_table_key(table)
    }
    verdict_ids = {row_table_id(row) for row in verdict_rows_by_id.values() if row_table_id(row)}
    verdict_stable_keys = {
        stable_table_key(row)
        for row in verdict_rows_by_id.values()
        if stable_table_key(row)
    }
    correction_ids = {
        row_table_id(row)
        for row in correction_rows_by_id.values()
        if row_table_id(row)
    }
    correction_stable_keys = {
        stable_table_key(row)
        for row in correction_rows_by_id.values()
        if stable_table_key(row)
    }
    return {
        "files": {
            "raw_responses": read_jsonl_file_stats(
                os.path.join(output_dir, "raw_responses.jsonl")
            ),
            "verdicts": read_jsonl_file_stats(os.path.join(output_dir, "verdicts.jsonl")),
            "correction_tables": read_jsonl_file_stats(
                os.path.join(output_dir, "correction_tables.jsonl")
            ),
            "corrections": read_jsonl_file_stats(
                os.path.join(output_dir, "corrections.jsonl")
            ),
        },
        "matches": {
            "loaded_table_ids": len(table_ids),
            "loaded_stable_keys": len(table_stable_keys),
            "verdict_id_overlap": len(table_ids & verdict_ids),
            "verdict_stable_overlap": len(table_stable_keys & verdict_stable_keys),
            "correction_id_overlap": len(table_ids & correction_ids),
            "correction_stable_overlap": len(table_stable_keys & correction_stable_keys),
        },
        "samples": {
            "loaded_table_ids": sorted(table_ids)[:5],
            "verdict_row_ids": sorted(verdict_ids)[:5],
            "correction_row_ids": sorted(correction_ids)[:5],
            "pending_verdict_ids": [
                task["identity"]["table_id"] for task in verdict_tasks[:10]
            ],
            "pending_correction_ids": [
                task["identity"]["table_id"] for task in correction_tasks[:10]
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full-corpus OCR cleaning.")
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--tables-json", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-workers", type=int, default=int(os.environ.get("LLM_MAX_WORKERS", "64")))
    parser.add_argument("--max-requests-per-minute", type=int, default=int(os.environ.get("LLM_MAX_REQUESTS_PER_MINUTE", "600")))
    parser.add_argument("--max-output-tokens", type=int, default=10000)
    parser.add_argument("--checkpoint-interval", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--cache-path", default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--stage", choices=["both", "verdict", "correction"], default="both")
    parser.add_argument("--allow-verdict-after-correction", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tables_json = args.tables_json or (DEV_TABLES_JSON if args.dev else FULL_TABLES_JSON)
    output_dir = args.output_dir or (DEV_OUTPUT_DIR if args.dev else FULL_OUTPUT_DIR)
    batch_size = args.batch_size or max(1, min(args.checkpoint_interval, args.max_workers * 10))
    os.makedirs(output_dir, exist_ok=True)
    recovery_stats = recover_outputs_from_raw(output_dir)

    tables = load_tables(tables_json)
    if args.limit:
        tables = tables[: args.limit]

    verdicts_path = os.path.join(output_dir, "verdicts.jsonl")
    correction_tables_path = os.path.join(output_dir, "correction_tables.jsonl")
    verdict_rows_by_id = read_jsonl_latest(verdicts_path)
    correction_rows_by_id = read_jsonl_latest(correction_tables_path)

    verdict_tasks = []
    if args.stage in ("both", "verdict"):
        verdict_tasks = build_tasks(tables, tables_json, verdict_rows_by_id, "verdict")

    correction_source = verdict_rows_by_id
    correction_tasks = []
    if args.stage in ("both", "correction"):
        correction_tables = select_error_tables(tables, correction_source)
        correction_tasks = build_tasks(
            correction_tables, tables_json, correction_rows_by_id, "correction"
        )

    diagnostics = resume_diagnostics(
        tables,
        output_dir,
        verdict_rows_by_id,
        correction_rows_by_id,
        verdict_tasks,
        correction_tasks,
    )
    plan = {
        "tables_json": tables_json,
        "output_dir": output_dir,
        "model": args.model,
        "max_output_tokens": args.max_output_tokens,
        "checkpoint_interval": args.checkpoint_interval,
        "batch_size": batch_size,
        "stage": args.stage,
        "dev": args.dev,
        "total_loaded_tables": len(tables),
        "pending_verdict_tasks": len(verdict_tasks),
        "pending_correction_tasks": len(correction_tasks),
        "existing_verdict_rows": len(verdict_rows_by_id),
        "existing_correction_rows": len(correction_rows_by_id),
        "recovery_from_raw": recovery_stats,
        "resume_diagnostics": diagnostics,
    }
    write_json(os.path.join(output_dir, "run_state.json"), {"updated_at": now_ts(), "plan": plan})
    print(json.dumps(plan, indent=2))

    correction_raw_rows = diagnostics["files"]["raw_responses"]["stage_counts"].get(
        "correction", 0
    )
    correction_started = len(correction_rows_by_id) > 0 or correction_raw_rows > 0
    if (
        args.stage == "both"
        and not args.dry_run
        and correction_started
        and verdict_tasks
        and not args.allow_verdict_after_correction
    ):
        raise SystemExit(
            "Refusing to run pass 1 again because correction-stage output already exists. "
            "Check resume_diagnostics in the printed plan. Use --stage correction to resume "
            "only corrections, or --allow-verdict-after-correction if you really intend to "
            "run pending verdict tasks."
        )

    if args.dry_run:
        summarize(output_dir, len(tables), args.model)
        return

    wrapper_kwargs = {
        "max_workers": args.max_workers,
        "max_requests_per_minute": args.max_requests_per_minute,
        "cache_enabled": not args.no_cache,
    }
    if args.cache_path:
        wrapper_kwargs["cache_path"] = args.cache_path
    wrapper = LLMWrapper(**wrapper_kwargs)

    started_at = time.time()
    try:
        if verdict_tasks:
            call_batches(
                wrapper,
                verdict_tasks,
                "verdict",
                VERDICT_PROMPT_VERSION,
                args.model,
                args.max_output_tokens,
                batch_size,
                args.checkpoint_interval,
                not args.no_cache,
                output_dir,
                process_verdict_payload,
            )
            verdict_rows_by_id = read_jsonl_latest(verdicts_path)

        if args.stage == "both":
            correction_tables = select_error_tables(tables, verdict_rows_by_id)
            correction_rows_by_id = read_jsonl_latest(correction_tables_path)
            correction_tasks = build_tasks(
                correction_tables, tables_json, correction_rows_by_id, "correction"
            )

        if correction_tasks:
            call_batches(
                wrapper,
                correction_tasks,
                "correction",
                CORRECTION_PROMPT_VERSION,
                args.model,
                args.max_output_tokens,
                batch_size,
                args.checkpoint_interval,
                not args.no_cache,
                output_dir,
                process_correction_payload,
            )
    finally:
        wrapper.flush_cache(wait=True)

    summary = summarize(output_dir, len(tables), args.model)
    summary["elapsed_seconds_this_process"] = time.time() - started_at
    write_json(os.path.join(output_dir, "summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
