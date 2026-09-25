"""Reconstruct datagen trajectories from a `follow_up_questions.py` run directory.

A run directory (see `follow_up_questions.main`) holds:

- `records.jsonl`          one row per seed question (last row per qid wins on resume)
- `generation_stats.jsonl` one `GenStats` row per agent run, in generation order
- `traces/<qid>.jsonl`     the question's flat `ExecutionContext` event stream
- `run_config.json`, `qa_pairs.json`, `usage_summary.json`

The event stream is *flat*: every agent that ran for a question (inquiry synth, duplicate
judge, follow-up synth, solver, equivalence judge) appended to the same file with no
agent tag. Each agent run does, however, start with exactly one `system_prompt` event
(`MultiTurnAgent.call` emits system + user before its first step), so the stream splits
cleanly into *segments* at those events. The stats rows are appended in the same
sequential order by `_run_agent`, so walking rows and segments together in order aligns
each agent run with its cost/latency/outcome row. Rows that made no LLM call (the
embed-only dedup check when the inquiry collection is empty) have no segment.

The result is a nested trajectory:

    seed question
      -> inquiry attempts[try]   = { gen, dedup }                (steps 2 -> 3, loop on !unique)
      -> followups[idx].attempts[try] = { gen, solver, equivalence, dedup }
                                                                  (steps 4 -> 5 -> 6, loop on !solvable | !unique)
"""

from __future__ import annotations

import functools
import json
import pathlib
import re

# ── constants ────────────────────────────────────────────────────────────────

# `usage_key` (== agent_id) prefixes written by follow_up_questions.py → node type
AGENT_PREFIX_TO_TYPE = {
    "inquiry_agent": "inquiry",
    "follow_up_agent": "follow_up",
    "solver_agent": "solver",
    "equivalence_agent": "equivalence",
    "duplicate_agent": "dedup",
}
# `call call_site=<name>` on LLM call events → node type (embed calls carry call_site=embed)
CALL_SITE_TO_TYPE = {
    "inquiry_synth_agent": "inquiry",
    "follow_up_question_synth_agent": "follow_up",
    "solver_agent": "solver",
    "equivalence_agent": "equivalence",
    "duplicate_agent": "dedup",
}
# fallback typing from the opening words of the system prompt (prompts.yaml)
PROMPT_PREFIX_TO_TYPE = [
    ("You are a data-generation assistant. You will be presented with an initial", "inquiry"),
    ("You are a data-generation assistant. You will be presented with one or more", "follow_up"),
    ("You are judging whether a generated line of inquiry", "inquiry_dedup"),
    ("You are judging whether a generated question-answer pair", "qa_dedup"),
    ("You are an agent whose job is to answer a question", "solver"),
    ("You are an agent whose job is to determine whether two answers", "equivalence"),
]
# stats `kind` disambiguates the two duplicate_agent uses
DEDUP_KIND_TO_TYPE = {"line_of_inquiry_dedup": "inquiry_dedup", "qa_pair_dedup": "qa_dedup"}

_CALL_SITE_RE = re.compile(r"\bcall_site=(\S+)")
_JSON_FENCE_RE = re.compile(r"```json\s*(.*?)```", re.S)
_ASSIST_STEP_RE = re.compile(r"\bstep=(\d+)")
_DOC_ID_RE = re.compile(r"'doc_id': '([^']+)'")
_DOC_HEADER_RE = re.compile(r"=== Document ID: (\S+) ===")
_N_DOCS_RE = re.compile(r"grounded in \*\*(\d+) supporting doc")
_N_NUGGETS_RE = re.compile(r"consist of \*{0,2}(\d+) nugget")  # tolerates the older `*N nuggets.` wording
_TOOL_NAME_RE = re.compile(r"([A-Za-z_]\w*)\s*\(")


# ── file loaders (cached on path + mtime) ────────────────────────────────────


def _mtime(path: pathlib.Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return -1.0


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


@functools.lru_cache(maxsize=64)
def _records_cached(path: str, _mt: float) -> list[dict]:
    return _read_jsonl(pathlib.Path(path))


@functools.lru_cache(maxsize=64)
def _stats_cached(path: str, _mt: float) -> list[dict]:
    return _read_jsonl(pathlib.Path(path))


@functools.lru_cache(maxsize=256)
def _events_cached(path: str, _mt: float) -> list[dict]:
    events = _read_jsonl(pathlib.Path(path))
    for i, e in enumerate(events):
        e["_idx"] = i
        # user/observation events repeat their whole payload in `message` (a repr of `data`);
        # keep only a short head so the per-question payload isn't doubled
        d = e.get("data")
        if isinstance(d, dict) and ("text" in d or "blocks" in d) and len(str(e.get("message", ""))) > 200:
            e["message"] = str(e["message"])[:200] + " …"
    return events


def load_records(run_dir: pathlib.Path) -> dict[str, dict]:
    """Last record per qid (a resumed run's successful retry supersedes its failed row)."""
    p = run_dir / "records.jsonl"
    out: dict[str, dict] = {}
    for r in _records_cached(str(p), _mtime(p)):
        out[str(r.get("qid"))] = r
    return out


def load_stats(run_dir: pathlib.Path) -> list[dict]:
    p = run_dir / "generation_stats.jsonl"
    return _stats_cached(str(p), _mtime(p))


def load_events(run_dir: pathlib.Path, qid: str) -> list[dict]:
    p = run_dir / "traces" / f"{qid}.jsonl"
    return _events_cached(str(p), _mtime(p))


def load_json(path: pathlib.Path) -> dict | list | None:
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def trace_qids(run_dir: pathlib.Path) -> list[str]:
    return sorted(p.stem for p in (run_dir / "traces").glob("*.jsonl"))


# ── stats rows ───────────────────────────────────────────────────────────────


def parse_usage_key(usage_key: str, qid: str) -> tuple[str, int | None, int] | None:
    """`<agent>_<qid>[_<idx>]_try<n>` → (node type, idx, attempt); None when not this qid's."""
    for prefix, typ in AGENT_PREFIX_TO_TYPE.items():
        head = f"{prefix}_{qid}"
        if not usage_key.startswith(head):
            continue
        m = re.fullmatch(r"(?:_(\d+))?_try(\d+)", usage_key[len(head) :])
        if m:
            return typ, (int(m.group(1)) if m.group(1) is not None else None), int(m.group(2))
    return None


def rows_for_qid(stats: list[dict], qid: str) -> list[dict]:
    """This question's stats rows, in file order, each annotated with `_type/_idx/_attempt`."""
    out: list[dict] = []
    for s in stats:
        sq = str(s.get("qid", ""))
        if sq != qid and not sq.startswith(qid + "_"):
            continue
        parsed = parse_usage_key(str(s.get("usage_key", "")), qid)
        if parsed is None:
            continue
        typ, idx, attempt = parsed
        if typ == "dedup":
            typ = DEDUP_KIND_TO_TYPE.get(str(s.get("kind")), "dedup")
        row = dict(s)
        row["_type"], row["_idx"], row["_attempt"] = typ, idx, attempt
        out.append(row)
    return out


def _row_expects_segment(row: dict) -> bool:
    """Did this row's agent actually start (and hence emit a system_prompt event)?"""
    return int(row.get("n_llm_calls") or 0) > 0 or row.get("terminate_state") == "error"


# ── event segmentation ───────────────────────────────────────────────────────


def _event_text(e: dict) -> str:
    d = e.get("data") or {}
    if isinstance(d.get("text"), str):
        return d["text"]
    if isinstance(d.get("blocks"), list):
        return "".join(str(b.get("text", "")) for b in d["blocks"] if isinstance(b, dict))
    return str(e.get("message", ""))


def _segment_type(events: list[dict]) -> str:
    sys_text = _event_text(events[0]) if events else ""
    typ = None
    for e in events:
        if e.get("kind") != "call":
            continue
        m = _CALL_SITE_RE.search(str(e.get("message", "")))
        if m and m.group(1) in CALL_SITE_TO_TYPE:
            typ = CALL_SITE_TO_TYPE[m.group(1)]
            break
    if typ is None:
        for prefix, t in PROMPT_PREFIX_TO_TYPE:
            if sys_text.startswith(prefix):
                typ = t
                break
    if typ == "dedup":
        typ = "inquiry_dedup" if "line of inquiry" in sys_text[:200].lower() else "qa_dedup"
    return typ or "unknown"


def parse_json_payload(text: str) -> dict | None:
    """The agent's final JSON object: last ```json fence, tolerating Python literals."""
    candidates = [m.group(1) for m in _JSON_FENCE_RE.finditer(text)]
    if not candidates:
        a, b = text.find("{"), text.rfind("}")
        if a >= 0 and b > a:
            candidates = [text[a : b + 1]]
    for cand in reversed(candidates):
        for attempt in (
            cand,
            cand.replace(": None", ": null").replace(": True", ": true").replace(": False", ": false"),
        ):
            try:
                obj = json.loads(attempt)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    return None


def _tool_calls(events: list[dict]) -> list[dict]:
    """tool_code notes paired with the doc ids their observation surfaced."""
    calls: list[dict] = []
    for e in events:
        kind, msg = e.get("kind"), str(e.get("message", ""))
        if kind == "note" and msg.startswith("tool_code"):
            code = msg[len("tool_code") :].strip()
            if len(code) >= 2 and code[0] == code[-1] and code[0] in "'\"":
                code = code[1:-1]
            code = code.replace("\\n", "\n").replace("\\t", "\t").replace("\\'", "'").replace('\\"', '"')
            first = next((ln.strip() for ln in code.split("\n") if ln.strip() and not ln.strip().startswith("#")), code)
            m = _TOOL_NAME_RE.search(first)
            calls.append(
                {"name": m.group(1) if m else "", "code": code, "event_idx": e["_idx"], "doc_ids": [], "obs_chars": 0}
            )
        elif kind == "observation" and calls:
            text = _event_text(e)
            seen: list[str] = []
            for d in _DOC_ID_RE.findall(text) + _DOC_HEADER_RE.findall(text) + re.findall(r"doc_id=(\S+?) \|", text):
                if d not in seen:
                    seen.append(d)
            calls[-1]["doc_ids"] = seen
            calls[-1]["obs_chars"] = len(text)
    return calls


def segment_events(events: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split a flat stream into per-agent segments at `system` events.

    Returns (segments, loose_events); loose events precede the first system prompt."""
    segments: list[dict] = []
    loose: list[dict] = []
    cur: list[dict] | None = None
    for e in events:
        if e.get("kind") == "system":
            cur = [e]
            segments.append({"events": cur})
            continue
        if cur is None:
            loose.append(e)
        else:
            cur.append(e)

    for i, seg in enumerate(segments):
        evs = seg["events"]
        ts = [e["t"] for e in evs if isinstance(e.get("t"), (int, float))]
        assistant = [e for e in evs if e.get("kind") == "assistant"]
        steps = [int(m.group(1)) for e in assistant for m in [_ASSIST_STEP_RE.search(str(e.get("message", "")))] if m]
        calls = [e for e in evs if e.get("kind") == "call"]
        cost = 0.0
        for c in calls:
            d = c.get("data") or {}
            if isinstance(d.get("cost"), (int, float)):
                cost += float(d["cost"])
        seg.update(
            {
                "i": i,
                "type": _segment_type(evs),
                "start_idx": evs[0]["_idx"],
                "end_idx": evs[-1]["_idx"],
                "t_start": min(ts) if ts else None,
                "t_end": max(ts) if ts else None,
                "n_llm_calls": sum(1 for c in calls if "call_site=embed" not in str(c.get("message", ""))),
                "n_embed_calls": sum(1 for c in calls if "call_site=embed" in str(c.get("message", ""))),
                "n_parse_retries": sum(1 for e in evs if str(e.get("message", "")).startswith("parse_retry")),
                "cost_usd": cost,
                "n_steps": max(steps) if steps else len(assistant),
                "system_prompt": _event_text(evs[0]),
                "user_prompt": next((_event_text(e) for e in evs if e.get("kind") == "user"), ""),
                "final_text": _event_text(assistant[-1]) if assistant else "",
                "output": parse_json_payload(_event_text(assistant[-1])) if assistant else None,
                "tool_calls": _tool_calls(evs),
            }
        )
    return segments, loose


# ── judge prompt parsing (neighbours shown to the duplicate judge) ───────────


def _parse_blocks(text: str, start_marker: str, end_marker: str, keys: list[str]) -> list[dict]:
    a = text.find(start_marker)
    if a < 0:
        return []
    b = text.find(end_marker, a)
    body = text[a + len(start_marker) : (b if b > 0 else len(text))]
    out: list[dict] = []
    for block in body.split("\n---\n"):
        block = block.strip()
        if not block:
            continue
        item: dict[str, str] = {}
        last: str | None = None
        for line in block.split("\n"):
            hit = next((k for k in keys if line.startswith(k + ":")), None)
            if hit is not None:
                item[hit] = line[len(hit) + 1 :].strip()
                last = hit
            elif last is not None:
                item[last] += "\n" + line
        if item:
            out.append(item)
    return out


def parse_dedup_neighbors(seg_type: str, system_prompt: str) -> list[dict]:
    if seg_type == "inquiry_dedup":
        items = _parse_blocks(
            system_prompt,
            "** Closest Existing Lines of Inquiry:**",
            "**Your Task:**",
            ["ID", "Line of Inquiry", "Seed Question", "Seed Answer"],
        )
        return [
            {
                "id": it.get("ID", ""),
                "line_of_inquiry": it.get("Line of Inquiry", ""),
                "question": it.get("Seed Question", ""),
                "answer": it.get("Seed Answer", ""),
            }
            for it in items
        ]
    if seg_type == "qa_dedup":
        items = _parse_blocks(
            system_prompt,
            "** Closest Existing Question-Answer Pairs:**",
            "**Your Task:**",
            ["QID", "Question", "Answer"],
        )
        return [
            {"id": it.get("QID", ""), "question": it.get("Question", ""), "answer": it.get("Answer", "")}
            for it in items
        ]
    return []


def verdict_of(output: dict | None) -> dict:
    """Normalise a duplicate-judge payload. `flagged` is what the judge *meant*; the generator
    only treats `duplicate is True` as a duplicate, so an ID string never triggers a retry."""
    if not isinstance(output, dict):
        return {"raw": None, "flagged": False, "duplicate_of": None, "reasoning": None, "parsed": False}
    dup = output.get("duplicate")
    flagged = dup not in (None, False, "", "None", "null", "false", "FALSE", "False")
    dup_of = None if not flagged else (str(dup) if dup is not True else "(true)")
    return {
        "raw": dup,
        "flagged": flagged,
        "duplicate_of": dup_of,
        "reasoning": output.get("reasoning"),
        "parsed": True,
        "script_saw_duplicate": dup is True,
    }


# ── alignment + trajectory assembly ──────────────────────────────────────────


def _align(rows: list[dict], segments: list[dict], lookahead: int = 3) -> list[dict]:
    """Attach segments to rows in order; returns orphan segments (unmatched)."""
    cursor = 0
    matched: set[int] = set()
    for row in rows:
        row["_seg"] = None
        if not _row_expects_segment(row):
            continue
        for j in range(cursor, min(len(segments), cursor + 1 + lookahead)):
            if j in matched:
                continue
            if segments[j]["type"] == row["_type"] or segments[j]["type"] == "unknown":
                row["_seg"] = j
                matched.add(j)
                cursor = j + 1
                break
    return [s for j, s in enumerate(segments) if j not in matched]


def _split_runs(rows: list[dict]) -> list[list[dict]]:
    """A resumed run re-generates a failed question from scratch (its trace file is
    rewritten but the stats rows accumulate), so split rows at every inquiry try 1."""
    runs: list[list[dict]] = []
    for r in rows:
        if r["_type"] == "inquiry" and r["_attempt"] == 1 or not runs:
            runs.append([])
        runs[-1].append(r)
    return runs


_TYPE_TO_KIND = {
    "inquiry": "line_of_inquiry",
    "inquiry_dedup": "line_of_inquiry_dedup",
    "follow_up": "follow_up_question",
    "solver": "solver",
    "equivalence": "solver",
    "qa_dedup": "qa_pair_dedup",
}


def _rows_from_segments(segments: list[dict], qid: str) -> tuple[list[dict], list[dict]]:
    """Stats-less fallback: replay the generator's control flow over the typed segments.

    Used when a question has no `generation_stats.jsonl` rows yet — a question still running
    (rows are only written once the whole question finishes) or an early smoke-test run. The
    solvable/unique flags are re-derived exactly as `follow_up_questions.py` derives them:
    solvable = exact answer match or an equivalence verdict containing "true"; unique = the
    judge payload's `duplicate is True` (so an ID string counts as unique, as in the script).
    A new follow-up segment retries the current idx when the previous attempt was rejected,
    else starts the next idx."""
    rows: list[dict] = []
    orphans: list[dict] = []
    inq_attempt = 0
    fu_idx, fu_attempt = -1, 0
    cur: dict | None = None

    def base(seg: dict, typ: str, idx: int | None, attempt: int) -> dict:
        wall = (seg["t_end"] - seg["t_start"]) if seg["t_end"] is not None and seg["t_start"] is not None else 0.0
        return {
            "usage_key": "(inferred from trace)",
            "kind": _TYPE_TO_KIND[typ],
            "qid": qid,
            "idx": idx,
            "attempt": attempt,
            "cost_usd": seg["cost_usd"],
            "wall_latency_s": round(wall, 3),
            "n_steps": seg["n_steps"],
            "n_llm_calls": seg["n_llm_calls"],
            "n_embed_calls": seg["n_embed_calls"],
            "terminate_state": "finished",
            "error": None,
            "solvable": None,
            "unique": None,
            "inferred": True,
            "_type": typ,
            "_idx": idx,
            "_attempt": attempt,
            "_seg": seg["i"],
        }

    for seg in segments:
        t = seg["type"]
        if t == "inquiry":
            inq_attempt += 1
            rows.append(base(seg, t, None, inq_attempt))
        elif t == "inquiry_dedup":
            inq_attempt = max(inq_attempt, 1)
            r = base(seg, t, None, inq_attempt)
            r["unique"] = not verdict_of(seg["output"]).get("script_saw_duplicate", False)
            rows.append(r)
        elif t == "follow_up":
            if cur is None or (cur["solvable"] is True and cur["unique"] is True):
                fu_idx, fu_attempt = fu_idx + 1, 1
            else:
                fu_attempt += 1
            cur = {"solvable": None, "unique": None, "gen_out": seg["output"], "solver_row": None}
            rows.append(base(seg, t, fu_idx, fu_attempt))
        elif t in ("solver", "equivalence", "qa_dedup"):
            if cur is None:
                orphans.append(seg)
                continue
            r = base(seg, t, fu_idx, fu_attempt)
            if t == "solver":
                exact = seg["output"] is not None and (cur["gen_out"] or {}).get("answer") == seg["output"].get(
                    "answer"
                )
                r["solvable"] = True if exact else None
                cur["solvable"], cur["solver_row"] = r["solvable"], r
            elif t == "equivalence":
                solvable = "true" in str((seg["output"] or {}).get("equal", "")).lower()
                r["solvable"] = cur["solvable"] = solvable
                if cur["solver_row"] is not None:
                    cur["solver_row"]["solvable"] = solvable
            else:
                r["unique"] = cur["unique"] = not verdict_of(seg["output"]).get("script_saw_duplicate", False)
            rows.append(r)
        else:
            orphans.append(seg)
    return rows, orphans


def _node(row: dict | None, segments: list[dict]) -> dict | None:
    if row is None:
        return None
    seg = segments[row["_seg"]] if row.get("_seg") is not None else None
    node = {
        "stats": {k: v for k, v in row.items() if not k.startswith("_")},
        "segment": seg,
        "output": seg["output"] if seg else None,
    }
    if seg and seg["type"] in ("inquiry_dedup", "qa_dedup"):
        node["neighbors"] = parse_dedup_neighbors(seg["type"], seg["system_prompt"])
        node["verdict"] = verdict_of(seg["output"])
    elif row["_type"] in ("inquiry_dedup", "qa_dedup"):
        node["neighbors"] = []
        node["verdict"] = {
            "raw": None,
            "flagged": False,
            "duplicate_of": None,
            "reasoning": None,
            "parsed": False,
            "note": "no judge ran (embedding-only check: collection was empty)",
        }
    if seg and seg["type"] == "follow_up":
        m = _N_DOCS_RE.search(seg["system_prompt"])
        node["n_docs_target"] = int(m.group(1)) if m else None
        m = _N_NUGGETS_RE.search(seg["system_prompt"])
        node["n_nuggets_target"] = int(m.group(1)) if m else None
    return node


def build_trajectory(run_dir: pathlib.Path, qid: str) -> dict:
    record = load_records(run_dir).get(qid)
    rows = rows_for_qid(load_stats(run_dir), qid)
    events = load_events(run_dir, qid)
    segments, loose = segment_events(events)

    runs = _split_runs(rows)
    current = runs[-1] if runs else []
    superseded = [r for run in runs[:-1] for r in run]
    inferred = False
    if current:
        orphans = _align(current, segments)
    else:  # no stats rows (yet): infer the structure from the trace alone
        current, orphans = _rows_from_segments(segments, qid)
        inferred = bool(current)

    # ── inquiry attempts (steps 2 → 3)
    inquiry: dict[int, dict] = {}
    for r in current:
        if r["_type"] in ("inquiry", "inquiry_dedup"):
            a = inquiry.setdefault(r["_attempt"], {"attempt": r["_attempt"], "gen": None, "dedup": None})
            a["gen" if r["_type"] == "inquiry" else "dedup"] = _node(r, segments)
    inquiry_attempts = [inquiry[k] for k in sorted(inquiry)]
    for a in inquiry_attempts:
        ded = a["dedup"]
        unique = ded["stats"].get("unique") if ded else None
        a["unique"] = unique
        a["accepted"] = unique is True
        a["error"] = (a["gen"] or {}).get("stats", {}).get("error") if a["gen"] else None
        a["verdict_flagged"] = bool(ded and ded.get("verdict", {}).get("flagged"))
        a["verdict_ignored"] = bool(a["verdict_flagged"] and unique is True)
        a["line_of_inquiry"] = ((a["gen"] or {}).get("output") or {}).get("line_of_inquiry") if a["gen"] else None
        a["doc_ids"] = ((a["gen"] or {}).get("output") or {}).get("doc_ids") if a["gen"] else None

    # ── follow-up attempts (steps 4 → 5 → 6)
    fups: dict[int, dict] = {}
    for r in current:
        if r["_type"] in ("follow_up", "solver", "equivalence", "qa_dedup") and r["_idx"] is not None:
            f = fups.setdefault(r["_idx"], {"idx": r["_idx"], "attempts": {}})
            a = f["attempts"].setdefault(
                r["_attempt"],
                {"attempt": r["_attempt"], "gen": None, "solver": None, "equivalence": None, "dedup": None},
            )
            key = {"follow_up": "gen", "solver": "solver", "equivalence": "equivalence", "qa_dedup": "dedup"}[
                r["_type"]
            ]
            a[key] = _node(r, segments)
    final_pairs = {p.get("idx"): p for p in (record or {}).get("qa_pairs", []) if p.get("idx") is not None}
    followups = []
    for idx in sorted(fups):
        f = fups[idx]
        attempts = [f["attempts"][k] for k in sorted(f["attempts"])]
        for a in attempts:
            out = (a["gen"] or {}).get("output") or {}
            a["question"], a["answer"], a["doc_ids"] = out.get("question"), out.get("answer"), out.get("doc_ids")
            a["n_docs_target"] = (a["gen"] or {}).get("n_docs_target")
            a["n_nuggets_target"] = (a["gen"] or {}).get("n_nuggets_target")
            solv_row = (a["solver"] or {}).get("stats", {})
            a["solvable"] = solv_row.get("solvable") if a["solver"] else None
            a["solver_answer"] = ((a["solver"] or {}).get("output") or {}).get("answer") if a["solver"] else None
            a["exact_match"] = bool(a["solver"]) and a["equivalence"] is None and a["solvable"] is True
            eq_out = ((a["equivalence"] or {}).get("output") or {}) if a["equivalence"] else {}
            a["equivalence_verdict"] = eq_out.get("equal")
            ded = a["dedup"]
            a["unique"] = ded["stats"].get("unique") if ded else None
            a["dedup_skipped"] = ded is None and a["solvable"] is not True
            a["verdict_flagged"] = bool(ded and ded.get("verdict", {}).get("flagged"))
            a["verdict_ignored"] = bool(a["verdict_flagged"] and a["unique"] is True)
            a["accepted"] = a["solvable"] is True and a["unique"] is True
            a["error"] = next(
                (
                    n["stats"].get("error")
                    for n in (a["gen"], a["solver"], a["equivalence"])
                    if n and n["stats"].get("error")
                ),
                None,
            )
            if a["accepted"]:
                a["reject_reason"] = None
            elif a["error"]:
                a["reject_reason"] = "error"
            elif a["solvable"] is False:
                a["reject_reason"] = "unsolvable"
            elif a["unique"] is False:
                a["reject_reason"] = "duplicate"
            else:
                a["reject_reason"] = "incomplete"
        followups.append(
            {
                "idx": idx,
                "attempts": attempts,
                "final": final_pairs.get(idx),
                "accepted_attempt": next((a["attempt"] for a in attempts if a["accepted"]), None),
                "n_docs_target": next(
                    (a["n_docs_target"] for a in attempts if a.get("n_docs_target") is not None),
                    (final_pairs.get(idx) or {}).get("n_docs_target"),
                ),
                "n_nuggets_target": next(
                    (a["n_nuggets_target"] for a in attempts if a.get("n_nuggets_target") is not None),
                    (final_pairs.get(idx) or {}).get("n_nuggets_target"),
                ),
            }
        )

    if inferred and inquiry_attempts and followups and not any(a["accepted"] for a in inquiry_attempts):
        inquiry_attempts[-1]["accepted"] = True  # no dedup step in this trace, yet follow-ups ran

    # seed doc ids: from the record when it succeeded, else from the inquiry user prompt
    seed_doc_ids: list[str] = []
    if record and record.get("qa_pairs"):
        seed_doc_ids = list(record["qa_pairs"][0].get("doc_ids") or [])
    if not seed_doc_ids:
        first_inq = next((s for s in segments if s["type"] == "inquiry"), None)
        if first_inq:
            seed_doc_ids = list(dict.fromkeys(_DOC_HEADER_RE.findall(first_inq["user_prompt"])))

    accepted_inquiry = next((a for a in inquiry_attempts if a["accepted"]), None)
    # the prompts/final text are already present as events; drop the duplicate copies from the payload
    for s in segments:
        for k in ("system_prompt", "user_prompt", "final_text"):
            s.pop(k, None)
    return {
        "qid": qid,
        "record": record,
        "inferred": inferred,
        "seed": {
            "question": (record or {}).get("seed_question") or _seed_question_from_prompt(segments),
            "answer": (record or {}).get("seed_answer"),
            "doc_ids": seed_doc_ids,
        },
        "line_of_inquiry": accepted_inquiry["line_of_inquiry"] if accepted_inquiry else None,
        "inquiry_attempts": inquiry_attempts,
        "followups": followups,
        "loose_events": loose,
        "orphan_segments": orphans,
        "superseded_rows": [
            {k: v for k, v in r.items() if not k.startswith("_")}
            | {"type": r["_type"], "idx": r["_idx"], "attempt": r["_attempt"]}
            for r in superseded
        ],
        "n_events": len(events),
        "n_segments": len(segments),
        "totals": _totals(current),
    }


def _seed_question_from_prompt(segments: list[dict]) -> str | None:
    first_inq = next((s for s in segments if s["type"] == "inquiry"), None)
    if not first_inq:
        return None
    m = re.search(r"Question: (.*?)\nAnswer:", first_inq["user_prompt"], re.S)
    return m.group(1).strip() if m else None


def _totals(rows: list[dict]) -> dict:
    return {
        "cost_usd": sum(float(r.get("cost_usd") or 0) for r in rows),
        "wall_latency_s": sum(float(r.get("wall_latency_s") or 0) for r in rows),
        "n_generations": len(rows),
        "n_errors": sum(1 for r in rows if r.get("error")),
    }


# ── run-level summaries (stats only — cheap) ─────────────────────────────────


def question_summary(rows: list[dict]) -> dict:
    """Per-question funnel counts from this question's (annotated) stats rows."""
    runs = _split_runs(rows)
    cur = runs[-1] if runs else []
    inquiry_tries = sum(1 for r in cur if r["_type"] == "inquiry")
    fu_tries = sum(1 for r in cur if r["_type"] == "follow_up")
    unsolvable = sum(1 for r in cur if r["_type"] == "solver" and r.get("solvable") is False)
    dup_inq = sum(1 for r in cur if r["_type"] == "inquiry_dedup" and r.get("unique") is False)
    dup_qa = sum(1 for r in cur if r["_type"] == "qa_dedup" and r.get("unique") is False)
    idxs = {r["_idx"] for r in cur if r["_type"] == "follow_up"}
    return {
        "inquiry_tries": inquiry_tries,
        "followup_tries": fu_tries,
        "n_followups_started": len(idxs),
        "n_unsolvable": unsolvable,
        "n_dup_inquiry": dup_inq,
        "n_dup_qa": dup_qa,
        "n_errors": sum(1 for r in cur if r.get("error")),
        "cost_usd": sum(float(r.get("cost_usd") or 0) for r in cur),
        "wall_latency_s": sum(float(r.get("wall_latency_s") or 0) for r in cur),
        "n_superseded_rows": sum(len(x) for x in runs[:-1]),
    }


def run_overview(run_dir: pathlib.Path) -> dict:
    records = load_records(run_dir)
    stats = load_stats(run_dir)
    qids = sorted(set(records) | set(trace_qids(run_dir)))
    trace_set = set(trace_qids(run_dir))
    questions = []
    funnel = {
        "n_questions": len(qids),
        "n_done": 0,
        "n_failed": 0,
        "n_in_progress": 0,
        "n_qa_pairs": 0,
        "inquiry_tries": 0,
        "followup_tries": 0,
        "n_unsolvable": 0,
        "n_dup_inquiry": 0,
        "n_dup_qa": 0,
        "cost_usd": 0.0,
    }
    for qid in qids:
        rec = records.get(qid)
        rows = rows_for_qid(stats, qid)
        inferred = False
        if not rows and qid in trace_set:
            rows, _ = _rows_from_segments(segment_events(load_events(run_dir, qid))[0], qid)
            inferred = bool(rows)
        summ = question_summary(rows)
        summ["inferred"] = inferred
        n_pairs = max(0, len((rec or {}).get("qa_pairs", [])) - 1) if rec else 0
        status = "done" if rec and not rec.get("failed") else ("failed" if rec else "in_progress")
        funnel["n_" + status] += 1
        funnel["n_qa_pairs"] += n_pairs
        for k in ("inquiry_tries", "followup_tries", "n_unsolvable", "n_dup_inquiry", "n_dup_qa", "cost_usd"):
            funnel[k] += summ[k]
        questions.append(
            {
                "qid": qid,
                "status": status,
                "seed_question": (rec or {}).get("seed_question", ""),
                "seed_answer": (rec or {}).get("seed_answer", ""),
                "error": (rec or {}).get("error", ""),
                "n_pairs": n_pairs,
                "has_trace": qid in trace_set,
                **summ,
            }
        )
    return {
        "questions": questions,
        "funnel": funnel,
        "run_config": load_json(run_dir / "run_config.json"),
        "usage_summary": load_json(run_dir / "usage_summary.json"),
    }


def run_verdicts(run_dir: pathlib.Path) -> dict[str, dict]:
    """Judge-verdict counts per question (needs the traces, so slower than `run_overview`)."""
    out: dict[str, dict] = {}
    for qid in trace_qids(run_dir):
        tr = build_trajectory(run_dir, qid)
        out[qid] = {
            "inquiry_verdict_dup": sum(1 for a in tr["inquiry_attempts"] if a["verdict_flagged"]),
            "inquiry_verdict_ignored": sum(1 for a in tr["inquiry_attempts"] if a["verdict_ignored"]),
            "qa_verdict_dup": sum(1 for f in tr["followups"] for a in f["attempts"] if a["verdict_flagged"]),
            "qa_verdict_ignored": sum(1 for f in tr["followups"] for a in f["attempts"] if a["verdict_ignored"]),
        }
    return out
