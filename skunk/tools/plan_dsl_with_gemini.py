"""DSL planning tool — direct google.genai, no Palimpzest.

Generates a DSL plan for every question in the benchmark CSV using Gemini 2.5
Flash. The 5-op DSL is described in DSL.md; this script's system prompt is the
same content. Output is data/dsl_planning_pass.csv with columns:
  uid, question, plan_text, parse_ok, validate_ok, head_op, tail_op, ops_used, problems

Usage:
  python tools/plan_dsl_with_gemini.py            # full pass over data/officeqa_pro.csv
  python tools/plan_dsl_with_gemini.py --smoke    # 7 representative questions
  python tools/plan_dsl_with_gemini.py --retry-failed  # only re-plan validate_ok=False rows
"""

from __future__ import annotations

import argparse
import os
import re
import time
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"
CSV = REPO_ROOT / "data" / "officeqa_pro.csv"
OUT_CSV = REPO_ROOT / "data" / "dsl_planning_pass.csv"

SMOKE_UIDS = ["UID0001", "UID0004", "UID0029", "UID0030", "UID0035", "UID0055", "UID0010"]

VALID_OPS = {"retrieve", "extract", "read_visual", "lookup_external", "compute"}
HEAD_OPS = {"retrieve", "lookup_external"}
TAIL_OP = "compute"


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


_load_env(ENV_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# DSL spec — kept in sync with DSL.md
# ─────────────────────────────────────────────────────────────────────────────

DSL_SPEC = """\
DSL — exactly 5 operators. Compose left-to-right with `-->`. Parallel branches: `[c1 ; c2 ; ...] --> next_op`.

QUOTING RULE — IMPORTANT
  All string argument values use SINGLE quotes ('...'), NEVER double quotes.
  This keeps plans JSON-safe when the planner output is wrapped in JSON.
  Example: retrieve(concept='national_defense', period='CY1940')

TYPES
  DocHandle         — list of pages (file_path, year, month, page).
  TypedValue        — value + dtype + unit + desc; for extract output, dtype='named' and value is a dict.
  FormattedString   — final answer text (only compute returns this).

OPERATORS

retrieve(concept, period, source_bulletin?) → DocHandle    [chain head only]
  concept:         str               domain tag(s) the page must report on
  period:          period string     CY1940, FY1981, Q3-1982, 1940-09, 2025-03-31, CY1940..CY1949, or comma-list
  source_bulletin: 'YYYY-MM'         optional; pin to one issue when the question explicitly names a bulletin

extract() → TypedValue
  No args. Reads ctx.question + the page text and returns a TypedValue with dtype='named' whose
  .value is a dict mapping snake_case names to scalars/lists/tables relevant to the question.
  The agent decides what's worth extracting; do NOT specify a 'concept' or 'mode'.

read_visual() → TypedValue
  No args. Same output shape as extract, but always uses vision (for charts/figures).

lookup_external(nl) → TypedValue                           [chain head capable]
  nl: str   natural-language description of the external data to look up
            (FX rate, CPI, event year, named entity, etc.).

compute() → FormattedString                                [chain terminator]
  No args. Receives ctx.question + the upstream extracted/looked-up values, plans the
  computation, generates Python, runs it, and a verifier LLM checks the output's format/unit
  matches what the question asks for. Returns the answer as a string. Fails with StepFailed if
  values are missing or all attempts fail.

PERIOD GRAMMAR
  Point:        CY1940, FY1981, Q3-1982, 1940-09, 2025-03-31
  Range:        CY1940..CY1949
  Enumeration:  1991-06,1996-06,2001-06

COMPOSITION RULES
  Chain head MUST be retrieve or lookup_external.
  Chain tail MUST be compute (it produces the final answer string).
  In `a --> b`: output_type(a) ∈ accepted_inputs(b).
  In `[c1; c2; ...] --> next`: every c_i ends in the same type; next accepts list[that_type].

Valid ops: retrieve, extract, read_visual, lookup_external, compute. No others.
"""

FEW_SHOT = """\
EXAMPLES — match this style precisely. Note SINGLE quotes around all string values.
extract() and compute() take NO arguments. compute is always the chain terminator.

UID0001 — Total US national defense expenditures for CY1940
  retrieve(concept='national_defense_expenditure', period='CY1940')
    --> extract()
    --> compute()

UID0004 — Absolute pct change in CY1953 vs CY1940 monthly national defense
  [
    retrieve(concept='national_defense_expenditure', period='CY1940') --> extract();
    retrieve(concept='national_defense_expenditure', period='CY1953') --> extract()
  ]
    --> compute()

UID0029 — Bulletin published in June 1970, average yield spread CY1960-69
  retrieve(concept='bond_yields', period='CY1960..CY1969', source_bulletin='1970-06')
    --> extract()
    --> compute()

UID0030 — Local maxima on line plots, page 5 of Sept 1990 bulletin
  retrieve(concept='line_plots_on_page', period='1990-09', source_bulletin='1990-09')
    --> read_visual()
    --> compute()

UID0035 — Benford first-digit count on a whole table
  retrieve(concept='receipts_table', period='1980-05', source_bulletin='1980-05')
    --> extract()
    --> compute()

UID0010 — USD->JPY conversion of Treasury investment as of 2025-03-31
  [
    retrieve(concept='foreign_exchange_securities_investments_japanese_yen', period='2025-03-31')
      --> extract();
    lookup_external(nl='USD/JPY exchange rate on 2025-03-31, Macrotrends')
  ]
    --> compute()

UID0055 — WWII end to Korean War start: change in Moody Aaa yield
  [
    lookup_external(nl='year WWII ended');
    lookup_external(nl='year Korean War started')
  ]
    --> retrieve(concept='moody_aaa_corporate_bond_yield', period='prev')
    --> extract()
    --> compute()
NOTE: period='prev' means the period is bound from the previous step's value at execution time.
"""

PLANNER_SYSTEM = (
    "You are a DSL planner for the OfficeQA benchmark. "
    "Given a question, output ONLY the DSL plan in surface text grammar — "
    "no JSON, no markdown fences, no commentary, no quotes around it.\n\n"
    + DSL_SPEC + "\n" + FEW_SHOT
)


# ─────────────────────────────────────────────────────────────────────────────
# Direct Gemini API call
# ─────────────────────────────────────────────────────────────────────────────

_gemini_client = None


def plan_via_gemini(question: str, repair_context: str | None = None) -> str:
    """Direct google.genai call. Plain-text plan output, no JSON wrapping."""
    global _gemini_client
    if _gemini_client is None:
        from google import genai  # noqa: PLC0415
        _gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    from google.genai import types  # noqa: PLC0415

    user = f"Question: {question}\n\nReturn the DSL plan for this question."
    if repair_context:
        user += "\n\n" + repair_context

    resp = _gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=PLANNER_SYSTEM,
            # 16k allows for thinking + the longest plans we've seen (~3k chars).
            max_output_tokens=16384,
            temperature=0.0,
        ),
    )
    text = (resp.text or "").strip()
    # strip markdown fences if the LLM wrapped despite instructions
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def plan_with_retry(question: str, max_attempts: int = 3) -> tuple[str, dict]:
    """Generate a plan; retry up to `max_attempts` with feedback if validation fails."""
    plan = ""
    analysis: dict = {}
    for attempt in range(max_attempts):
        try:
            if attempt == 0:
                plan = plan_via_gemini(question)
            else:
                repair = (
                    "Structural issues: "
                    f"{', '.join(analysis.get('problems', [])) or 'unspecified'}.\n"
                    "Failing plan:\n"
                    f"{plan}\n\n"
                    "Regenerate with strict adherence to the DSL grammar:\n"
                    "  • all `[`/`]` and `(`/`)` MUST be balanced\n"
                    "  • single-quoted strings only — never triple-quotes (`'''`) or double-quotes\n"
                    "  • the OUTER chain MUST start with `retrieve(...)` or `lookup_external(...)` and end with `compute()`\n"
                    "  • only the 5 ops: retrieve, extract, read_visual, lookup_external, compute"
                )
                plan = plan_via_gemini(question, repair_context=repair)
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                time.sleep(5 * (attempt + 1))
                continue
            raise
        analysis = analyze_plan(plan)
        if analysis["validate_ok"]:
            return plan, analysis
    return plan, analysis


# ─────────────────────────────────────────────────────────────────────────────
# DSL parser & validator (lightweight — for grammar checks only)
# ─────────────────────────────────────────────────────────────────────────────

OP_NAME_RE = re.compile(r"\b([a-z_][a-z_0-9]*)\s*\(")


def _strip_string_literals(text: str) -> str:
    """Replace contents of '...' / \"...\" with spaces.

    Prevents Python builtins inside compute(code='result = sum(...)') from
    being misidentified as DSL ops by OP_NAME_RE.
    """
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in ('"', "'"):
            quote = ch
            out.append(quote)
            i += 1
            while i < len(text) and text[i] != quote:
                if text[i] == "\\" and i + 1 < len(text):
                    out.append(" "); out.append(" "); i += 2
                else:
                    out.append(" "); i += 1
            if i < len(text):
                out.append(quote); i += 1
        else:
            out.append(ch); i += 1
    return "".join(out)


def _strip_for_parse(text: str) -> str:
    text = re.sub(r"#.*", "", text)
    text = re.sub(r"\bNOTE:.*$", "", text, flags=re.MULTILINE)
    return text.strip()


def _balanced(text: str, open_ch: str, close_ch: str) -> bool:
    depth = 0
    for ch in text:
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _outer_chain_head(text: str) -> str | None:
    s = text.lstrip()
    while s.startswith("["):
        s = s[1:].lstrip()
    m = OP_NAME_RE.match(s)
    return m.group(1) if m else None


def _outer_chain_tail(text: str) -> str | None:
    depth_b = depth_p = 0
    last_arrow_end = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "[":
            depth_b += 1
        elif ch == "]":
            depth_b -= 1
        elif ch == "(":
            depth_p += 1
        elif ch == ")":
            depth_p -= 1
        elif depth_b == 0 and depth_p == 0 and text[i:i+3] == "-->":
            last_arrow_end = i + 3
            i += 3
            continue
        i += 1
    s = text[last_arrow_end:].lstrip()
    while s.startswith("["):
        depth = 1
        j = 1
        while j < len(s) and depth > 0:
            if s[j] == "[":
                depth += 1
            elif s[j] == "]":
                depth -= 1
            j += 1
        inner = s[1:j-1]
        d = 0
        last_semi = -1
        for k, c in enumerate(inner):
            if c == "[":
                d += 1
            elif c == "]":
                d -= 1
            elif c == ";" and d == 0:
                last_semi = k
        s = (inner[last_semi+1:] if last_semi >= 0 else inner).lstrip()
    m = OP_NAME_RE.match(s)
    return m.group(1) if m else None


def analyze_plan(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {"parse_ok": False, "validate_ok": False, "ops": [],
                "head": None, "tail": None, "problems": ["empty_plan"]}

    cleaned = _strip_for_parse(text)
    op_scan = _strip_string_literals(cleaned)
    problems: list[str] = []

    if not _balanced(cleaned, "[", "]"):
        problems.append("unbalanced_brackets")
    if not _balanced(cleaned, "(", ")"):
        problems.append("unbalanced_parens")

    ops = OP_NAME_RE.findall(op_scan)
    for op in ops:
        if op not in VALID_OPS:
            problems.append(f"invalid_op:{op}")

    head = _outer_chain_head(op_scan) if not problems else None
    tail = _outer_chain_tail(op_scan) if not problems else None

    if head is not None and head not in HEAD_OPS:
        problems.append(f"head_not_retrieve_or_lookup_external:{head}")
    if tail is not None and tail != TAIL_OP:
        problems.append(f"tail_not_compute:{tail}")

    parse_ok = "unbalanced_brackets" not in problems and "unbalanced_parens" not in problems
    validate_ok = parse_ok and not problems
    return {"parse_ok": parse_ok, "validate_ok": validate_ok,
            "ops": ops, "head": head, "tail": tail, "problems": problems}


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def _is_empty(s: object) -> bool:
    return s is None or (isinstance(s, float) and pd.isna(s)) or \
           (isinstance(s, str) and (not s.strip() or s.strip().lower() == "nan"))


def _row_from_analysis(uid: str, question: str, plan: str, analysis: dict) -> dict:
    return {
        "uid": uid,
        "question": question,
        "plan_text": plan,
        "parse_ok": analysis["parse_ok"],
        "validate_ok": analysis["validate_ok"],
        "head_op": analysis.get("head") or "",
        "tail_op": analysis.get("tail") or "",
        "ops_used": "|".join(analysis.get("ops", [])),
        "problems": "|".join(analysis.get("problems", [])),
    }


def _print_summary(df: pd.DataFrame) -> None:
    n = len(df)
    parse_rate = df["parse_ok"].astype(bool).sum() / n
    validate_rate = df["validate_ok"].astype(bool).sum() / n
    print("\n" + "=" * 60)
    print(f"DSL PLANNING PASS — {n} questions")
    print("=" * 60)
    print(f"  parse_ok:    {df['parse_ok'].astype(bool).sum()}/{n}  ({100*parse_rate:.1f}%)")
    print(f"  validate_ok: {df['validate_ok'].astype(bool).sum()}/{n}  ({100*validate_rate:.1f}%)")

    op_freq: Counter = Counter()
    for ops in df["ops_used"]:
        if isinstance(ops, str) and ops:
            op_freq.update(ops.split("|"))
    print("\n  Op frequency:")
    for op, ct in op_freq.most_common():
        print(f"    {op:18s} {ct}")

    plan_lens = df["ops_used"].apply(lambda s: 0 if not isinstance(s, str) or not s else s.count("|") + 1)
    print(f"\n  Plan length: median={plan_lens.median():.0f} ops, "
          f"mean={plan_lens.mean():.1f}, max={plan_lens.max()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help=f"Run on {len(SMOKE_UIDS)} representative rows only.")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Load existing dsl_planning_pass.csv and re-plan only validate_ok=False rows.")
    args = parser.parse_args()

    if args.retry_failed:
        if not OUT_CSV.exists():
            raise FileNotFoundError(f"no existing CSV at {OUT_CSV}; run a full pass first")
        df = pd.read_csv(OUT_CSV)
        failed = df[~df["validate_ok"].astype(bool)]
        print(f"[retry] {len(failed)}/{len(df)} rows validate_ok=False — retrying with feedback...")
        for _, r in failed.iterrows():
            print(f"  [retry] {r['uid']}  prev_problems={r['problems']}")
            plan, analysis = plan_with_retry(r["question"], max_attempts=3)
            idx = df.index[df["uid"] == r["uid"]][0]
            for k, v in _row_from_analysis(r["uid"], r["question"], plan, analysis).items():
                df.at[idx, k] = v
            status = "✓" if analysis["validate_ok"] else "✗"
            print(f"    {status}  validate_ok={analysis['validate_ok']}  problems={analysis.get('problems')}")
        df.to_csv(OUT_CSV, index=False)
        n_ok = df["validate_ok"].astype(bool).sum()
        print(f"\n[retry] wrote → {OUT_CSV}  validate_ok now {n_ok}/{len(df)}")
        return

    df = pd.read_csv(CSV)
    if args.smoke:
        df = df[df["uid"].isin(SMOKE_UIDS)].reset_index(drop=True)
    print(f"[run] planning {len(df)} questions with Gemini 2.5 Flash...")

    rows = []
    for i, r in df.iterrows():
        plan, analysis = plan_with_retry(r["question"], max_attempts=3)
        rows.append(_row_from_analysis(r["uid"], r["question"], plan, analysis))
        if (i + 1) % 10 == 0 or i == len(df) - 1:
            print(f"  [{i+1}/{len(df)}] {r['uid']} validate_ok={analysis['validate_ok']}")

    out_df = pd.DataFrame(rows)
    out_path = OUT_CSV if not args.smoke else OUT_CSV.with_name("dsl_planning_pass_smoke.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\n[out] wrote {len(out_df)} rows to {out_path}")
    _print_summary(out_df)


if __name__ == "__main__":
    main()
