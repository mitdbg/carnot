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
DSL — 4 operators, flat plan shape. Compose left-to-right with `-->`.

PLAN SHAPE — every plan is one of:
  retrieve(concept='...', period='...'[, source_bulletin='...']) --> extract([visual_only=True]) --> compute()
  lookup_external(nl='...') --> compute()
  [ branch_1 ; branch_2 ; ... ] --> compute()

Each branch in a parallel is itself either `retrieve(...) --> extract(...)` or `lookup_external(...)`.
compute() is always the implicit terminator, always takes no args.
NO nested parallels, NO chains that skip extract after retrieve, NO compute anywhere except at the end.

QUOTING RULE — IMPORTANT
  All string argument values use SINGLE quotes ('...'), NEVER double quotes.
  This keeps plans JSON-safe when the planner output is wrapped in JSON.
  Example: retrieve(concept='national_defense', period='CY1940')

TYPES
  DocHandle         — list of pages (file_path, month, page).
  TypedValue        — keyed dict of values + per-entry meta (unit, quote, dims).
  FormattedString   — final answer text (only compute returns this).

OPERATORS

retrieve(concept, period, source_bulletin?) → DocHandle
  concept:         str               domain tag(s) the page must report on
  period:          period string     CY1940, FY1981, Q3-1982, 1940-09, 2025-03-31, CY1940..CY1949, or comma-list
  source_bulletin: 'YYYY-MM'         optional; pin to one issue when the question explicitly names a bulletin

extract(visual_only?) → TypedValue
  visual_only: bool (optional, default False). Set True for charts/figures.
  Returns a keyed dict of values relevant to the question; the agent decides what to extract.

lookup_external(nl) → TypedValue
  nl: str   natural-language description of the external data to look up
            (FX rate, CPI, event year, named entity, etc.).

compute() → FormattedString
  No args. Receives ctx.question + upstream values, plans the computation, generates Python,
  runs it, and a verifier LLM checks the output's format/unit. Returns the answer string.

PERIOD GRAMMAR
  Point:        CY1940, FY1981, Q3-1982, 1940-09, 2025-03-31
  Range:        CY1940..CY1949
  Enumeration:  1991-06,1996-06,2001-06

Valid ops: retrieve, extract, lookup_external, compute. No others.
"""

FEW_SHOT = """\
EXAMPLES — match this style precisely. Note SINGLE quotes around all string values.
extract() and compute() take NO mandatory arguments. compute is always the implicit chain terminator.

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
    --> extract(visual_only=True)
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

UID0055 — Year WWII ended and year Korean War started (parallel external lookups)
  [
    lookup_external(nl='year WWII ended');
    lookup_external(nl='year Korean War started')
  ]
    --> compute()
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
_gemini_model: str = "gemini-2.5-flash"


def plan_via_gemini(question: str, repair_context: str | None = None) -> str:
    """Direct google.genai call. Plain-text plan output, no JSON wrapping."""
    global _gemini_client, _gemini_model
    if _gemini_client is None:
        import sys  # noqa: PLC0415
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from skunk.config import SkunkConfig  # noqa: PLC0415
        from google import genai  # noqa: PLC0415
        cfg = SkunkConfig.from_env()
        _gemini_model = cfg.gemini_model
        if cfg.use_vertex:
            project = os.environ.get("GOOGLE_CLOUD_PROJECT")
            if not project:
                raise RuntimeError("GOOGLE_CLOUD_PROJECT not set (required for Vertex AI)")
            _gemini_client = genai.Client(
                vertexai=True,
                project=project,
                location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
            )
        else:
            _gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    from google.genai import types  # noqa: PLC0415

    user = f"Question: {question}\n\nReturn the DSL plan for this question."
    if repair_context:
        user += "\n\n" + repair_context

    resp = _gemini_client.models.generate_content(
        model=_gemini_model,
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
                    "  • only the 4 ops: retrieve, extract, lookup_external, compute"
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
# Plan analysis — uses the real dsl.parse() / dsl.validate()
# ─────────────────────────────────────────────────────────────────────────────


def analyze_plan(text: str) -> dict:
    """Return a dict summarising whether the plan parses and validates."""
    if not isinstance(text, str) or not text.strip():
        return {"parse_ok": False, "validate_ok": False, "ops": [],
                "head": None, "tail": None, "problems": ["empty_plan"]}

    # Import lazily so the tool can run from anywhere with the package installed.
    import sys
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from skunk.dsl import LookupBranch, ParseError, RetrieveBranch, parse, validate

    try:
        plan = parse(text)
    except ParseError as e:
        return {"parse_ok": False, "validate_ok": False, "ops": [],
                "head": None, "tail": None, "problems": [f"parse_error:{e}"]}

    v = validate(plan)
    ops: list[str] = []
    for b in plan.branches:
        if isinstance(b, RetrieveBranch):
            ops.extend(["retrieve", "extract"])
        elif isinstance(b, LookupBranch):
            ops.append("lookup_external")
    ops.append("compute")

    # head/tail are now structural facts; report them for back-compat with the CSV.
    head = "retrieve" if plan.branches and isinstance(plan.branches[0], RetrieveBranch) \
        else "lookup_external" if plan.branches else None
    tail = "compute"

    problems = [f"validate:{e}" for e in v.errors]
    return {
        "parse_ok": True,
        "validate_ok": v.ok,
        "ops": ops,
        "head": head,
        "tail": tail,
        "problems": problems,
    }


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
