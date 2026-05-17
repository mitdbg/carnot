"""lookup_external operator — Gemini with Google Search; FRED/BLS via emitted Python.

Two output modes. MODE 2 is a fenced Python block that calls
`fetch_fred_series` or `fetch_bls_series` (injected into the sandbox); used
when the request names FRED or BLS. MODE 1 is a 3-line literal: value,
unit, publisher-token; used for every other named publisher (Macrotrends,
Bank of England, Bloomberg, …) and for unsourced lookups. The publisher
token on line 3 is validated against `resp.grounding_titles` (the resolved
domain Gemini reports for each grounding chunk).
"""

from __future__ import annotations

import json
import os
import re
import ssl
import urllib.parse
import urllib.request
from datetime import date as _date, timedelta

import certifi

from skunk.common import HarnessContext
from skunk.plan import AnnotatedValue
from skunk.executor import SkunkExecutor
from skunk.operator import OpNode, StepFailed, parse_llm_value, strip_code_fences

# ---------------------------------------------------------------------------
# Static system prompt block
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a precise data assistant for economic indicators, FX rates,
historical dates, and named entities. Your training covers through
mid-2025 — treat any date before July 2025 as historical; never refuse
on grounds of "future."

## Source routing

Identify the publisher named in the request (usually after "from X",
"according to X", "published by X"). Data labels like "BLS CPI-U" or
"FRED series IRLTLT01GBM156N" describe what the value is; they are NOT
the publisher unless the request literally says "from FRED" / "from BLS".

  Publisher is FRED         → MODE 2, fetch_fred_series.
  Publisher is BLS          → MODE 2, fetch_bls_series.
    (also "Bureau of Labor Statistics")
  Any other publisher       → MODE 1 (Google Search).
    Examples: Federal Reserve Bank of Minneapolis, Macrotrends, Bank of
    England, Bloomberg, Yahoo Finance, World Bank. Do NOT substitute the
    FRED/BLS API even if they hold the same data — the question named a
    specific site; go to that site.
  No publisher named        → MODE 2 if a clean FRED/BLS series fits,
                              otherwise MODE 1 with line 3 = "none".

## MODE 1 — Google Search

Procedure (mandatory):
  (a) Compose a Google Search query that includes the publisher (use a
      `site:` filter or the publisher name, e.g.
      `site:macrotrends.net USD JPY 2025-03-31`).
  (b) Issue the search via your Google Search tool. Answering from
      training memory is rejected — your response is accepted only if
      the search returned a grounding chunk from the publisher's site.
  (c) Read the snippets and extract the value at the publisher's full
      printed precision.
  (d) Emit exactly three lines — no labels, no JSON, no prose, no markdown:
        Line 1: Python literal at the publisher's full printed precision.
                Use float for numeric, "double-quoted" for strings, a list
                for multiple values in request order.
        Line 2: snake_case unit token describing what the value
                represents. Examples seen here: `year`, `cpi`, `fx_rate`,
                `usd_millions`, `usd_billions`, `pct`, `count`, `rate`,
                `text`. Invent a similar token when the request is for
                something else.
        Line 3: bare publisher token (e.g. `macrotrends`,
                `minneapolisfed`, `bankofengland`, `bloomberg`,
                `yahoofinance`), or "none" if no publisher was named.

Example:
  149.918
  fx_rate
  macrotrends

## MODE 2 — FRED / BLS REST API

Output ONLY one fenced Python block (no prose). Assign `value` and `unit`.
The helpers return floats at maximum API precision; do not round.

`fetch_fred_series(series_id, date)` — St. Louis Fed FRED REST API.
  Accepts any valid FRED series ID. Date forms: 'YYYY' (annual mean) |
  'YYYY-MM' (first of month) | 'YYYY-MM-DD' (that day).
  Examples used here: DEXJPUS, DEXUSUK, DEXUSEU (daily FX); CPIAUCNS
  (CPI-U monthly NSA); DGS10 (US 10y Treasury daily); IRLTLT01GBM156N
  (UK 10y Gilt monthly); IRLTLT01JPM156N (Japan 10y monthly).

`fetch_bls_series(series_id, date)` — Bureau of Labor Statistics REST API.
  Accepts any valid BLS series ID; same date semantics as FRED.
  Examples used here: CUUR0000SA0 (CPI-U All Items NSA);
  CUUR0000SA0L1E (Core CPI); CES0000000001 (Total nonfarm payrolls).

Example:
```python
value = fetch_fred_series("DEXJPUS", "2025-03-31")
unit = "fx_rate"
```

## Precision and other rules

- Maximum precision the publisher reports or the API returns. Never
  round, format, or simplify — that's the final output agent's job.
  If the source prints 26.766, return 26.766; if 26.8, return 26.8.
- FX rates are positive floats.
- Named-entity answers: double quotes on line 1, unit `text`.
- Never return float('nan'), None, or non-literal expressions.
- "unknown" is acceptable only in MODE 1 when search returned grounding
  chunks but the page lacked the value. Line 3 must still name the
  source you searched.
"""

class LookupExternalExecutor(SkunkExecutor):
    name: str = "lookup_external"
    system_prompt: str = _SYSTEM_PROMPT

_SSL_CTX = ssl.create_default_context(cafile=certifi.where())

_PY_FENCE_RE = re.compile(r"```python\b[^\n]*\n(.*?)\n```", re.DOTALL)

def _fetch_fred(series_id: str, date_str: str) -> float:
    """Fetch a value from the FRED REST API. Raises on any failure."""
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        raise RuntimeError("FRED_API_KEY not set")

    if len(date_str) == 4:
        start, end = f"{date_str}-01-01", f"{date_str}-12-31"
    elif len(date_str) == 7:
        start = end = f"{date_str}-01"
    else:
        try:
            d = _date.fromisoformat(date_str)
            start, end = date_str, str(d + timedelta(days=4))
        except ValueError:
            start = end = date_str

    url = ("https://api.stlouisfed.org/fred/series/observations?" +
           urllib.parse.urlencode({"series_id": series_id, "observation_start": start,
                                   "observation_end": end, "sort_order": "asc",
                                   "api_key": api_key, "file_type": "json"}))
    with urllib.request.urlopen(url, timeout=8, context=_SSL_CTX) as r:
        observations = [
            float(o["value"])
            for o in json.loads(r.read()).get("observations", [])
            if o.get("value", ".") != "."
        ]
    if not observations:
        raise RuntimeError(f"FRED returned no observations for {series_id} {date_str}")
    return sum(observations) / len(observations) if len(date_str) == 4 else observations[0]

def _fetch_bls(series_id: str, date_str: str) -> float:
    """Fetch a value from the BLS public REST API. Raises on any failure.

    Date forms mirror _fetch_fred:
      - 'YYYY'        → BLS's M13 annual entry if present, else mean of M01..M12
      - 'YYYY-MM'     → that month's value
      - 'YYYY-MM-DD' → that month's value (BLS is monthly; day ignored)
    """
    try:
        if len(date_str) == 4:
            year = int(date_str)
            month = None
        else:
            year = int(date_str[:4])
            month = int(date_str[5:7])
    except ValueError as e:
        raise RuntimeError(f"BLS bad date {date_str!r}: {e}") from e

    params = {"startyear": str(year), "endyear": str(year)}
    api_key = os.environ.get("BLS_API_KEY", "")
    if api_key:
        params["registrationkey"] = api_key
    url = (f"https://api.bls.gov/publicAPI/v2/timeseries/data/{series_id}?" +
           urllib.parse.urlencode(params))
    with urllib.request.urlopen(url, timeout=10, context=_SSL_CTX) as r:
        payload = json.loads(r.read())
    if payload.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS error: {payload.get('status')} / {payload.get('message')}")
    series = payload.get("Results", {}).get("series", [])
    if not series or not series[0].get("data"):
        raise RuntimeError(f"BLS returned no data for {series_id} {date_str}")
    data = series[0]["data"]

    if month is None:
        for d in data:
            if d["period"] == "M13":
                return float(d["value"])
        monthly = [float(d["value"]) for d in data
                   if d["period"].startswith("M") and d["period"] != "M13"]
        if not monthly:
            raise RuntimeError(f"BLS no monthly data for {series_id} {year}")
        return sum(monthly) / len(monthly)

    target = f"M{month:02d}"
    for d in data:
        if d["period"] == target:
            return float(d["value"])
    raise RuntimeError(f"BLS no observation for {series_id} {year}-{target}")

def run(op: OpNode, prev: None, ctx: HarnessContext) -> list[AnnotatedValue]:
    target = op.args.get("target", "")
    if not target:
        raise StepFailed("lookup_external", "Missing 'target' arg")
    src = op.args.get("src") or None

    # If the planner provided a `src` hint, prepend a one-line directive so the
    # operator's source-routing logic biases toward it.
    user_msg = target if not src else f"Preferred source: {src}.\n{target}"

    ctx.emit("lookup_external", "calling gemini", target=target, src=src)
    resp = ctx.llm_client.call(
        LookupExternalExecutor().assemble_system_prompt(ctx),
        user_msg, thinking_budget=-1, use_google_search=True, ctx=ctx,
    )
    raw = resp.text or ""
    ctx.emit("lookup_external", "gemini response",
             raw=raw[:500],
             grounding_titles=resp.grounding_titles[:8])

    m = _PY_FENCE_RE.search(raw)
    if m:
        code = strip_code_fences(m.group(0))
        env: dict = {
            "fetch_fred_series": _fetch_fred,
            "fetch_bls_series": _fetch_bls,
        }
        try:
            exec(compile(code, "<lookup_external>", "exec"), env)  # noqa: S102
        except Exception as e:
            raise StepFailed(
                "lookup_external",
                f"API code execution failed: {type(e).__name__}: {e}\nCode:\n{code[:400]}",
            ) from e
        if "value" not in env or "unit" not in env:
            raise StepFailed(
                "lookup_external",
                f"API code did not set both `value` and `unit`:\n{code[:400]}",
            )
        value, unit = env["value"], env["unit"]
        ctx.emit("lookup_external", "api executed",
                 value=repr(value)[:200], unit=unit)
        return [AnnotatedValue(description=target, value=value, unit=unit)]

    try:
        value, unit = parse_llm_value(raw)
    except ValueError as e:
        raise StepFailed(
            "lookup_external",
            f"Cannot parse response: {e}\nRaw: {raw[:200]}",
        ) from e

    ctx.emit("lookup_external", "parsed direct",
             value=repr(value)[:200], unit=unit)
    return [AnnotatedValue(description=target, value=value, unit=unit)]
