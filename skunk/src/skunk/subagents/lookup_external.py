"""lookup_external subagent — Gemini with Google Search; FRED/BLS via emitted Python.

The model is told which authoritative APIs we have (FRED, BLS), and how to choose:
  - Source named that has an API (FRED, BLS) → emit MODE 2 Python calling that API.
  - Source named without an API (Macrotrends, Bank of England, Bloomberg, …)
    → MODE 1 with a third line declaring the source; we verify the model's search
      actually grounded to that source by checking the response's grounding-chunk
      titles (which Gemini sets to the resolved domain, e.g. "macrotrends.net").
  - No source named → model picks the best path; line 3 is "none"; no verification.

We exec MODE 2 Python in a sandbox with `fetch_fred_series` and `fetch_bls_series`
injected. MODE 1 is the 2/3-line literal contract; line 3 is validated against
`resp.grounding_titles`.
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
from skunk.dsl import AnnotatedValue, OpNode
from skunk.subagents.base import StepFailed, parse_llm_value, strip_code_fences

_SSL_CTX = ssl.create_default_context(cafile=certifi.where())

_SYSTEM = """\
You are a precise data assistant for economic indicators, FX rates, historical
dates, and named entities. Your training covers through mid-2025 — treat any
date before July 2025 as historical, never refuse on grounds of "future."

SOURCE-FOLLOWING — READ FIRST:

The question may name a specific publisher or site. If so, you MUST fetch the
value from that publisher — even if you "know" it from training, even if
another source seems more authoritative, EVEN IF the publisher republishes
data from somewhere else. Never silently substitute.

Routing depends on which publisher is named:

  - "FRED" → MODE 2, fetch_fred_series. FRED is explicitly named.
  - "BLS" or "Bureau of Labor Statistics" → MODE 2, fetch_bls_series.
    BLS is explicitly named.
  - ANY OTHER publisher (Macrotrends, Federal Reserve Bank of Minneapolis,
    Bank of England, Bloomberg, Yahoo Finance, World Bank, …) → MODE 1
    with Google Search. Do NOT use the FRED or BLS API for these, even when
    they republish or mirror the same underlying data. The question named a
    site; you go to that site.
  - No publisher named → MODE 2 if a clean FRED/BLS series obviously fits,
    otherwise MODE 1 with line 3 = "none".

MODE 1 — Google Search with grounding to the named publisher:

You MUST invoke Google Search before responding. Include the publisher in
your query — examples:
  "site:macrotrends.net USD JPY exchange rate 2025-03-31"
  "Federal Reserve Bank of Minneapolis CPI 1953 annual"
  "bankofengland.co.uk official bank rate June 1968"
Do not return "unknown" or any other value unless your search produced at
least one grounding chunk from the publisher's site.

Return exactly THREE lines — no labels, no JSON, no prose, no markdown:
  Line 1: a Python literal AT THE PUBLISHER'S FULL PRINTED PRECISION — do
          NOT round, truncate, or drop trailing zeros. If the page shows
          26.766, return 26.766 (not 26.77 or 26.8). If it shows 14.0,
          return 14.0 (not 14). Use float for numeric, list for multiple,
          "double-quoted" for strings.
  Line 2: unit — one of: year, cpi, fx_rate, usd_millions, usd_billions,
                          pct, count, rate, text
  Line 3: the bare publisher token you searched (e.g. "macrotrends",
          "minneapolisfed", "bankofengland", "bloomberg", "yahoofinance"),
          or "none" if the question did not name a publisher.

MODE 2 — authoritative REST API (FRED or BLS only, only when explicitly named):

Return ONLY a single fenced Python block (no prose). Assign `value` and
`unit`. The helpers return floats at maximum API precision; pass them
through — do not round.

`fetch_fred_series(series_id, date)` — St. Louis Fed FRED REST API.
  Dates: 'YYYY' annual mean | 'YYYY-MM' first of month | 'YYYY-MM-DD' that day.
  Common series:
    DEXJPUS, DEXUSUK, DEXUSEU (daily FX);
    CPIAUCNS (CPI-U monthly NSA);
    DGS10 (US 10y Treasury daily);
    IRLTLT01GBM156N (UK 10y Gilt monthly);
    IRLTLT01JPM156N (Japan 10y monthly).

`fetch_bls_series(series_id, date)` — Bureau of Labor Statistics REST API.
  Same date semantics. Common series:
    CUUR0000SA0 (CPI-U All Items NSA);
    CUUR0000SA0L1E (Core CPI);
    CES0000000001 (Total nonfarm payrolls).

Example MODE 1 (Macrotrends named):
149.918
fx_rate
macrotrends

Example MODE 1 (no source named, named-entity question):
"Financial Management Service"
text
none

Example MODE 2 (FRED named):
```python
value = fetch_fred_series("DEXJPUS", "2025-03-31")
unit = "fx_rate"
```

Example MODE 2 (BLS named, annual CPI):
```python
value = fetch_bls_series("CUUR0000SA0", "1953")
unit = "cpi"
```

PRECISION RULE (applies to BOTH modes):
- Return the value at the maximum precision the publisher reports or the API
  returns. Do not round, format, or simplify. Rounding is the final output
  agent's job, not yours. If the source prints 26.766, return 26.766; if it
  prints 26.8, return 26.8 — match the source exactly.

Rules:
- FX rates are positive floats.
- Multiple dates in MODE 1: line 1 is a Python list in request order.
- Named-entity answers: double quotes on line 1, unit `text`.
- Never return float('nan'), None, or non-literal expressions.
- "unknown" is only acceptable in MODE 1 when your search returned grounding
  chunks but the page did not contain the requested value. Even then, line 3
  must name the source you searched.
"""

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
            year = int(date_str); month = None
        else:
            year = int(date_str[:4]); month = int(date_str[5:7])
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


def _normalize_source_token(s: str) -> str:
    """Lowercase, strip non-alphanumerics — for fuzzy substring match against
    grounding chunk titles. 'Macrotrends' / 'macrotrends.net' both → 'macrotrends'."""
    return re.sub(r"[^a-z0-9]+", "", s.lower())


# Substrings that route to a REST API (MODE 2) rather than search (MODE 1).
# When the NL contains one of these, we don't wrap the user message with a
# search-shaped preamble — let the system prompt's MODE 2 routing kick in.
_API_SOURCE_FRAGMENTS = ("fred", "bls", "bureau of labor")


def _shape_user_message(nl: str) -> str:
    """Wrap the NL in an explicit search directive when no API source is named.

    Without this, Gemini often answers from training memory and skips Google
    Search entirely (grounding_titles empty), which then trips our validator.
    Phrasing the user turn as a search action makes the model use the tool.
    """
    if any(s in nl.lower() for s in _API_SOURCE_FRAGMENTS):
        return nl
    return (
        "Use Google Search to look up the value below. If the request names a "
        "specific publisher or site, use a `site:` filter or include the "
        "publisher name in your query so your results come from that site. "
        "You MUST actually invoke Google Search before responding — even if "
        "you think you know the answer from training — and your response "
        "will be rejected unless it has at least one grounding chunk from "
        "the named publisher.\n\n"
        f"Request: {nl}"
    )


def run(op: OpNode, prev: None, ctx: HarnessContext) -> list[AnnotatedValue]:
    nl = op.args.get("nl", "")
    if not nl:
        raise StepFailed("lookup_external", "Missing 'nl' arg")

    user_msg = _shape_user_message(nl)
    ctx.emit("lookup_external", "calling gemini", nl=nl,
             wrapped=user_msg != nl)
    resp = ctx.llm_client.call(_SYSTEM, user_msg, thinking_budget=-1, use_google_search=True)
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
        return [AnnotatedValue(description=nl, value=value, unit=unit)]

    # MODE 1: 2 or 3 lines. parse_llm_value reads lines 1-2; we read line 3 ourselves.
    try:
        value, unit = parse_llm_value(raw)
    except ValueError as e:
        raise StepFailed(
            "lookup_external",
            f"Cannot parse response: {e}\nRaw: {raw[:200]}",
        ) from e

    lines = [ln.strip() for ln in raw.strip().splitlines()
             if ln.strip() and not ln.strip().startswith("```")]
    declared_source = lines[2] if len(lines) >= 3 else "none"

    # Validate: if the model declared a source, search MUST have been invoked
    # AND at least one chunk title must mention that source. No retry — fail
    # cleanly so the orchestrator can surface it.
    if declared_source.lower() != "none":
        if not resp.grounding_titles:
            raise StepFailed(
                "lookup_external",
                f"Model declared source {declared_source!r} but invoked no "
                f"Google Search (grounding_titles empty). NL: {nl!r}",
            )
        needle = _normalize_source_token(declared_source)
        haystack = [_normalize_source_token(t) for t in resp.grounding_titles]
        if not any(needle and needle in h for h in haystack):
            raise StepFailed(
                "lookup_external",
                f"Model declared source {declared_source!r} but no grounding "
                f"chunk title matched. Grounding titles: {resp.grounding_titles[:5]}",
            )

    ctx.emit("lookup_external", "parsed direct",
             value=repr(value)[:200], unit=unit,
             declared_source=declared_source)
    return [AnnotatedValue(description=nl, value=value, unit=unit)]
