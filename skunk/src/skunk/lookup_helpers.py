"""Pure data-fetching helpers for `LookupAgent`'s code-execution tools.

Each function fetches from one named source and returns a primitive.
Helpers raise `RuntimeError` on missing data / network failure, except
`fetch_url` which soft-fails with a `"[fetch_url error: ...]"` marker
so loops over `tavily_search` results can skip bad URLs.

`final_answer(value, unit, source)` is the loop terminator —
`LocalPythonExecutor` detects the call by identity and sets
`is_final_answer=True`.
"""

from __future__ import annotations

import json
import os
import re
import ssl
import time
import urllib.parse
import urllib.request
from datetime import date as _date, timedelta
from typing import Any


try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()


# === shared date / HTTP scaffold =====================================

_DATE_YEAR_RE = re.compile(r"^\d{4}$")
_DATE_YEAR_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")


def _parse_date(date_str: str) -> tuple[str, int, int | None, int | None]:
    """Return (kind, year, month, day). kind ∈ {"Y","YM","YMD"}."""
    if _DATE_YEAR_RE.match(date_str):
        return ("Y", int(date_str), None, None)
    if _DATE_YEAR_MONTH_RE.match(date_str):
        return ("YM", int(date_str[:4]), int(date_str[5:7]), None)
    d = _date.fromisoformat(date_str)
    return ("YMD", d.year, d.month, d.day)


def _last_day_of_month(year: int, month: int) -> int:
    if month == 12:
        return 31
    return (_date(year, month + 1, 1) - timedelta(days=1)).day


def _fetch_json(url: str, timeout: int = 15, retries: int = 4) -> Any:
    """GET `url`, parse JSON. On HTTP 429 / 5xx, retry with exponential
    backoff (0.5s, 1s, 2s, 4s) so transient rate limits don't bubble up
    and burn lookup-agent steps."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout, context=_SSL_CTX) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(0.5 * (2 ** attempt))
                continue
            raise


# === fetch_fred ======================================================

def fetch_fred(series_id: str, date_str: str) -> float:
    """FRED REST API. `date_str` is YYYY (annual mean), YYYY-MM
    (monthly mean), or YYYY-MM-DD (closest observation within ±5d)."""
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        raise RuntimeError("FRED_API_KEY not set")
    kind, year, month, day = _parse_date(date_str)
    if kind == "Y":
        start, end = f"{year:04d}-01-01", f"{year:04d}-12-31"
    elif kind == "YM":
        start = f"{year:04d}-{month:02d}-01"
        end = f"{year:04d}-{month:02d}-{_last_day_of_month(year, month):02d}"
    else:
        d = _date(year, month, day)
        start = (d - timedelta(days=5)).isoformat()
        end = (d + timedelta(days=5)).isoformat()
    url = "https://api.stlouisfed.org/fred/series/observations?" + urllib.parse.urlencode({
        "series_id": series_id, "observation_start": start,
        "observation_end": end, "sort_order": "asc",
        "api_key": api_key, "file_type": "json",
    })
    values = [(o["date"], float(o["value"])) for o in _fetch_json(url, timeout=10).get("observations", [])
              if o.get("value", ".") != "."]
    if not values:
        raise RuntimeError(f"FRED returned no observations for {series_id} {date_str}")
    if kind == "YMD":
        target = _date(year, month, day)
        return min(values, key=lambda dv: abs((_date.fromisoformat(dv[0]) - target).days))[1]
    return sum(v for _, v in values) / len(values)


# === fetch_bls =======================================================

def fetch_bls(series_id: str, date_str: str) -> float:
    """BLS public REST API. `date_str` is YYYY or YYYY-MM(-DD).
    Y returns M13 if present, else mean of M01..M12."""
    kind, year, month, _ = _parse_date(date_str)
    params = {"startyear": str(year), "endyear": str(year)}
    if (key := os.environ.get("BLS_API_KEY", "")):
        params["registrationkey"] = key
    url = (f"https://api.bls.gov/publicAPI/v2/timeseries/data/{series_id}?"
           + urllib.parse.urlencode(params))
    payload = _fetch_json(url, timeout=15)
    if payload.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS error: {payload.get('status')} / {payload.get('message')}")
    series = payload.get("Results", {}).get("series", [])
    if not series or not series[0].get("data"):
        raise RuntimeError(f"BLS returned no data for {series_id} {date_str}")
    data = series[0]["data"]
    periods = {d["period"] for d in data}
    if kind == "Y":
        if periods - {f"M{i:02d}" for i in range(1, 14)}:
            raise RuntimeError(f"BLS series {series_id} is not monthly")
        for d in data:
            if d["period"] == "M13":
                return float(d["value"])
        monthly = [float(d["value"]) for d in data if d["period"] != "M13"]
        return sum(monthly) / len(monthly)
    target = f"M{month:02d}"
    for d in data:
        if d["period"] == target:
            return float(d["value"])
    raise RuntimeError(f"BLS series {series_id} has no period {target} for {year}")


# === fetch_world_bank ================================================

def fetch_world_bank(country_iso3: str, indicator: str, year_str: str) -> float:
    """World Bank Indicators API. Annual only. Common indicators:
    `NY.GDP.MKTP.CN`, `NY.GDP.MKTP.CD`, `SP.POP.TOTL`, `NY.GDP.PCAP.CD`."""
    kind, year, _, _ = _parse_date(year_str)
    if kind != "Y":
        raise RuntimeError(f"fetch_world_bank: annual only; got {year_str!r}")
    url = (f"https://api.worldbank.org/v2/country/{country_iso3}/indicator/{indicator}"
           "?" + urllib.parse.urlencode({"date": str(year), "format": "json"}))
    payload = _fetch_json(url, timeout=15)
    if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
        raise RuntimeError(f"World Bank returned no observations for {country_iso3} {indicator} {year}")
    value = payload[1][0].get("value")
    if value is None:
        raise RuntimeError(f"World Bank: null value for {country_iso3} {indicator} {year}")
    return float(value)


# === tavily_search + fetch_url =======================================

_TAVILY_CLIENT: Any = None


def _get_tavily() -> Any:
    global _TAVILY_CLIENT
    if _TAVILY_CLIENT is None:
        from tavily import TavilyClient
        api_key = os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY not set")
        _TAVILY_CLIENT = TavilyClient(api_key=api_key)
    return _TAVILY_CLIENT


def tavily_search(query: str, max_results: int = 5) -> list[dict]:
    """Tavily web search. Returns `[{title, url, content, score}, ...]`.
    Each hit's `content` snippet often carries the value directly;
    otherwise follow up with `fetch_url(url)`."""
    return _get_tavily().search(
        query=query, max_results=max_results, search_depth="basic",
    ).get("results", [])


_TAGS_TO_DROP_RE = re.compile(r"<(script|style|noscript)\b[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def fetch_url(url: str, max_chars: int = 50000) -> str:
    """Fetch a URL and return cleaned page text. Soft-fails on HTTP /
    network errors, returning `"[fetch_url error: ...]"` so loops over
    `tavily_search` results can skip bad URLs."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 skunk"})
    try:
        with urllib.request.urlopen(req, timeout=20, context=_SSL_CTX) as r:
            body = r.read().decode("utf-8", errors="replace")
    except Exception as e:
        return f"[fetch_url error: {type(e).__name__}: {e} on {url}]"
    body = _TAGS_TO_DROP_RE.sub("", body)
    body = _HTML_TAG_RE.sub("", body)
    return re.sub(r"\s+", " ", body).strip()[:max_chars]


# `final_answer` lives in `skunk.multi_turn_agent` and is shared by all
# MultiTurnAgent subclasses; each subclass declares its expected
# payload schema in its system prompt.
