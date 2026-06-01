"""Lookup-agent tools — concrete `Tool` subclasses (the shared `Tool` ABC lives
in `multi_turn_agent`). Each co-locates its API (`__call__`, what the model
invokes) and its prompt documentation (`doc`). `resolve_lookup_tools` selects the
active list from an explicit override or `SkunkConfig.lookup_tools` (else all
registered tools); the lookup agent pairs that list with a free-text
prioritization string (`DEFAULT_PRIORITIZATION`) when none is supplied.

Shared HTTP/date helpers live here; each tool body keeps its own
`get_rate_limiter(...)` call so the process-wide rate caps still apply. Tools
fetch from one source and return a primitive (or snippets), raising
`RuntimeError` on failure — except `fetch_url`, which soft-fails with a
`"[fetch_url error: ...]"` marker."""

from __future__ import annotations

import calendar
import json
import os
import re
import ssl
import time
import urllib.parse
import urllib.request
from datetime import date as _date, timedelta
from typing import TYPE_CHECKING, Any

from skunk.common import _RateLimiter, get_rate_limiter
from skunk.multi_turn_agent import Tool

if TYPE_CHECKING:
    from skunk.config import SkunkConfig


try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()


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


def _fetch_json(url: str, limiter: _RateLimiter, timeout: int = 15, retries: int = 4) -> Any:
    """GET `url`, parse JSON, retrying HTTP 429 / 5xx with exponential backoff.
    Acquires one `limiter` slot before each request (incl. retries) so the whole
    process stays under the source's per-key rate cap."""
    for attempt in range(retries):
        limiter.acquire()
        try:
            with urllib.request.urlopen(url, timeout=timeout, context=_SSL_CTX) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(0.5 * (2 ** attempt))
                continue
            raise


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


# High-signal historical-data sites; override via `include_domains=[...]`.
_DEFAULT_INCLUDE_DOMAINS = [
    "fred.stlouisfed.org", "measuringworth.com", "macrotrends.net",
    "bls.gov", "imf.org", "data.worldbank.org",
    "federalreserve.gov", "treasurydirect.gov", "stats.bis.org",
    "bankofengland.co.uk", "ecb.europa.eu", "boj.or.jp",
]

# Tags that never carry primary content — dropped before extracting visible text.
_DROP_TAGS = {
    "script", "style", "noscript", "nav", "header", "footer",
    "aside", "iframe", "svg", "form", "button", "menu",
}


class FredTool(Tool):
    name = "fetch_fred"

    def __call__(self, series_id: str, date_str: str) -> float:
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
            end = f"{year:04d}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"  # [1] = last day of month
        else:
            d = _date(year, month, day)
            start = (d - timedelta(days=5)).isoformat()
            end = (d + timedelta(days=5)).isoformat()
        url = "https://api.stlouisfed.org/fred/series/observations?" + urllib.parse.urlencode({
            "series_id": series_id, "observation_start": start,
            "observation_end": end, "sort_order": "asc",
            "api_key": api_key, "file_type": "json",
        })
        values = [(o["date"], float(o["value"]))
                  for o in _fetch_json(url, timeout=10, limiter=get_rate_limiter("fred")).get("observations", [])
                  if o.get("value", ".") != "."]
        if not values:
            raise RuntimeError(f"FRED returned no observations for {series_id} {date_str}")
        if kind == "YMD":
            target = _date(year, month, day)
            return min(values, key=lambda dv: abs((_date.fromisoformat(dv[0]) - target).days))[1]
        return sum(v for _, v in values) / len(values)

    doc = """\
### fetch_fred(series_id, date)
FRED API. `date`: YYYY (annual mean), YYYY-MM, YYYY-MM-DD.
```python
val = fetch_fred("CPIAUCSL", "1953")     # annual mean of CPI in 1953
val = fetch_fred("DGS10", "2020-03-15")  # 10y yield near date
```"""


class BlsTool(Tool):
    name = "fetch_bls"

    def __call__(self, series_id: str, date_str: str) -> float:
        """BLS public REST API. `date_str` is YYYY or YYYY-MM(-DD).
        Y returns M13 if present, else mean of M01..M12."""
        kind, year, month, _ = _parse_date(date_str)
        params = {"startyear": str(year), "endyear": str(year)}
        if (key := os.environ.get("BLS_API_KEY", "")):
            params["registrationkey"] = key
        url = (f"https://api.bls.gov/publicAPI/v2/timeseries/data/{series_id}?"
               + urllib.parse.urlencode(params))
        payload = _fetch_json(url, timeout=15, limiter=get_rate_limiter("bls"))
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

    doc = """\
### fetch_bls(series_id, date)
BLS API. Same date format.
```python
val = fetch_bls("CUUR0000SA0", "1953")   # CPI-U 1953 annual
```"""


class WorldBankTool(Tool):
    name = "fetch_world_bank"

    def __call__(self, country_iso3: str, indicator: str, year_str: str) -> float:
        """World Bank Indicators API. Annual only. Common indicators:
        `NY.GDP.MKTP.CN`, `NY.GDP.MKTP.CD`, `SP.POP.TOTL`, `NY.GDP.PCAP.CD`."""
        kind, year, _, _ = _parse_date(year_str)
        if kind != "Y":
            raise RuntimeError(f"fetch_world_bank: annual only; got {year_str!r}")
        url = (f"https://api.worldbank.org/v2/country/{country_iso3}/indicator/{indicator}"
               "?" + urllib.parse.urlencode({"date": str(year), "format": "json"}))
        payload = _fetch_json(url, timeout=15, limiter=get_rate_limiter("world_bank"))
        if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
            raise RuntimeError(f"World Bank returned no observations for {country_iso3} {indicator} {year}")
        value = payload[1][0].get("value")
        if value is None:
            raise RuntimeError(f"World Bank: null value for {country_iso3} {indicator} {year}")
        return float(value)

    doc = """\
### fetch_world_bank(iso3, indicator, year)
Annual indicators.
```python
val = fetch_world_bank("DEU", "NY.GDP.MKTP.CD", "1996")  # Germany nominal GDP
```"""


class TavilySearchTool(Tool):
    name = "tavily_search"

    def __call__(
        self,
        query: str,
        max_results: int = 10,
        include_domains: list[str] | None = None,
    ) -> list[dict]:
        """Tavily web search → `[{title, url, content, score}, ...]`. `include_domains`
        defaults to curated historical-data sites (pass `[]` for whole-web). Snippets
        only — `include_answer` synthesis is disabled (it can hallucinate numbers)."""
        if include_domains is None:
            include_domains = _DEFAULT_INCLUDE_DOMAINS
        kwargs: dict[str, Any] = {
            "query": query, "max_results": max_results, "search_depth": "basic",
        }
        if include_domains:
            kwargs["include_domains"] = include_domains
        get_rate_limiter("tavily").acquire()
        return _get_tavily().search(**kwargs).get("results", [])

    doc = """\
### tavily_search(query, max_results=10, include_domains=None)
Web search, restricted by default to authoritative historical-data
sites (FRED, BLS, IMF, World Bank, central banks, MeasuringWorth, ...).
Returns `[{title, url, content, score}, ...]`. The `content` snippet
often carries the number directly. Pass `include_domains=[]` to
search the whole web.
```python
hits = tavily_search("annual average GBP USD exchange rate 1941")
for h in hits:
    print(h["url"], "—", h["content"])
```"""


class FetchUrlTool(Tool):
    name = "fetch_url"

    def __call__(self, url: str, max_chars: int = 50000) -> str:
        """Fetch a URL → cleaned page text (BeautifulSoup, boilerplate dropped, prefers
        <main>/<article>). Soft-fails on errors with `"[fetch_url error: ...]"`."""
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 skunk"})
        try:
            with urllib.request.urlopen(req, timeout=20, context=_SSL_CTX) as r:
                body = r.read().decode("utf-8", errors="replace")
        except Exception as e:
            return f"[fetch_url error: {type(e).__name__}: {e} on {url}]"
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(body, "html.parser")
        for tag in soup.find_all(_DROP_TAGS):
            tag.decompose()
        container = soup.find("main") or soup.find("article") or soup.body or soup
        text = container.get_text(" ", strip=True)
        return re.sub(r"\s+", " ", text)[:max_chars]

    doc = """\
### fetch_url(url)
Fetch and return cleaned page text. Soft-fails on errors.
```python
text = fetch_url("https://example.com/historical-rates")
```"""


# All available lookup tools, keyed by the name the model calls. `final_answer`
# is NOT here — it is the always-injected loop terminator, not a pluggable tool.
_REGISTRY: dict[str, Tool] = {
    t.name: t for t in (FredTool(), BlsTool(), WorldBankTool(), TavilySearchTool(), FetchUrlTool())
}

# Default prioritization guidance (was the inline paragraph in the agent prompt).
DEFAULT_PRIORITIZATION = """\
When you see a number that plausibly answers the target in some tool
output, your VERY NEXT block should be final_answer. Many historical
lookups have no single canonical precision — multiple sources may
report slightly different values. Pick the first plausible hit and
commit. Searching for confirmation is the dominant failure mode."""


def resolve_lookup_tools(config: "SkunkConfig", explicit: list[Tool] | None = None) -> list[Tool]:
    """Pick the active lookup tools: an `explicit` list wins; else the names in
    `config.lookup_tools`; else all registered tools."""
    if explicit is not None:
        return explicit
    names = config.lookup_tools
    if not names:
        return list(_REGISTRY.values())
    unknown = [n for n in names if n not in _REGISTRY]
    if unknown:
        raise ValueError(f"Unknown lookup tool(s) {unknown!r}; valid names: {sorted(_REGISTRY)}")
    return [_REGISTRY[n] for n in names]
