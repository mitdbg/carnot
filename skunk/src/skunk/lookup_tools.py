"""Lookup-agent tools — concrete `Tool` subclasses (the shared `Tool` ABC lives
in `multi_turn_agent`). Each co-locates its API (`__call__`, what the model
invokes) and its prompt documentation (`doc`). `resolve_lookup_tools` selects the
active list from an explicit override or `SkunkConfig.lookup_tools` (else all
registered tools); the `DEFAULT_PRIORITIZATION` guidance below is folded
statically into the lookup agent's `briefing`.

Each tool is a **thin wrapper** over one external API: it adds only
authentication and shared transport (a per-source `get_rate_limiter(...)` cap +
retry/backoff in `_fetch_json`) and returns the API's parsed JSON for the agent
to navigate — the agent chooses endpoints, params, and response handling. The
exception is `fetch_url`, a fetch-and-clean utility (no upstream API) that
soft-fails with a `"[fetch_url error: ...]"` marker."""

from __future__ import annotations

import json
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from skunk.common import _RateLimiter, get_rate_limiter
from skunk.config import SkunkConfig
from skunk.multi_turn_agent import Tool


try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()


def _fetch_json(url: str, limiter: _RateLimiter, timeout: int = 15, retries: int = 4,
                max_backoff: float = 30.0, data: dict | None = None) -> Any:
    """Fetch `url`, parse JSON, retrying HTTP 429 / 5xx with exponential backoff.
    GET by default; if `data` is given it is sent as a JSON POST body. Acquires one
    `limiter` slot before each request (incl. retries) so the whole process stays
    under the source's per-key rate cap.

    Backoff honors a `Retry-After` header when the server sends one; otherwise it
    is exponential (2, 4, 8, ... s), capped at `max_backoff`. The longer waits
    matter for sources like FRED whose 429s reflect a rolling-window throttle that
    a sub-second retry only re-triggers — riding it out is the only thing that
    clears it."""
    body = json.dumps(data).encode() if data is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    for attempt in range(retries):
        limiter.acquire()
        try:
            req = urllib.request.Request(url, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                retry_after = e.headers.get("Retry-After") if e.headers else None
                wait = float(retry_after) if (retry_after and retry_after.isdigit()) else 2.0 * (2 ** attempt)
                time.sleep(min(wait, max_backoff))
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


# Tags that never carry primary content — dropped before extracting visible text.
_DROP_TAGS = {
    "script", "style", "noscript", "nav", "header", "footer",
    "aside", "iframe", "svg", "form", "button", "menu",
}


class FredTool(Tool):
    name = "fetch_fred"

    def __call__(self, endpoint: str, params: dict | None = None) -> dict:
        """Thin wrapper over the **full FRED API**
        (https://fred.stlouisfed.org/docs/api/fred/). `endpoint` is any FRED path
        (e.g. "series/observations", "series/search"); `params` are that endpoint's
        query args. The tool only injects authentication (`api_key`) + `file_type=json`
        and returns the parsed JSON — you choose the endpoint, the params, and parse
        the response yourself."""
        api_key = os.environ.get("FRED_API_KEY", "")
        if not api_key:
            raise RuntimeError("FRED_API_KEY not set")
        query = {**(params or {}), "api_key": api_key, "file_type": "json"}
        url = "https://api.stlouisfed.org/fred/" + endpoint.strip("/") + "?" + urllib.parse.urlencode(query)
        return _fetch_json(url, timeout=15, limiter=get_rate_limiter("fred"))

    doc = """\
### fetch_fred(endpoint, params={})
FRED API (https://fred.stlouisfed.org/docs/api/fred/). `endpoint` is a FRED path,
`params` its query args; returns parsed JSON. Pull a whole series in one call via
`frequency`/`aggregation_method`; don't loop single observations.
```python
resp = fetch_fred("series/observations",
                  {"series_id": "PSAVERT", "observation_start": "1959-01-01",
                   "observation_end": "1990-12-31",
                   "frequency": "a", "aggregation_method": "avg"})
obs = [(o["date"], float(o["value"])) for o in resp["observations"] if o["value"] != "."]
hits = fetch_fred("series/search", {"search_text": "personal saving rate"})  # find a series_id
```
FX/rate spot series (EXUSUK, EXCAUS, DEX*) from ~1971; CPI (CPIAUCSL SA, CPIAUCNS
NSA) from 1947. Pre-API history returns nothing — use tavily_search / fetch_url."""


class BlsTool(Tool):
    name = "fetch_bls"

    def __call__(self, payload: dict) -> dict:
        """Thin wrapper over the BLS Public Data API v2 `timeseries/data` endpoint
        (https://www.bls.gov/developers/api_signature_v2.htm). `payload` is the POST
        body (e.g. {"seriesid": ["CUUR0000SA0"], "startyear": "1960",
        "endyear": "1962", "annualaverage": true, "calculations": true}). The tool
        injects your registrationkey (if BLS_API_KEY is set) and returns the parsed
        JSON — you choose the body fields and parse the response."""
        body = dict(payload)
        if (key := os.environ.get("BLS_API_KEY", "")):
            body.setdefault("registrationkey", key)
        return _fetch_json("https://api.bls.gov/publicAPI/v2/timeseries/data/",
                           timeout=15, limiter=get_rate_limiter("bls"), data=body)

    doc = """\
### fetch_bls(payload)
BLS Public Data API v2 timeseries/data (https://www.bls.gov/developers/api_signature_v2.htm).
`payload` is the POST body; returns parsed JSON.
```python
resp = fetch_bls({"seriesid": ["CUUR0000SA0"], "startyear": "1960",
                  "endyear": "1962", "annualaverage": True})
data = resp["Results"]["series"][0]["data"]   # [{year, period, periodName, value}, ...]
```
≤20 years and ≤50 series per request (10/25 without a key). CPI-U: CUUR0000SA0
(NSA) / CUSR0000SA0 (SA); annual average is period "M13" (set annualaverage=True)."""


class WorldBankTool(Tool):
    name = "fetch_world_bank"

    def __call__(self, path: str, params: dict | None = None) -> Any:
        """Thin wrapper over the World Bank Indicators API v2
        (https://datahelpdesk.worldbank.org/knowledgebase/articles/889392).
        `path` is the API path after `/v2/`, e.g.
        "country/USA/indicator/NY.GDP.MKTP.CD"; `params` are query args
        (date="2003:2012", per_page=100, ...). No key required; the tool adds
        format=json and returns the parsed JSON — you parse it."""
        query = {**(params or {}), "format": "json"}
        url = "https://api.worldbank.org/v2/" + path.strip("/") + "?" + urllib.parse.urlencode(query)
        return _fetch_json(url, timeout=15, limiter=get_rate_limiter("world_bank"))

    doc = """\
### fetch_world_bank(path, params={})
World Bank Indicators API v2 (https://datahelpdesk.worldbank.org/knowledgebase/articles/889392).
`path` follows `/v2/`, `params` its query args; returns a [metadata, [observations]] list.
```python
resp = fetch_world_bank("country/USA/indicator/NY.GDP.MKTP.CD",
                        {"date": "2003:2012", "per_page": 100})
rows = resp[1]   # [{"date": "2012", "value": 16253970000000.0, ...}, ...]
```
Annual data from ~1960; country via ISO-3 codes. Common: NY.GDP.MKTP.CD,
NY.GDP.MKTP.CN, SP.POP.TOTL, NY.GDP.PCAP.CD."""


class TavilySearchTool(Tool):
    name = "tavily_search"

    def __call__(self, query: str, **kwargs: Any) -> dict:
        """Thin wrapper over the Tavily Search API
        (https://docs.tavily.com/api-reference/endpoint/search). `query` plus any
        kwargs (max_results, search_depth, topic, time_range, include_domains,
        exclude_domains, ...) pass straight through to the Tavily client; auth
        (TAVILY_API_KEY) is injected at the client. `include_answer=False` is forced
        (answer synthesis can hallucinate numbers). Returns the Tavily JSON, whose
        results are short snippets only — fetch_url a promising hit for full text."""
        kwargs.pop("include_answer", None)        # forced off below; not the model's to set
        kwargs.setdefault("max_results", 10)      # prefer breadth; the model may override
        get_rate_limiter("tavily").acquire()
        return _get_tavily().search(query, include_answer=False, **kwargs)

    doc = """\
### tavily_search(query, **kwargs)
Tavily web search (https://docs.tavily.com/api-reference/endpoint/search). `query`
+ kwargs (max_results [default 10], search_depth, time_range, include_domains,
exclude_domains, ...); returns JSON. Results are short snippets only (`content`) —
the exact figure is often NOT in the snippet, so fetch_url a promising hit's `url`
for the full page text (where data tables live).
```python
resp = tavily_search("annual average GBP USD exchange rate 1941",
                     include_domains=["measuringworth.com"])
for h in resp["results"]:
    print(h["url"], "—", h["content"])
page = fetch_url(resp["results"][0]["url"])   # full text of the best hit
```
For values the structured APIs lack — pre-1971 FX, pre-API series, one-off figures."""


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
Fetch a URL → cleaned page text. Soft-fails on errors.
```python
text = fetch_url("https://example.com/historical-rates")
```
Read a specific page found via tavily_search when you need more than the snippet."""


# All available lookup tools, keyed by the name the model calls. The final answer
# is NOT here — it is emitted as a ```json``` block and parsed outside the sandbox,
# not a pluggable tool.
_REGISTRY: dict[str, Tool] = {
    t.name: t for t in (FredTool(), BlsTool(), WorldBankTool(), TavilySearchTool(), FetchUrlTool())
}

# Default prioritization guidance (was the inline paragraph in the agent prompt).
DEFAULT_PRIORITIZATION = """\
If `src` is pinned, the value must come from that publisher: keep searching until you
read it from that named source, even if another source offers a close number first.
Never substitute a different publisher's figure for a pinned source — same-named series
from BLS/FRED and a pinned publisher can differ, and that difference is the point.
If `src` is null, commit the first plausible hit: once a tool output contains a number
that answers the target, your next block is your final-answer ```json``` block. Historical
values vary slightly across sources — don't keep searching for confirmation.
When a search returns the right-looking URL but not the full content on it, `fetch_url`
that URL and read the value off the page rather than re-searching."""


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
