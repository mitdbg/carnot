"""Lookup-agent tools — concrete `Tool` subclasses (the shared `Tool` ABC lives
in `multi_turn_agent`). `resolve_lookup_tools()` returns a list of active tools
specified by a `LookupAgentConfig`, otherwise it returns all registered tools.

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

from skunk.agents.multi_turn_agent import Tool
from skunk.common import _RateLimiter, get_rate_limiter
from skunk.config import LookupAgentConfig

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
    doc = """\
### fetch_fred(endpoint, params={})
FRED API. `endpoint` is a FRED path,
`params` its query args; auth + `file_type=json` are injected; returns parsed JSON.
Pull a whole series in one call via `frequency`/`aggregation_method`; don't loop
single observations.
```python
resp = fetch_fred("series/observations",
                    {"series_id": "PSAVERT", "observation_start": "1959-01-01",
                    "observation_end": "1990-12-31",
                    "frequency": "a", "aggregation_method": "avg"})
obs = [(o["date"], float(o["value"])) for o in resp["observations"] if o["value"] != "."]
hits = fetch_fred("series/search", {"search_text": "personal saving rate"})  # find a series_id
```
**Endpoints** you'll use most: `series/observations` (the data values) and
`series/search` (find a `series_id` from words). Others: `series` (one series'
metadata), `category/series`, `release/series`, `tags/series`.

**`series/observations` params:**
- `series_id` (required).
- `observation_start` / `observation_end`: `YYYY-MM-DD` (defaults: full history).
- `frequency`: aggregate to a LOWER frequency — `d`, `w`, `bw`, `m`, `q`, `sa`, `a`
    (default: the series' native frequency; you cannot up-sample above native, e.g. a
    monthly series can't be requested as `d`).
- `aggregation_method`: `avg` (default), `sum`, `eop` (end of period) — only takes
    effect together with `frequency`.
- `units` (value transform): `lin` (levels, default), `chg`, `ch1` (chg from yr ago),
    `pch` (% chg), `pc1` (% chg from yr ago), `pca`, `cch`, `cca`, `log`.
- `sort_order`: `asc` (default) / `desc`. `limit`: 1–100000 (default 100000); `offset`.
- `output_type`: `1` latest-revised (default), `4` initial-release-only; or pass
    `vintage_dates` (comma-sep `YYYY-MM-DD`) to get data as it stood on those dates.
- Returns `{"observations": [{"date","value", ...}, ...]}`; a `value` of `"."` is
    missing — skip it.

**`series/search` params:** `search_text` (required); `search_type` `full_text`
(default) or `series_id`; `limit`; `order_by` (e.g. `search_rank`, `popularity`,
`observation_start`); narrow with `filter_variable`+`filter_value` or `tag_names`.
Returns `{"seriess": [{"id","title","frequency","units","seasonal_adjustment",
"observation_start","observation_end","popularity"}, ...]}` — pick the `id`, then
read it with `series/observations`.

FX/rate spot series (EXUSUK, EXCAUS, DEX*) from ~1971; CPI (CPIAUCSL SA, CPIAUCNS
NSA) from 1947. Pre-API history returns nothing."""

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


class BlsTool(Tool):
    name = "fetch_bls"
    doc = """\
### fetch_bls(payload)
BLS Public Data API v2 `timeseries/data` endpoint. `payload` is the POST body;
your `registrationkey` is injected for you; returns parsed JSON.
```python
resp = fetch_bls({"seriesid": ["CUUR0000SA0"], "startyear": "1960",
                    "endyear": "1962", "annualaverage": True})
data = resp["Results"]["series"][0]["data"]   # [{year, period, periodName, value}, ...]
```
**Payload fields:**
- `seriesid`: list of BLS series IDs (the only required field). IDs are uppercase and
    may contain `_ - #` but no lowercase/special chars.
- `startyear` / `endyear`: 4-digit year strings; span ≤20 years per request.
- `annualaverage`: `True` to include the annual average, returned as a row with
    `period == "M13"` (`periodName == "Annual"`).
- `calculations`: `True` adds a `calculations` object per data point with
    `net_changes` and `pct_changes` over 1/3/6/12-period windows.
- `catalog`: `True` adds series metadata (where licensed); `aspects`: `True` adds
    data aspects. `registrationkey` is handled for you (don't set it).

**Response:** `{"status","responseTime","message",
"Results": {"series": [{"seriesID","data": [...]}]}}`. Each `data` entry is
`{"year","period","periodName","value","footnotes": [...]}` (+ `calculations` if
requested). Monthly `period` is `"M01"`–`"M12"`; quarterly `"Q01"`–`"Q04"`; annual
average `"M13"`. `value` is a string — cast it. `status != "REQUEST_SUCCEEDED"`
means the query failed; read `message` for why.

**Per-request caps:** ≤50 series and ≤20 years per call — split a wider span across
calls. CPI-U: `CUUR0000SA0` (NSA) / `CUSR0000SA0` (SA)."""

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


class WorldBankTool(Tool):
    name = "fetch_world_bank"
    doc = """\
### fetch_world_bank(path, params={})
World Bank Indicators API v2 (no key required). `path` follows
`/v2/`, `params` its query args; `format=json` is injected; returns a
`[metadata, [observations]]` list.
```python
resp = fetch_world_bank("country/USA/indicator/NY.GDP.MKTP.CD",
                        {"date": "2003:2012", "per_page": 100})
meta, rows = resp           # rows: [{"date": "2012", "value": 1.6e13, ...}, ...]
```
**Path:** `country/{code}/indicator/{indicator}`. `{code}` is an ISO-3 country code
(or 2-letter), `all`, or several joined by `;` (e.g. `USA;GBR;JPN`). Multiple
indicators likewise join with `;`. Aggregate regions have their own codes (e.g.
`WLD` world, `EUU` EU, `OED` OECD).

**Params:** `date` a year or `start:end` range (e.g. `"2003:2012"`); `per_page`
(default 50, max 32500 — set high to avoid paging) and `page`; `mrv=N` most-recent N
values; `mrnev=N` most-recent N non-empty; `gapfill=Y`; `frequency` `Y`/`Q`/`M`;
`source` to pin a specific database.

**Response:** element `[0]` is metadata (`page`, `pages`, `per_page`, `total`); check
`pages > 1`. Element `[1]` is the observations list, each
`{"indicator": {"id","value"}, "country": {"id","value"}, "countryiso3code",
"date","value","unit","obs_status","decimal"}`. `value` is a number or `null`
(missing); newest year first. Annual data from ~1960. Common indicators:
`NY.GDP.MKTP.CD` (GDP current US$), `NY.GDP.MKTP.CN` (GDP current LCU),
`SP.POP.TOTL` (population), `NY.GDP.PCAP.CD` (GDP per capita US$)."""

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


class TavilySearchTool(Tool):
    name = "tavily_search"
    doc = """\
### tavily_search(query, **kwargs)
Tavily web search.
`query` + kwargs; auth is injected and `include_answer` is forced off (its synthesized
answer can hallucinate numbers — don't rely on it). Returns JSON. Results are short
snippets only (`content`) — the exact figure is often NOT in the snippet, so fetch_url
a promising hit's `url` for the full page text.
```python
resp = tavily_search("annual average GBP USD exchange rate 1941",
                        include_domains=["measuringworth.com"])
for h in resp["results"]:
    print(h["url"], "—", h["content"], h["score"])
page = fetch_url(resp["results"][0]["url"])   # full text of the best hit
```
**Useful kwargs:**
- `max_results`: 0–20 (default 10 here). `search_depth`: `basic` (default) or
    `advanced` (deeper, costs more) — use `advanced` for hard/obscure figures.
- `include_domains` / `exclude_domains`: lists of domains to whitelist/blacklist —
    the strongest lever for steering to an authoritative source.
- `topic`: `general` (default), `news`, or `finance`. `time_range`: `day`/`week`/
    `month`/`year` (or `d`/`w`/`m`/`y`); or `start_date`/`end_date` as `YYYY-MM-DD`.
- `include_raw_content=True` returns each hit's cleaned full text inline (`raw_content`),
    which can save a follow-up `fetch_url`. `chunks_per_source` (1–3, advanced only).

**Response:** `{"query","results": [...], "response_time", ...}`. Each result is
`{"title","url","content" (snippet),"score" (relevance 0–1),"raw_content" (if
requested)}`. Results are ranked by `score`; prefer the top hit from a trusted domain."""

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


class FetchUrlTool(Tool):
    name = "fetch_url"
    doc = """\
### fetch_url(url, max_chars=50000)
Fetch a URL → cleaned visible page text (HTML stripped, nav/scripts/footers dropped,
prefers `<main>`/`<article>`). Truncated to `max_chars` (default 50000); raise it for
long pages. Soft-fails: on any error returns a `"[fetch_url error: ...]"` string
instead of raising, so check for that marker before parsing.
```python
text = fetch_url("https://example.com/historical-rates")
```
Read a specific page found via tavily_search when you need more than the snippet.
Returns plain text only — no JS-rendered content and no tables-as-structure."""

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


# all available lookup tools, keyed by the tool name
_REGISTRY: dict[str, Tool] = {
    t.name: t for t in (FredTool(), BlsTool(), WorldBankTool(), TavilySearchTool(), FetchUrlTool())
}


def resolve_lookup_tools(config: LookupAgentConfig) -> list[Tool]:
    """Returns the active lookup tools. Tool names in `config.lookup_tools` take precedence if specified,
    otherwise it returns all registered tools."""
    names = config.lookup_tools
    if not names:
        return list(_REGISTRY.values())

    unknown = [n for n in names if n not in _REGISTRY]
    if unknown:
        raise ValueError(f"Unknown lookup tool(s) {unknown!r}; valid names: {sorted(_REGISTRY)}")

    return [_REGISTRY[n] for n in names]
