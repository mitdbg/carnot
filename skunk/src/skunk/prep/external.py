"""Tier 4 — external data cache + live-fetch interface.

Single interface for all external lookups:
  result = fetch(resource, cache_dir=..., cache_only=False, **kwargs)

Supported resources:
  - cpi_u        : BLS CPI-U (U.S. City Average, 1982-84=100)
  - fx_macrotrends: Macrotrends USD/XXX monthly and daily rates

Cache files (CSV):
  cache/external/cpi_u.csv
  cache/external/fx_macrotrends.csv

Usage:
    python -m skunk.prep.external \
        --resource cpi_u \
        --cache-dir cache
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fetch(resource: str, cache_dir: str, cache_only: bool = False, **kwargs: Any) -> Any:
    """Fetch data from cache; on miss, fetch live (unless cache_only=True).

    Returns a pandas DataFrame or a scalar, depending on the resource.
    """
    cache_path = Path(cache_dir) / "external"
    cache_path.mkdir(parents=True, exist_ok=True)

    resource_lower = resource.lower()

    if "cpi" in resource_lower:
        return _fetch_cpi_u(cache_path, cache_only)
    if any(k in resource_lower for k in ("fx", "macrotrends", "rate")):
        return _fetch_fx_macrotrends(cache_path, cache_only, **kwargs)
    if "bls" in resource_lower:
        return _fetch_bls(cache_path, cache_only, **kwargs)

    raise ValueError(f"Unknown external resource: {resource!r}. "
                     f"Supported: cpi_u, fx_macrotrends, bls")


# ---------------------------------------------------------------------------
# CPI-U (BLS)
# ---------------------------------------------------------------------------

_CPI_CSV = "cpi_u.csv"

def _fetch_cpi_u(cache_path: Path, cache_only: bool) -> Any:
    import pandas as pd
    csv = cache_path / _CPI_CSV
    if csv.exists():
        return pd.read_csv(csv)
    if cache_only:
        raise RuntimeError("CPI-U cache not available and cache_only=True")
    df = _download_cpi_u()
    df.to_csv(csv, index=False)
    print(f"[external] CPI-U cached to {csv}")
    return df


def _download_cpi_u() -> Any:
    """Download CPI-U from BLS public API (series CUUR0000SA0)."""
    import urllib.request
    import json
    import pandas as pd

    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    payload = json.dumps({
        "seriesid": ["CUUR0000SA0"],
        "startyear": "1913",
        "endyear": "2025",
    }).encode()
    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    records = []
    for series in data.get("Results", {}).get("series", []):
        for item in series.get("data", []):
            records.append({
                "year": int(item["year"]),
                "period": item["period"],     # M01-M12, M13=annual avg
                "period_name": item["periodName"],
                "value": float(item["value"]),
            })

    df = pd.DataFrame(records)
    return df.sort_values(["year", "period"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# FX rates (Macrotrends scrape)
# ---------------------------------------------------------------------------

_FX_CSV = "fx_macrotrends.csv"

def _fetch_fx_macrotrends(cache_path: Path, cache_only: bool, **kwargs: Any) -> Any:
    import pandas as pd
    csv = cache_path / _FX_CSV
    if csv.exists():
        return pd.read_csv(csv)
    if cache_only:
        raise RuntimeError("FX cache not available and cache_only=True")
    pair = kwargs.get("pair", "USD/JPY")
    df = _download_fx(pair)
    df.to_csv(csv, index=False)
    print(f"[external] FX rates cached to {csv}")
    return df


def _download_fx(pair: str = "USD/JPY") -> Any:
    """Download FX historical data. Uses an open FX API as fallback."""
    import urllib.request
    import pandas as pd
    from datetime import datetime, timedelta

    # Simple FRED API for major pairs (free, no key needed for some)
    # Example: USD/JPY → DEXJPUS on FRED
    _FRED_SERIES = {
        "USD/JPY": "DEXJPUS",
        "USD/CAD": "DEXCAUS",
        "USD/EUR": "DEXUSEU",
        "USD/GBP": "DEXUSUK",
    }
    series_id = _FRED_SERIES.get(pair.upper())

    if series_id:
        url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv"
               f"?id={series_id}&vintage_date=2025-12-31")
        with urllib.request.urlopen(url, timeout=30) as resp:
            import io
            df = pd.read_csv(io.StringIO(resp.read().decode()))
        df.columns = ["date", "rate"]
        df["pair"] = pair
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        return df.sort_values("date").reset_index(drop=True)

    # Fallback: exchangerate.host (free public API)
    today = datetime.today().strftime("%Y-%m-%d")
    base, quote = pair.split("/")
    url = f"https://api.exchangerate.host/timeseries?start_date=1999-01-01&end_date={today}&base={base}&symbols={quote}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read())

    records = []
    for date_str, rates in data.get("rates", {}).items():
        rate = rates.get(quote)
        if rate is not None:
            records.append({"date": date_str, "pair": pair, "rate": float(rate)})

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# BLS generic
# ---------------------------------------------------------------------------

def _fetch_bls(cache_path: Path, cache_only: bool, **kwargs: Any) -> Any:
    import pandas as pd
    series = kwargs.get("series", "CUUR0000SA0")
    csv = cache_path / f"bls_{series}.csv"
    if csv.exists():
        return pd.read_csv(csv)
    if cache_only:
        raise RuntimeError(f"BLS cache for {series} not available and cache_only=True")
    df = _download_cpi_u()  # reuse for CPI variants; extend as needed
    df.to_csv(csv, index=False)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource", choices=["cpi_u", "fx_macrotrends", "all"], default="all")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--pair", default="USD/JPY", help="FX pair, e.g. USD/JPY")
    args = parser.parse_args()

    cache_path = Path(args.cache_dir) / "external"
    cache_path.mkdir(parents=True, exist_ok=True)

    if args.resource in ("cpi_u", "all"):
        print("[external] Downloading CPI-U...")
        _fetch_cpi_u(cache_path, cache_only=False)

    if args.resource in ("fx_macrotrends", "all"):
        print(f"[external] Downloading FX ({args.pair})...")
        _fetch_fx_macrotrends(cache_path, cache_only=False, pair=args.pair)
