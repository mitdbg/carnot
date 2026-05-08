"""
Measure retrieval drift in the OfficeQA benchmark.

For each question, asks an LLM to identify which bulletin a date-only lookup
would return (ignoring publication lag), then compares to the golden
source_docs. Reports month-offset statistics.
"""

import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from google import genai

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "data" / "officeqa_pro.csv"
ENV_PATH = REPO_ROOT / ".env"
CACHE_PATH = REPO_ROOT / "data" / "drift_cache.json"
OUTPUT_PATH = REPO_ROOT / "data" / "drift_analysis.csv"


def _load_env(path: Path) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ (no-op if missing)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

_AVAILABLE_BULLETINS = """\
treasury_bulletin_1939_01 treasury_bulletin_1939_02 treasury_bulletin_1939_03 treasury_bulletin_1939_04 treasury_bulletin_1939_05 treasury_bulletin_1939_06 treasury_bulletin_1939_07 treasury_bulletin_1939_08 treasury_bulletin_1939_09 treasury_bulletin_1939_10 treasury_bulletin_1939_11 treasury_bulletin_1939_12
treasury_bulletin_1940_01 treasury_bulletin_1940_02 treasury_bulletin_1940_03 treasury_bulletin_1940_04 treasury_bulletin_1940_05 treasury_bulletin_1940_06 treasury_bulletin_1940_07 treasury_bulletin_1940_08 treasury_bulletin_1940_09 treasury_bulletin_1940_10 treasury_bulletin_1940_11 treasury_bulletin_1940_12
treasury_bulletin_1941_01 treasury_bulletin_1941_02 treasury_bulletin_1941_03 treasury_bulletin_1941_04 treasury_bulletin_1941_05 treasury_bulletin_1941_06 treasury_bulletin_1941_07 treasury_bulletin_1941_08 treasury_bulletin_1941_09 treasury_bulletin_1941_10 treasury_bulletin_1941_11 treasury_bulletin_1941_12
treasury_bulletin_1942_01 treasury_bulletin_1942_02 treasury_bulletin_1942_03 treasury_bulletin_1942_04 treasury_bulletin_1942_05 treasury_bulletin_1942_06 treasury_bulletin_1942_07 treasury_bulletin_1942_08 treasury_bulletin_1942_09 treasury_bulletin_1942_10 treasury_bulletin_1942_11 treasury_bulletin_1942_12
treasury_bulletin_1943_01 treasury_bulletin_1943_02 treasury_bulletin_1943_03 treasury_bulletin_1943_04 treasury_bulletin_1943_05 treasury_bulletin_1943_06 treasury_bulletin_1943_07 treasury_bulletin_1943_08 treasury_bulletin_1943_09 treasury_bulletin_1943_10 treasury_bulletin_1943_11 treasury_bulletin_1943_12
treasury_bulletin_1944_01 treasury_bulletin_1944_02 treasury_bulletin_1944_03 treasury_bulletin_1944_04 treasury_bulletin_1944_05 treasury_bulletin_1944_06 treasury_bulletin_1944_08 treasury_bulletin_1944_09 treasury_bulletin_1944_10 treasury_bulletin_1944_11 treasury_bulletin_1944_12
treasury_bulletin_1945_01 treasury_bulletin_1945_02 treasury_bulletin_1945_03 treasury_bulletin_1945_04 treasury_bulletin_1945_05 treasury_bulletin_1945_06 treasury_bulletin_1945_07 treasury_bulletin_1945_08 treasury_bulletin_1945_09 treasury_bulletin_1945_10 treasury_bulletin_1945_11 treasury_bulletin_1945_12
treasury_bulletin_1946_01 treasury_bulletin_1946_02 treasury_bulletin_1946_03 treasury_bulletin_1946_04 treasury_bulletin_1946_05 treasury_bulletin_1946_06 treasury_bulletin_1946_07 treasury_bulletin_1946_08 treasury_bulletin_1946_09 treasury_bulletin_1946_10 treasury_bulletin_1946_11 treasury_bulletin_1946_12
treasury_bulletin_1947_01 treasury_bulletin_1947_02 treasury_bulletin_1947_03 treasury_bulletin_1947_04 treasury_bulletin_1947_05 treasury_bulletin_1947_06 treasury_bulletin_1947_07 treasury_bulletin_1947_08 treasury_bulletin_1947_09 treasury_bulletin_1947_10 treasury_bulletin_1947_11 treasury_bulletin_1947_12
treasury_bulletin_1948_01 treasury_bulletin_1948_02 treasury_bulletin_1948_03 treasury_bulletin_1948_04 treasury_bulletin_1948_05 treasury_bulletin_1948_06 treasury_bulletin_1948_07 treasury_bulletin_1948_08 treasury_bulletin_1948_09 treasury_bulletin_1948_10 treasury_bulletin_1948_11 treasury_bulletin_1948_12
treasury_bulletin_1949_01 treasury_bulletin_1949_02 treasury_bulletin_1949_03 treasury_bulletin_1949_04 treasury_bulletin_1949_05 treasury_bulletin_1949_06 treasury_bulletin_1949_07 treasury_bulletin_1949_08 treasury_bulletin_1949_09 treasury_bulletin_1949_10 treasury_bulletin_1949_11 treasury_bulletin_1949_12
treasury_bulletin_1950_01 treasury_bulletin_1950_02 treasury_bulletin_1950_03 treasury_bulletin_1950_04 treasury_bulletin_1950_05 treasury_bulletin_1950_06 treasury_bulletin_1950_07 treasury_bulletin_1950_08 treasury_bulletin_1950_09 treasury_bulletin_1950_10 treasury_bulletin_1950_11 treasury_bulletin_1950_12
treasury_bulletin_1951_01 treasury_bulletin_1951_02 treasury_bulletin_1951_03 treasury_bulletin_1951_04 treasury_bulletin_1951_05 treasury_bulletin_1951_06 treasury_bulletin_1951_07 treasury_bulletin_1951_08 treasury_bulletin_1951_09 treasury_bulletin_1951_10 treasury_bulletin_1951_11 treasury_bulletin_1951_12
treasury_bulletin_1952_01 treasury_bulletin_1952_02 treasury_bulletin_1952_03 treasury_bulletin_1952_04 treasury_bulletin_1952_05 treasury_bulletin_1952_06 treasury_bulletin_1952_07 treasury_bulletin_1952_08 treasury_bulletin_1952_09 treasury_bulletin_1952_10 treasury_bulletin_1952_11 treasury_bulletin_1952_12
treasury_bulletin_1953_01 treasury_bulletin_1953_02 treasury_bulletin_1953_03 treasury_bulletin_1953_04 treasury_bulletin_1953_05 treasury_bulletin_1953_06 treasury_bulletin_1953_07 treasury_bulletin_1953_08 treasury_bulletin_1953_09 treasury_bulletin_1953_10 treasury_bulletin_1953_11 treasury_bulletin_1953_12
treasury_bulletin_1954_01 treasury_bulletin_1954_02 treasury_bulletin_1954_03 treasury_bulletin_1954_04 treasury_bulletin_1954_05 treasury_bulletin_1954_06 treasury_bulletin_1954_07 treasury_bulletin_1954_08 treasury_bulletin_1954_09 treasury_bulletin_1954_10 treasury_bulletin_1954_11 treasury_bulletin_1954_12
treasury_bulletin_1955_01 treasury_bulletin_1955_02 treasury_bulletin_1955_03 treasury_bulletin_1955_04 treasury_bulletin_1955_05 treasury_bulletin_1955_06 treasury_bulletin_1955_07 treasury_bulletin_1955_08 treasury_bulletin_1955_09 treasury_bulletin_1955_10 treasury_bulletin_1955_11 treasury_bulletin_1955_12
treasury_bulletin_1956_01 treasury_bulletin_1956_02 treasury_bulletin_1956_03 treasury_bulletin_1956_04 treasury_bulletin_1956_05 treasury_bulletin_1956_06 treasury_bulletin_1956_07 treasury_bulletin_1956_08 treasury_bulletin_1956_09 treasury_bulletin_1956_10 treasury_bulletin_1956_11 treasury_bulletin_1956_12
treasury_bulletin_1957_01 treasury_bulletin_1957_02 treasury_bulletin_1957_03 treasury_bulletin_1957_04 treasury_bulletin_1957_05 treasury_bulletin_1957_06 treasury_bulletin_1957_07 treasury_bulletin_1957_08 treasury_bulletin_1957_09 treasury_bulletin_1957_10 treasury_bulletin_1957_11 treasury_bulletin_1957_12
treasury_bulletin_1958_01 treasury_bulletin_1958_02 treasury_bulletin_1958_03 treasury_bulletin_1958_04 treasury_bulletin_1958_05 treasury_bulletin_1958_06 treasury_bulletin_1958_07 treasury_bulletin_1958_08 treasury_bulletin_1958_09 treasury_bulletin_1958_10 treasury_bulletin_1958_11 treasury_bulletin_1958_12
treasury_bulletin_1959_01 treasury_bulletin_1959_02 treasury_bulletin_1959_03 treasury_bulletin_1959_04 treasury_bulletin_1959_05 treasury_bulletin_1959_06 treasury_bulletin_1959_07 treasury_bulletin_1959_08 treasury_bulletin_1959_09 treasury_bulletin_1959_10 treasury_bulletin_1959_11 treasury_bulletin_1959_12
treasury_bulletin_1960_01 treasury_bulletin_1960_02 treasury_bulletin_1960_03 treasury_bulletin_1960_04 treasury_bulletin_1960_05 treasury_bulletin_1960_06 treasury_bulletin_1960_07 treasury_bulletin_1960_08 treasury_bulletin_1960_09 treasury_bulletin_1960_10 treasury_bulletin_1960_11 treasury_bulletin_1960_12
treasury_bulletin_1961_01 treasury_bulletin_1961_02 treasury_bulletin_1961_03 treasury_bulletin_1961_04 treasury_bulletin_1961_05 treasury_bulletin_1961_06 treasury_bulletin_1961_07 treasury_bulletin_1961_08 treasury_bulletin_1961_09 treasury_bulletin_1961_10 treasury_bulletin_1961_11 treasury_bulletin_1961_12
treasury_bulletin_1962_01 treasury_bulletin_1962_02 treasury_bulletin_1962_03 treasury_bulletin_1962_04 treasury_bulletin_1962_05 treasury_bulletin_1962_06 treasury_bulletin_1962_07 treasury_bulletin_1962_08 treasury_bulletin_1962_09 treasury_bulletin_1962_10 treasury_bulletin_1962_11 treasury_bulletin_1962_12
treasury_bulletin_1963_01 treasury_bulletin_1963_02 treasury_bulletin_1963_03 treasury_bulletin_1963_04 treasury_bulletin_1963_05 treasury_bulletin_1963_06 treasury_bulletin_1963_07 treasury_bulletin_1963_08 treasury_bulletin_1963_09 treasury_bulletin_1963_10 treasury_bulletin_1963_11 treasury_bulletin_1963_12
treasury_bulletin_1964_01 treasury_bulletin_1964_02 treasury_bulletin_1964_03 treasury_bulletin_1964_04 treasury_bulletin_1964_05 treasury_bulletin_1964_06 treasury_bulletin_1964_07 treasury_bulletin_1964_08 treasury_bulletin_1964_09 treasury_bulletin_1964_10 treasury_bulletin_1964_11 treasury_bulletin_1964_12
treasury_bulletin_1965_01 treasury_bulletin_1965_02 treasury_bulletin_1965_03 treasury_bulletin_1965_04 treasury_bulletin_1965_05 treasury_bulletin_1965_06 treasury_bulletin_1965_07 treasury_bulletin_1965_08 treasury_bulletin_1965_09 treasury_bulletin_1965_10 treasury_bulletin_1965_11 treasury_bulletin_1965_12
treasury_bulletin_1966_01 treasury_bulletin_1966_02 treasury_bulletin_1966_03 treasury_bulletin_1966_04 treasury_bulletin_1966_05 treasury_bulletin_1966_06 treasury_bulletin_1966_07 treasury_bulletin_1966_08 treasury_bulletin_1966_09 treasury_bulletin_1966_10 treasury_bulletin_1966_11 treasury_bulletin_1966_12
treasury_bulletin_1967_01 treasury_bulletin_1967_02 treasury_bulletin_1967_03 treasury_bulletin_1967_04 treasury_bulletin_1967_05 treasury_bulletin_1967_06 treasury_bulletin_1967_07 treasury_bulletin_1967_08 treasury_bulletin_1967_09 treasury_bulletin_1967_10 treasury_bulletin_1967_11 treasury_bulletin_1967_12
treasury_bulletin_1968_01 treasury_bulletin_1968_02 treasury_bulletin_1968_03 treasury_bulletin_1968_04 treasury_bulletin_1968_05 treasury_bulletin_1968_06 treasury_bulletin_1968_07 treasury_bulletin_1968_08 treasury_bulletin_1968_09 treasury_bulletin_1968_10 treasury_bulletin_1968_11 treasury_bulletin_1968_12
treasury_bulletin_1969_01 treasury_bulletin_1969_02 treasury_bulletin_1969_03 treasury_bulletin_1969_04 treasury_bulletin_1969_05 treasury_bulletin_1969_06 treasury_bulletin_1969_07 treasury_bulletin_1969_08 treasury_bulletin_1969_09 treasury_bulletin_1969_10 treasury_bulletin_1969_11 treasury_bulletin_1969_12
treasury_bulletin_1970_01 treasury_bulletin_1970_02 treasury_bulletin_1970_03 treasury_bulletin_1970_04 treasury_bulletin_1970_05 treasury_bulletin_1970_06 treasury_bulletin_1970_07 treasury_bulletin_1970_08 treasury_bulletin_1970_09 treasury_bulletin_1970_10 treasury_bulletin_1970_11 treasury_bulletin_1970_12
treasury_bulletin_1971_01 treasury_bulletin_1971_02 treasury_bulletin_1971_03 treasury_bulletin_1971_04 treasury_bulletin_1971_05 treasury_bulletin_1971_06 treasury_bulletin_1971_07 treasury_bulletin_1971_08 treasury_bulletin_1971_09 treasury_bulletin_1971_10 treasury_bulletin_1971_11 treasury_bulletin_1971_12
treasury_bulletin_1972_01 treasury_bulletin_1972_02 treasury_bulletin_1972_03 treasury_bulletin_1972_04 treasury_bulletin_1972_05 treasury_bulletin_1972_06 treasury_bulletin_1972_07 treasury_bulletin_1972_08 treasury_bulletin_1972_09 treasury_bulletin_1972_10 treasury_bulletin_1972_11 treasury_bulletin_1972_12
treasury_bulletin_1973_01 treasury_bulletin_1973_02 treasury_bulletin_1973_03 treasury_bulletin_1973_04 treasury_bulletin_1973_05 treasury_bulletin_1973_06 treasury_bulletin_1973_07 treasury_bulletin_1973_08 treasury_bulletin_1973_09 treasury_bulletin_1973_10 treasury_bulletin_1973_11 treasury_bulletin_1973_12
treasury_bulletin_1974_01 treasury_bulletin_1974_02 treasury_bulletin_1974_03 treasury_bulletin_1974_04 treasury_bulletin_1974_05 treasury_bulletin_1974_06 treasury_bulletin_1974_07 treasury_bulletin_1974_08 treasury_bulletin_1974_09 treasury_bulletin_1974_10 treasury_bulletin_1974_11 treasury_bulletin_1974_12
treasury_bulletin_1975_01 treasury_bulletin_1975_02 treasury_bulletin_1975_03 treasury_bulletin_1975_04 treasury_bulletin_1975_05 treasury_bulletin_1975_06 treasury_bulletin_1975_07 treasury_bulletin_1975_08 treasury_bulletin_1975_09 treasury_bulletin_1975_10 treasury_bulletin_1975_11 treasury_bulletin_1975_12
treasury_bulletin_1976_01 treasury_bulletin_1976_02 treasury_bulletin_1976_03 treasury_bulletin_1976_04 treasury_bulletin_1976_05 treasury_bulletin_1976_06 treasury_bulletin_1976_07 treasury_bulletin_1976_08 treasury_bulletin_1976_09 treasury_bulletin_1976_10 treasury_bulletin_1976_11 treasury_bulletin_1976_12
treasury_bulletin_1977_01 treasury_bulletin_1977_02 treasury_bulletin_1977_03 treasury_bulletin_1977_04 treasury_bulletin_1977_05 treasury_bulletin_1977_06 treasury_bulletin_1977_07 treasury_bulletin_1977_08 treasury_bulletin_1977_09 treasury_bulletin_1977_10 treasury_bulletin_1977_11 treasury_bulletin_1977_12
treasury_bulletin_1978_01 treasury_bulletin_1978_02 treasury_bulletin_1978_03 treasury_bulletin_1978_04 treasury_bulletin_1978_05 treasury_bulletin_1978_06 treasury_bulletin_1978_07 treasury_bulletin_1978_08 treasury_bulletin_1978_09 treasury_bulletin_1978_10 treasury_bulletin_1978_11 treasury_bulletin_1978_12
treasury_bulletin_1979_01 treasury_bulletin_1979_02 treasury_bulletin_1979_03 treasury_bulletin_1979_04 treasury_bulletin_1979_05 treasury_bulletin_1979_06 treasury_bulletin_1979_07 treasury_bulletin_1979_08 treasury_bulletin_1979_09 treasury_bulletin_1979_10 treasury_bulletin_1979_11 treasury_bulletin_1979_12
treasury_bulletin_1980_01 treasury_bulletin_1980_02 treasury_bulletin_1980_03 treasury_bulletin_1980_04 treasury_bulletin_1980_05 treasury_bulletin_1980_06 treasury_bulletin_1980_07 treasury_bulletin_1980_08 treasury_bulletin_1980_09 treasury_bulletin_1980_10 treasury_bulletin_1980_11 treasury_bulletin_1980_12
treasury_bulletin_1981_01 treasury_bulletin_1981_02 treasury_bulletin_1981_03 treasury_bulletin_1981_04 treasury_bulletin_1981_05 treasury_bulletin_1981_06 treasury_bulletin_1981_07 treasury_bulletin_1981_08 treasury_bulletin_1981_09 treasury_bulletin_1981_10 treasury_bulletin_1981_11 treasury_bulletin_1981_12
treasury_bulletin_1982_01 treasury_bulletin_1982_02 treasury_bulletin_1982_03 treasury_bulletin_1982_04 treasury_bulletin_1982_05 treasury_bulletin_1982_06 treasury_bulletin_1982_07 treasury_bulletin_1982_08 treasury_bulletin_1982_09 treasury_bulletin_1982_10 treasury_bulletin_1982_11
treasury_bulletin_1983_03 treasury_bulletin_1983_06 treasury_bulletin_1983_09 treasury_bulletin_1983_12
treasury_bulletin_1984_03 treasury_bulletin_1984_06 treasury_bulletin_1984_09 treasury_bulletin_1984_12
treasury_bulletin_1985_03 treasury_bulletin_1985_06 treasury_bulletin_1985_09 treasury_bulletin_1985_12
treasury_bulletin_1986_03 treasury_bulletin_1986_06 treasury_bulletin_1986_09 treasury_bulletin_1986_12
treasury_bulletin_1987_03 treasury_bulletin_1987_06 treasury_bulletin_1987_09 treasury_bulletin_1987_12
treasury_bulletin_1988_03 treasury_bulletin_1988_06 treasury_bulletin_1988_09 treasury_bulletin_1988_12
treasury_bulletin_1989_03 treasury_bulletin_1989_06 treasury_bulletin_1989_09 treasury_bulletin_1989_12
treasury_bulletin_1990_03 treasury_bulletin_1990_06 treasury_bulletin_1990_09 treasury_bulletin_1990_12
treasury_bulletin_1991_03 treasury_bulletin_1991_06 treasury_bulletin_1991_09 treasury_bulletin_1991_12
treasury_bulletin_1992_03 treasury_bulletin_1992_06 treasury_bulletin_1992_09 treasury_bulletin_1992_12
treasury_bulletin_1993_03 treasury_bulletin_1993_06 treasury_bulletin_1993_09 treasury_bulletin_1993_12
treasury_bulletin_1994_03 treasury_bulletin_1994_06 treasury_bulletin_1994_09 treasury_bulletin_1994_12
treasury_bulletin_1995_03 treasury_bulletin_1995_06 treasury_bulletin_1995_09 treasury_bulletin_1995_12
treasury_bulletin_1996_03 treasury_bulletin_1996_06 treasury_bulletin_1996_09 treasury_bulletin_1996_12
treasury_bulletin_1997_03 treasury_bulletin_1997_06 treasury_bulletin_1997_09 treasury_bulletin_1997_12
treasury_bulletin_1998_03 treasury_bulletin_1998_06 treasury_bulletin_1998_09 treasury_bulletin_1998_12
treasury_bulletin_1999_03 treasury_bulletin_1999_06 treasury_bulletin_1999_09 treasury_bulletin_1999_12
treasury_bulletin_2000_03 treasury_bulletin_2000_06 treasury_bulletin_2000_09 treasury_bulletin_2000_12
treasury_bulletin_2001_03 treasury_bulletin_2001_06 treasury_bulletin_2001_09 treasury_bulletin_2001_12
treasury_bulletin_2002_03 treasury_bulletin_2002_06 treasury_bulletin_2002_09 treasury_bulletin_2002_12
treasury_bulletin_2003_03 treasury_bulletin_2003_06 treasury_bulletin_2003_09 treasury_bulletin_2003_12
treasury_bulletin_2004_03 treasury_bulletin_2004_06 treasury_bulletin_2004_09 treasury_bulletin_2004_12
treasury_bulletin_2005_03 treasury_bulletin_2005_06 treasury_bulletin_2005_09 treasury_bulletin_2005_12
treasury_bulletin_2006_03 treasury_bulletin_2006_06 treasury_bulletin_2006_09 treasury_bulletin_2006_12
treasury_bulletin_2007_03 treasury_bulletin_2007_06 treasury_bulletin_2007_09 treasury_bulletin_2007_12
treasury_bulletin_2008_03 treasury_bulletin_2008_06 treasury_bulletin_2008_09 treasury_bulletin_2008_12
treasury_bulletin_2009_03 treasury_bulletin_2009_06 treasury_bulletin_2009_09 treasury_bulletin_2009_12
treasury_bulletin_2010_03 treasury_bulletin_2010_06 treasury_bulletin_2010_09 treasury_bulletin_2010_12
treasury_bulletin_2011_03 treasury_bulletin_2011_06 treasury_bulletin_2011_09 treasury_bulletin_2011_12
treasury_bulletin_2012_03 treasury_bulletin_2012_06 treasury_bulletin_2012_09 treasury_bulletin_2012_12
treasury_bulletin_2013_03 treasury_bulletin_2013_06 treasury_bulletin_2013_09 treasury_bulletin_2013_12
treasury_bulletin_2014_03 treasury_bulletin_2014_06 treasury_bulletin_2014_09 treasury_bulletin_2014_12
treasury_bulletin_2015_03 treasury_bulletin_2015_06 treasury_bulletin_2015_09 treasury_bulletin_2015_12
treasury_bulletin_2016_03 treasury_bulletin_2016_06 treasury_bulletin_2016_09 treasury_bulletin_2016_12
treasury_bulletin_2017_03 treasury_bulletin_2017_06 treasury_bulletin_2017_09 treasury_bulletin_2017_12
treasury_bulletin_2018_03 treasury_bulletin_2018_06 treasury_bulletin_2018_09 treasury_bulletin_2018_12
treasury_bulletin_2019_03 treasury_bulletin_2019_06 treasury_bulletin_2019_09 treasury_bulletin_2019_12
treasury_bulletin_2020_03 treasury_bulletin_2020_06 treasury_bulletin_2020_09 treasury_bulletin_2020_12
treasury_bulletin_2021_03 treasury_bulletin_2021_06 treasury_bulletin_2021_09 treasury_bulletin_2021_12
treasury_bulletin_2022_03 treasury_bulletin_2022_06 treasury_bulletin_2022_09 treasury_bulletin_2022_12
treasury_bulletin_2023_03 treasury_bulletin_2023_06 treasury_bulletin_2023_09 treasury_bulletin_2023_12
treasury_bulletin_2024_03 treasury_bulletin_2024_06 treasury_bulletin_2024_09 treasury_bulletin_2024_12
treasury_bulletin_2025_03 treasury_bulletin_2025_06 treasury_bulletin_2025_09\
"""

SYSTEM_PROMPT = (
    "You are simulating a NAIVE retrieval agent over U.S. Treasury Bulletin PDFs.\n\n"
    "AVAILABLE BULLETINS (the only files that exist):\n"
    + _AVAILABLE_BULLETINS + "\n\n"
    "Note the publication cadence:\n"
    "  1939-1982: published every month (01-12)\n"
    "  1983-2025: published quarterly only (03, 06, 09, 12)\n"
    "  1944 is missing July (no treasury_bulletin_1944_07).\n\n"
    "Your task: given a question, output the bulletin(s) a date-only lookup would fetch — "
    "i.e., the bulletin(s) that literally cover the date(s) named in the question, "
    "with NO knowledge of publication lag. Assume data appears in "
    "the bulletin for the period it describes.\n\n"
    "RULES:\n"
    "1. Always pick a bulletin that EXISTS in the list above. If the ideal month is not "
    "   published, snap to the nearest available bulletin on or after that date.\n"
    "2. Specific month (e.g. 'September 1953') → treasury_bulletin_1953_09 (exact match).\n"
    "3. Full calendar year (e.g. 'calendar year 1940') → last available bulletin of that "
    "   year (treasury_bulletin_1940_12 if monthly, treasury_bulletin_1940_12 same result).\n"
    "4. Multi-year range where data from EVERY year is needed separately (e.g. '1940 through "
    "   1949 individually') → one bulletin per year, each the last available of that year.\n"
    "5. Multi-year range where only the AGGREGATE over the range is needed → last available "
    "   bulletin of the final year of the range.\n"
    "6. Quarter (e.g. 'third calendar quarter of 1982') → last month of that quarter "
    "   (Q1=03, Q2=06, Q3=09, Q4=12), snapped to an existing bulletin.\n"
    "7. 'As of' a date (e.g. 'as of March 31 2025') → bulletin for that month: "
    "   treasury_bulletin_2025_03.\n"
    "8. Distinct periods that each need a separate lookup → one entry per period.\n"
    "9. A range like '1950 to 1990' where the question needs individual years → list "
    "   one bulletin per year (the last available month of each year).\n\n"
    "Output ONLY valid JSON — no markdown, no preamble:\n"
    '{"naive_bulletins": ["treasury_bulletin_YYYY_MM", ...], "reasoning": "one sentence"}'
)


def extract_naive_bulletins(client: genai.Client, question: str) -> list[str]:
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=question,
                config=genai.types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=4096,
                    temperature=0.0,
                ),
            )
            text = resp.text.strip()
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                raise ValueError(f"no JSON object in response: {text!r}")
            data = json.loads(m.group(0))
            return data["naive_bulletins"]
        except Exception as e:
            if "503" in str(e) and attempt < 2:
                time.sleep(10 * (attempt + 1))
                continue
            raise


def parse_source_files(raw: str) -> list[str]:
    parts = re.split(r"[\r\n,]+", str(raw))
    results = []
    for p in parts:
        p = p.strip()
        m = re.search(r"treasury_bulletin_(\d{4})_(\d{2})", p)
        if m:
            results.append(f"treasury_bulletin_{m.group(1)}_{m.group(2)}")
    return results


def bulletin_to_ym(name: str) -> tuple[int, int] | None:
    m = re.search(r"treasury_bulletin_(\d{4})_(\d{2})", name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def month_offset(naive: str, golden: str) -> int | None:
    n = bulletin_to_ym(naive)
    g = bulletin_to_ym(golden)
    if n is None or g is None:
        return None
    return (g[0] - n[0]) * 12 + (g[1] - n[1])


def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def main():
    _load_env(ENV_PATH)
    df = pd.read_csv(CSV_PATH)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    cache = load_cache()

    rows = []
    for i, row in df.iterrows():
        uid = row["uid"]
        question = row["question"]
        golden_raw = str(row["source_files"])

        if uid not in cache:
            try:
                naive = extract_naive_bulletins(client, question)
                cache[uid] = naive
                save_cache(cache)
                time.sleep(0.15)  # stay well under rate limits
            except Exception as e:
                print(f"  [WARN] {uid} failed: {e}", file=sys.stderr)
                cache[uid] = []
                save_cache(cache)

        naive_bulletins = cache[uid]
        golden_bulletins = parse_source_files(golden_raw)

        if not naive_bulletins or not golden_bulletins:
            continue

        # For each naive bulletin, find the minimum offset to any golden bulletin
        # (pick the closest golden as the intended target)
        offsets = []
        for nb in naive_bulletins:
            local = [month_offset(nb, gb) for gb in golden_bulletins]
            local = [o for o in local if o is not None]
            if local:
                offsets.append(min(local, key=abs))

        if not offsets:
            continue

        # Representative offset: minimum absolute offset across all naive bulletins
        rep_offset = min(offsets, key=abs)

        rows.append({
            "uid": uid,
            "question_snippet": question[:80].replace("\n", " "),
            "naive_bulletins": "|".join(naive_bulletins),
            "golden_bulletins": "|".join(golden_bulletins),
            "offset_months": rep_offset,
        })

        status = "✓" if rep_offset == 0 else f"+{rep_offset}" if rep_offset > 0 else str(rep_offset)
        print(f"  {uid}: naive={naive_bulletins[0]} golden={golden_bulletins[0]} offset={status}")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved per-question table → {OUTPUT_PATH}")

    # ── Statistics ────────────────────────────────────────────────────────────
    offsets = result_df["offset_months"].tolist()
    n = len(offsets)

    print(f"\n{'='*60}")
    print(f"RETRIEVAL DRIFT ANALYSIS  (n={n} questions)")
    print(f"{'='*60}")

    dist = Counter(offsets)
    print(f"\n{'Offset (months)':>16}  {'Count':>6}  {'%':>6}")
    print(f"  {'-'*34}")
    for k in sorted(dist):
        pct = 100 * dist[k] / n
        bar = "█" * int(pct / 2)
        label = f"+{k}" if k > 0 else str(k)
        print(f"  {label:>14}  {dist[k]:>6}  {pct:>5.1f}%  {bar}")

    mean_off = sum(offsets) / n
    sorted_off = sorted(offsets)
    median_off = sorted_off[n // 2]
    exact = dist.get(0, 0)
    late = sum(v for k, v in dist.items() if k > 0)
    early = sum(v for k, v in dist.items() if k < 0)

    print(f"\n  Exact match (offset=0): {exact}/{n}  ({100*exact/n:.1f}%)")
    print(f"  Golden later than naive: {late}/{n}  ({100*late/n:.1f}%)")
    print(f"  Golden earlier than naive: {early}/{n}  ({100*early/n:.1f}%)")
    print(f"  Mean drift: {mean_off:+.2f} months")
    print(f"  Median drift: {median_off:+d} months")

    # ── Examples per bucket ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("EXAMPLES BY OFFSET")
    print(f"{'='*60}")

    buckets: dict[int, list] = defaultdict(list)
    for r in rows:
        buckets[r["offset_months"]].append(r)

    for k in sorted(buckets):
        label = f"+{k}" if k > 0 else str(k)
        examples = buckets[k][:2]
        print(f"\n  offset={label} ({len(buckets[k])} questions)")
        for ex in examples:
            print(f"    [{ex['uid']}] {ex['question_snippet'][:70]}")
            print(f"           naive={ex['naive_bulletins'].split('|')[0]}  "
                  f"golden={ex['golden_bulletins'].split('|')[0]}")


if __name__ == "__main__":
    main()
