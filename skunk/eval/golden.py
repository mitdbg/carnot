"""Parse `source_docs` URLs in officeqa_pro.csv into golden PageRefs.

The benchmark CSV's `source_docs` column contains URLs of the form:
  https://fraser.stlouisfed.org/title/treasury-bulletin-407/january-1941-6529?page=15

The ?page=N query parameter is the **bulletin printed page number** (canonical),
not the PDF page index. GoldenPage.page therefore matches PageRef.page directly.

For each question, we extract the (year, month, page) tuple from the URL.
A question may have multiple URLs separated by whitespace/newlines.

Verified: 100% of the 133 questions in officeqa_pro.csv have at least one ?page=N.

"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Map of url month-name → "MM"
_MONTH_MAP = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}

_URL_RE = re.compile(
    r"/(?P<month>january|february|march|april|may|june|july|august|september|october|november|december)"
    r"-(?P<year>\d{4})[^?]*\?page=(?P<page>\d+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GoldenPage:
    year: int
    month: str    # "MM"
    page: int

    @property
    def bulletin_id(self) -> str:
        return f"{self.year:04d}-{self.month}"


def parse_source_docs(source_docs: str) -> list[GoldenPage]:
    """Extract every (year, month, page) tuple from a source_docs cell."""
    out: list[GoldenPage] = []
    if not isinstance(source_docs, str):
        return out
    for m in _URL_RE.finditer(source_docs):
        out.append(GoldenPage(
            year=int(m.group("year")),
            month=_MONTH_MAP[m.group("month").lower()],
            page=int(m.group("page")),
        ))
    return out


def load_golden(csv_path: str | Path) -> dict[str, list[GoldenPage]]:
    """Map uid → list[GoldenPage] for every question."""
    df = pd.read_csv(csv_path)
    return {row["uid"]: parse_source_docs(str(row.get("source_docs", "")))
            for _, row in df.iterrows()}
