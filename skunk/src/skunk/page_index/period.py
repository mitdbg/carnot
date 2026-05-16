"""Parse a DSL period string into a (start_iso, end_iso) interval, or
into a list of intervals for enumerations.

Period grammar (from DSL.md):
  point       ::= "CY"YYYY | "FY"YYYY | "Q"[1-4]"-"YYYY
                | YYYY"-"MM"-"DD | YYYY"-"MM | YYYY
  range       ::= point ".." point            (inclusive)
  enumeration ::= point ("," point)+
"""

from __future__ import annotations

import calendar
import re


_POINT_RE = re.compile(
    r"^(?:"
    r"CY(?P<cy>\d{4})"
    r"|FY(?P<fy>\d{4})"
    r"|Q(?P<q>[1-4])-(?P<qy>\d{4})"
    r"|(?P<ymd>\d{4}-\d{2}-\d{2})"
    r"|(?P<ym>\d{4}-\d{2})"
    r"|(?P<y>\d{4})"
    r")$"
)


def _last_day(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def parse_point(p: str) -> tuple[str, str]:
    """Return (start_iso, end_iso) for a single point in the period grammar."""
    m = _POINT_RE.match(p)
    if not m:
        raise ValueError(f"unrecognized period point: {p!r}")
    if m.group("cy"):
        y = int(m.group("cy"))
        return f"{y:04d}-01-01", f"{y:04d}-12-31"
    if m.group("fy"):
        y = int(m.group("fy"))
        # Pre-1977: FY ends June 30 (FYy = Jul (y-1) .. Jun y).
        # 1977 onward: FY ends Sep 30 (FYy = Oct (y-1) .. Sep y).
        # The bulletin corpus spans both; pick the convention by year.
        if y < 1977:
            return f"{y - 1:04d}-07-01", f"{y:04d}-06-30"
        return f"{y - 1:04d}-10-01", f"{y:04d}-09-30"
    if m.group("q"):
        q = int(m.group("q"))
        y = int(m.group("qy"))
        starts = {1: (1, 1), 2: (4, 1), 3: (7, 1), 4: (10, 1)}
        ends = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
        sm, sd = starts[q]
        em, ed = ends[q]
        return f"{y:04d}-{sm:02d}-{sd:02d}", f"{y:04d}-{em:02d}-{ed:02d}"
    if m.group("ymd"):
        s = m.group("ymd")
        return s, s
    if m.group("ym"):
        s = m.group("ym")
        y, mo = int(s[:4]), int(s[5:7])
        return f"{y:04d}-{mo:02d}-01", f"{y:04d}-{mo:02d}-{_last_day(y, mo):02d}"
    y = int(m.group("y"))
    return f"{y:04d}-01-01", f"{y:04d}-12-31"


def period_to_intervals(period: str) -> list[tuple[str, str]]:
    """Parse a period string into a list of (start_iso, end_iso) intervals.

    - A point          → one interval
    - A range a..b     → one interval spanning a.start to b.end
    - An enumeration   → one interval per comma-separated point
    """
    if ".." in period:
        a, b = period.split("..", 1)
        a_start, _ = parse_point(a)
        _, b_end = parse_point(b)
        if a_start > b_end:
            raise ValueError(f"range start > end: {period!r}")
        return [(a_start, b_end)]
    if "," in period:
        return [parse_point(p.strip()) for p in period.split(",") if p.strip()]
    return [parse_point(period)]


def intervals_overlap(a_start: str, a_end: str,
                      b_start: str, b_end: str) -> bool:
    return not (a_end < b_start or a_start > b_end)


# ---------------------------------------------------------------------------
# Verbatim-date parser (for catalog rows' `dates` field).
# ---------------------------------------------------------------------------
#
# The `dates` field on a PageCatalogRow holds VERBATIM date strings as they
# appear on the page — extracted by extract_fields.py at build time. They
# come in five recurring shapes:
#
#   "1934"                       bare year
#   "January 1939", "Jan. 1939"  month-year
#   "June 30, 1938"              month-day-year
#   "Fiscal Year 1939"           FY (apply the FYxxxx rule)
#   "1932-1939", "1932–1939"     year range (hyphen, en-dash, or em-dash)
#
# Anything else parses to []. Callers can therefore treat the parser as a
# best-effort signal and fall back to other heuristics when it returns nothing.

_MONTHS_FULL = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)
_MONTHS_ABBR = (
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "sept", "oct", "nov", "dec",
)
_MONTH_TO_NUM: dict[str, int] = {
    m: i + 1 for i, m in enumerate(_MONTHS_FULL)
}
# Abbreviation indices are aligned with month positions (1..12) except
# `sept` which is an extra spelling of September.
_MONTH_TO_NUM.update({
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
})

# Order matters: try most specific first.
_MONTH_DAY_YEAR_RE = re.compile(
    r"^(?P<mon>[A-Za-z]+)\.?\s+(?P<day>\d{1,2}),?\s+(?P<year>\d{4})$",
)
_MONTH_YEAR_RE = re.compile(
    r"^(?P<mon>[A-Za-z]+)\.?\s+(?P<year>\d{4})$",
)
_FY_RE = re.compile(
    r"^(?:fiscal\s+year|fy)\s+(?P<year>\d{4})$",
    re.IGNORECASE,
)
_YEAR_RANGE_RE = re.compile(
    r"^(?P<a>\d{4})\s*[\-–—]\s*(?P<b>\d{4})$",
)
_BARE_YEAR_RE = re.compile(r"^(\d{4})$")


def verbatim_date_to_intervals(s: str) -> list[tuple[str, str]]:
    """Best-effort parse of a verbatim date string into ISO intervals.

    Returns `[]` for inputs we can't recognize; callers should treat that
    as "no signal" rather than "no overlap".
    """
    if not s:
        return []
    t = s.strip().lower().replace(",", "")
    # Strip trailing period after month abbreviation, repeated whitespace.
    t = re.sub(r"\s+", " ", t).strip()

    m = _FY_RE.match(t)
    if m:
        try:
            return [parse_point(f"FY{int(m.group('year'))}")]
        except ValueError:
            return []
    m = _MONTH_DAY_YEAR_RE.match(t)
    if m:
        mn = _MONTH_TO_NUM.get(m.group("mon").rstrip("."))
        if mn is None:
            return []
        y = int(m.group("year"))
        d = int(m.group("day"))
        if 1 <= d <= _last_day(y, mn):
            iso = f"{y:04d}-{mn:02d}-{d:02d}"
            return [(iso, iso)]
        return []
    m = _MONTH_YEAR_RE.match(t)
    if m:
        mn = _MONTH_TO_NUM.get(m.group("mon").rstrip("."))
        if mn is None:
            return []
        y = int(m.group("year"))
        return [(f"{y:04d}-{mn:02d}-01",
                 f"{y:04d}-{mn:02d}-{_last_day(y, mn):02d}")]
    m = _YEAR_RANGE_RE.match(t)
    if m:
        a, b = int(m.group("a")), int(m.group("b"))
        if a > b:
            a, b = b, a
        return [(f"{a:04d}-01-01", f"{b:04d}-12-31")]
    m = _BARE_YEAR_RE.match(t)
    if m:
        y = int(m.group(1))
        return [(f"{y:04d}-01-01", f"{y:04d}-12-31")]
    return []
