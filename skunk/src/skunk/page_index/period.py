"""Parse a DSL period string into year bounds for the chapter-year filter.

Period grammar (from DSL.md):
  point       ::= "CY"YYYY | "FY"YYYY | "Q"[1-4]"-"YYYY
                | YYYY"-"MM"-"DD | YYYY"-"MM | YYYY
  range       ::= point ".." point            (inclusive)
  enumeration ::= point ("," point)+

Only `period_year_window` is part of the public surface; the other
helpers are internal scaffolding.
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


def _parse_point(p: str) -> tuple[str, str]:
    """Return (start_iso, end_iso) for a single period grammar point."""
    m = _POINT_RE.match(p)
    if not m:
        raise ValueError(f"unrecognized period point: {p!r}")
    if m.group("cy"):
        y = int(m.group("cy"))
        return f"{y:04d}-01-01", f"{y:04d}-12-31"
    if m.group("fy"):
        y = int(m.group("fy"))
        # Pre-1977 FY ends June 30 (FYy = Jul (y-1) .. Jun y).
        # 1977 onward FY ends Sep 30 (FYy = Oct (y-1) .. Sep y).
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


def _period_to_intervals(period: str) -> list[tuple[str, str]]:
    """Parse a period string into a list of (start_iso, end_iso) intervals.

    - A point          → one interval
    - A range a..b     → one interval spanning a.start to b.end
    - An enumeration   → one interval per comma-separated point
    """
    if ".." in period:
        a, b = period.split("..", 1)
        a_start, _ = _parse_point(a)
        _, b_end = _parse_point(b)
        if a_start > b_end:
            raise ValueError(f"range start > end: {period!r}")
        return [(a_start, b_end)]
    if "," in period:
        return [_parse_point(p.strip()) for p in period.split(",") if p.strip()]
    return [_parse_point(period)]


def period_year_window(period: str | None) -> tuple[int, int] | None:
    """Reduce a DSL period to its `(min_year, max_year)` envelope.

    Used by the chapter-year retrieve filter to intersect against each
    page's pre-computed year envelope (`PageCatalogRow.min_year` /
    `max_year`). Returns `None` for an unparseable or empty period;
    callers treat that as "no year filter".
    """
    if not period:
        return None
    try:
        intervals = _period_to_intervals(period)
    except (TypeError, ValueError):
        return None
    if not intervals:
        return None
    starts = [int(s[:4]) for s, _ in intervals]
    ends = [int(e[:4]) for _, e in intervals]
    return min(starts), max(ends)
