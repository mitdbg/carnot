"""Treasury Bulletin period parser.

Period grammar (from DSL.md):
  point       ::= "CY"YYYY | "FY"YYYY | "Q"[1-4]"-"YYYY
                | YYYY"-"MM"-"DD | YYYY"-"MM | YYYY
  range       ::= point ".." point            (inclusive)
  enumeration ::= point ("," point)+

Treasury-specific rule: FY < 1977 ends June 30 (FYy = Jul (y-1) .. Jun y).
1977 onward FY ends Sep 30 (FYy = Oct (y-1) .. Sep y). The U.S. federal
fiscal year boundary moved in 1976 under PL 93-344; first full year
under the new boundary was FY1977.
"""

from __future__ import annotations

import calendar
import logging
import re

log = logging.getLogger(__name__)


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

# U.S. federal fiscal-year boundary change.
_FY_BOUNDARY_YEAR = 1977


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
        if y < _FY_BOUNDARY_YEAR:
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


# ---------------------------------------------------------------------------
# Verbatim-date parser — turns page-side date strings produced by
# `_extract_dates` (e.g. "Fiscal Year 1991", "December 31, 1949",
# "1932-1939", "1940") into ISO `(start, end)` intervals. Used at retrieve
# time to overlap-check against the planner-emitted period.
# ---------------------------------------------------------------------------

_MONTHS_FULL = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)
_MONTH_TO_NUM: dict[str, int] = {m: i + 1 for i, m in enumerate(_MONTHS_FULL)}
_MONTH_TO_NUM.update({
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
})

# Order matters: try most specific first so e.g. "January 5, 1949"
# doesn't fall through to the month-year shape.
_VERB_MONTH_DAY_YEAR_RE = re.compile(
    r"^(?P<mon>[a-z]+)\.?\s+(?P<day>\d{1,2}),?\s+(?P<year>\d{4})$",
)
_VERB_MONTH_YEAR_RE = re.compile(
    r"^(?P<mon>[a-z]+)\.?\s+(?P<year>\d{4})$",
)
_VERB_FY_RE = re.compile(
    r"^(?:fiscal\s+year|fy)\s*(?P<year>\d{4})$",
)
_VERB_CY_RE = re.compile(
    r"^(?:calendar\s+year|cy)\s*(?P<year>\d{4})$",
)
_VERB_Q_RE = re.compile(
    r"^q(?P<q>[1-4])\s*(?P<year>\d{4})$",
)
_VERB_YEAR_RANGE_RE = re.compile(
    r"^(?P<a>\d{4})\s*(?:-|through|to)\s*"
    r"(?:[a-z]+\.?\s*)?(?P<b>\d{4})$",
)
_VERB_BARE_YEAR_RE = re.compile(r"^(?P<y>\d{4})$")
# `YYYY-MM` bulletin-month marker — seeded by extraction so every page
# carries its source issue's publication month as a date signal.
_VERB_YM_RE = re.compile(r"^(?P<y>\d{4})-(?P<m>\d{2})$")
# `M/D/YYYY` slash date — extraction normalizes 2-digit years to 4-digit
# before storing, so this parser only handles the 4-digit form.
_VERB_MD_Y_RE = re.compile(
    r"^(?P<mon>0?[1-9]|1[0-2])/(?P<day>0?[1-9]|[12]\d|3[01])/(?P<year>\d{4})$"
)


def _verbatim_to_intervals(s: str) -> list[tuple[str, str]]:
    """Parse one verbatim date STRING into ISO `(start, end)` intervals.

    Returns `[]` for inputs that don't match any recognized shape; callers
    treat that as "no signal" and use other heuristics.
    """
    if not s:
        return []
    t = s.strip().lower().replace(",", "")
    t = re.sub(r"\s+", " ", t).strip()

    m = _VERB_FY_RE.match(t)
    if m:
        try: return [_parse_point(f"FY{int(m.group('year'))}")]
        except ValueError: return []
    m = _VERB_CY_RE.match(t)
    if m:
        try: return [_parse_point(f"CY{int(m.group('year'))}")]
        except ValueError: return []
    m = _VERB_Q_RE.match(t)
    if m:
        try: return [_parse_point(f"Q{m.group('q')}-{int(m.group('year'))}")]
        except ValueError: return []
    m = _VERB_MD_Y_RE.match(t)
    if m:
        mn = int(m.group("mon"))
        d = int(m.group("day"))
        y = int(m.group("year"))
        if 1 <= d <= _last_day(y, mn):
            iso = f"{y:04d}-{mn:02d}-{d:02d}"
            return [(iso, iso)]
        return []
    m = _VERB_YM_RE.match(t)
    if m:
        y, mo = int(m.group("y")), int(m.group("m"))
        if 1 <= mo <= 12:
            return [(f"{y:04d}-{mo:02d}-01",
                     f"{y:04d}-{mo:02d}-{_last_day(y, mo):02d}")]
        return []
    m = _VERB_MONTH_DAY_YEAR_RE.match(t)
    if m:
        mn = _MONTH_TO_NUM.get(m.group("mon").rstrip("."))
        if mn is None:
            return []
        y = int(m.group("year")); d = int(m.group("day"))
        if 1 <= d <= _last_day(y, mn):
            iso = f"{y:04d}-{mn:02d}-{d:02d}"
            return [(iso, iso)]
        return []
    m = _VERB_MONTH_YEAR_RE.match(t)
    if m:
        mn = _MONTH_TO_NUM.get(m.group("mon").rstrip("."))
        if mn is None:
            return []
        y = int(m.group("year"))
        return [(f"{y:04d}-{mn:02d}-01",
                 f"{y:04d}-{mn:02d}-{_last_day(y, mn):02d}")]
    m = _VERB_YEAR_RANGE_RE.match(t)
    if m:
        a, b = int(m.group("a")), int(m.group("b"))
        if a > b:
            a, b = b, a
        return [(f"{a:04d}-01-01", f"{b:04d}-12-31")]
    m = _VERB_BARE_YEAR_RE.match(t)
    if m:
        y = int(m.group("y"))
        return [(f"{y:04d}-01-01", f"{y:04d}-12-31")]
    return []


def _intervals_overlap(a_start: str, a_end: str,
                       b_start: str, b_end: str) -> bool:
    return not (a_end < b_start or a_start > b_end)


class TreasuryPeriodParser:
    """`PeriodParser` impl for Treasury Bulletin period strings."""

    def year_window(self, period: str | None) -> tuple[int, int] | None:
        ivs = self.intervals(period)
        if not ivs:
            return None
        starts = [int(s[:4]) for s, _ in ivs]
        ends = [int(e[:4]) for _, e in ivs]
        return min(starts), max(ends)

    def intervals(
        self, period: str | None,
    ) -> list[tuple[str, str]] | None:
        if not period:
            return None
        try:
            return _period_to_intervals(period)
        except (TypeError, ValueError) as e:
            log.warning("period %r unparseable: %s", period, e)
            return None

    def verbatim_date_to_intervals(self, s: str) -> list[tuple[str, str]]:
        """Parse one of `row.dates` (a verbatim date string from the page)
        into ISO `(start, end)` intervals. Returns `[]` for unrecognized
        shapes."""
        return _verbatim_to_intervals(s)

    def dates_overlap_period(
        self, dates: list[str], period_intervals: list[tuple[str, str]],
    ) -> bool:
        """True iff any date string in `dates` parses to an interval that
        overlaps any of `period_intervals`. Empty `dates` (or all dates
        unparseable) returns False — callers decide what to do with
        no-signal pages."""
        if not dates or not period_intervals:
            return False
        for s in dates:
            for d_start, d_end in _verbatim_to_intervals(s):
                for p_start, p_end in period_intervals:
                    if _intervals_overlap(d_start, d_end, p_start, p_end):
                        return True
        return False
