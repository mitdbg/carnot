"""Runtime stage: Period parser — period string → year window / ISO intervals.

The retrieve operator applies this filter against each candidate page's
year envelope. Period grammar is corpus-specific (e.g. fiscal-year
boundary rules differ between countries / publishers).
"""

from __future__ import annotations

from typing import Protocol


class PeriodParser(Protocol):
    """Parse a DSL period string into year-level or ISO-interval form.

    Both methods return None for an unparseable or empty period; callers
    treat that as "no period filter".
    """

    def year_window(self, period: str | None) -> tuple[int, int] | None: ...

    def intervals(
        self, period: str | None,
    ) -> list[tuple[str, str]] | None:
        """Parse into a list of `(start_iso, end_iso)` intervals.

        Day-level granularity; used by retrieve to overlap-check against
        per-page structured date envelopes.
        """
        ...

    def dates_overlap_period(
        self, dates: list[str], period_intervals: list[tuple[str, str]],
    ) -> bool:
        """True iff any verbatim date string in `dates` parses to an
        interval that overlaps any of `period_intervals`. Returns False
        for empty / unparseable date lists; callers decide the default."""
        ...
