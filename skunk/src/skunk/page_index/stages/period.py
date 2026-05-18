"""Runtime stage: Period parser — period string → (min_year, max_year).

The retrieve operator applies this filter against each candidate page's
year envelope. Period grammar is corpus-specific (e.g. fiscal-year
boundary rules differ between countries / publishers).
"""

from __future__ import annotations

from typing import Protocol


class PeriodParser(Protocol):
    """Parse a DSL period string into a `(min_year, max_year)` envelope.

    Returns None for an unparseable or empty period; callers treat that
    as "no year filter".
    """

    def year_window(self, period: str | None) -> tuple[int, int] | None: ...
