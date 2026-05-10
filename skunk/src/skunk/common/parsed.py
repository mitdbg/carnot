"""Per-page text slicing of the pre-parsed Treasury Bulletin corpus.

Files live at:
  {OFFICEQA_PARSED_DIR}/treasury_bulletin_{YYYY}_{MM}.txt

Page numbers appear as bare integers on their own lines (bulletin footer).
We use the longest monotonically-increasing subsequence of values in [1, 500]
to identify true page markers and build a slice map.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from skunk.common.context import HarnessContext
from skunk.dsl import PageRef

_PAGE_RE = re.compile(r"^\s*(\d{1,4})\s*$")
_PARSED_DEFAULT = Path.home() / "Desktop/officeqa/treasury_bulletins_parsed/transformed/treasury_bulletins_transformed"


def parsed_dir() -> Path:
    d = os.environ.get("OFFICEQA_PARSED_DIR")
    return Path(d) if d else _PARSED_DEFAULT


def _lis_indices(vals: list[int]) -> list[int]:
    """Return indices into vals for the longest strictly increasing subsequence.

    True LIS, not a greedy "drop-if-decreasing" filter: stray non-page integers
    in body text (table values, dates) are non-monotonic and would corrupt a
    greedy run. LIS reconstructs the consistent page-marker sequence around them.
    """
    n = len(vals)
    if not n:
        return []
    dp = [1] * n
    parent = [-1] * n
    for i in range(1, n):
        for j in range(i):
            if vals[j] < vals[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j
    best = max(range(n), key=lambda i: dp[i])
    path: list[int] = []
    while best != -1:
        path.append(best)
        best = parent[best]
    return list(reversed(path))


def build_parsed_page_index(month_str: str, cache_dir: str) -> dict[str, list[int]]:
    """Build (or load) a {bulletin_page_str: [start_line, end_line]} map.

    Lines are 0-based. The page-number marker is a footer; text for page N
    spans lines [start, end] inclusive, with the marker line itself excluded.
    """
    cache_path = Path(cache_dir) / "parsed_index" / f"{month_str}.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    year, mon = month_str.split("-")
    bulk = parsed_dir() / f"treasury_bulletin_{year}_{mon}.txt"
    if not bulk.exists():
        return {}

    lines = bulk.read_text(errors="replace").splitlines()

    candidates: list[tuple[int, int]] = []
    for i, line in enumerate(lines):
        m = _PAGE_RE.match(line)
        if m:
            v = int(m.group(1))
            if 1 <= v <= 500:
                candidates.append((i, v))

    if not candidates:
        return {}

    vals = [v for _, v in candidates]
    lis_idx = _lis_indices(vals)
    surviving = [candidates[i] for i in lis_idx]

    result: dict[str, list[int]] = {}
    prev_end = -1
    for line_idx, page_num in surviving:
        start = prev_end + 1
        end = line_idx - 1
        if start <= end:
            result[str(page_num)] = [start, end]
        prev_end = line_idx

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result))
    return result


def get_parsed_text_for_page(ref: PageRef, ctx: HarnessContext) -> str | None:
    if ref.month is None or ref.page is None:
        return None
    index = build_parsed_page_index(ref.month, ctx.cache_dir)
    entry = index.get(str(ref.page))
    if entry is None:
        return None

    year, mon = ref.month.split("-")
    bulk = parsed_dir() / f"treasury_bulletin_{year}_{mon}.txt"
    if not bulk.exists():
        return None

    start, end = entry
    lines = bulk.read_text(errors="replace").splitlines()
    slice_lines = lines[start : end + 1]
    return "\n".join(slice_lines) if slice_lines else None
