"""Pytest root conftest: put the grc-officeqa app root on sys.path so tests import
the `officeqa` package (and `eval.*`) without an install step — the same layout
qatfd uses (repo dir on sys.path, package inside)."""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
