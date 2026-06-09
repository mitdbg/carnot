"""Thin entrypoint for the OfficeQA Cup practice harness.

The implementation lives in :mod:`cup_kit.practice_server` so it can be
imported by the smoke test alongside the rest of the kit. This shim
exists so teams can run the more memorable ``python practice_server.py``
from the kit's top-level directory.
"""

from __future__ import annotations

from cup_kit.practice_server import main

if __name__ == "__main__":
    raise SystemExit(main())
