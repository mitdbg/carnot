"""Sweep generators: each module exposes `NAME`, `add_arguments(parser)` for its knobs, and
`cells(args) -> list[Cell]`. Register new sweeps in `SWEEPS`."""

from __future__ import annotations

from qatfd.k8s.sweeps import bootstrap_enrich_ub

SWEEPS = {
    bootstrap_enrich_ub.NAME: bootstrap_enrich_ub,
}
