"""Post-processing scripts for qatfd experiment results.

This package houses everything that turns the per-question ``report.csv`` files
written under ``results/`` into paper-ready artifacts (LaTeX tables, plots, ...).

It is intentionally standalone: modules here read ``report.csv`` / ``config.yaml``
directly and do NOT import the ``qatfd`` runtime package, so they can run without
the (heavy) eval/runtime dependencies installed.
"""
