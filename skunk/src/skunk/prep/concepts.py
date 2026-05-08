"""Tier 3 — concept dictionary (lazy, bootstrapped from CSV annotations).

Loads concepts.yaml (if it exists) for subagent consumption.
Can also bootstrap from the annotated OfficeQA CSV.

Usage:
    python -m skunk.prep.concepts \
        --csv /path/to/officeqa.csv \
        --output cache/concepts.yaml
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


def bootstrap_from_csv(csv_path: str, output_path: str) -> None:
    """Build concepts.yaml from the 'concepts' column of the annotated CSV."""
    import pandas as pd
    from collections import Counter

    df = pd.read_csv(csv_path)
    if "concepts" not in df.columns:
        print("[concepts] No 'concepts' column found in CSV")
        return

    concept_counts: Counter = Counter()
    for raw in df["concepts"].dropna():
        try:
            items = ast.literal_eval(str(raw))
            if isinstance(items, list):
                for item in items:
                    concept_counts[str(item).strip()] += 1
        except Exception:
            pass

    # Build a YAML file with definition placeholders for top concepts
    try:
        import yaml
    except ImportError:
        _write_yaml_manually(dict(concept_counts), output_path)
        return

    # Sort by frequency descending
    sorted_concepts = dict(sorted(concept_counts.items(), key=lambda x: -x[1]))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# OfficeQA concept dictionary\n")
        f.write("# Format: 'concept term': 'definition'\n\n")
        for term, count in sorted_concepts.items():
            # Write a placeholder definition (to be filled in manually)
            safe_term = term.replace("'", '"')
            safe_def = f"(count={count}) — definition needed"
            f.write(f"'{safe_term}': '{safe_def}'\n")

    print(f"[concepts] Wrote {len(sorted_concepts)} concepts to {output_path}")


def _write_yaml_manually(concepts: dict, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for term, count in sorted(concepts.items(), key=lambda x: -x[1]):
            safe_term = term.replace('"', '\\"')
            f.write(f'"{safe_term}": "(count={count}) definition needed"\n')


def load_concepts(concepts_yaml_path: str) -> dict[str, str]:
    """Load the concept dict from YAML. Returns {term: definition}."""
    p = Path(concepts_yaml_path)
    if not p.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(p.read_text()) or {}
    except ImportError:
        # Fallback manual YAML-ish parser
        result = {}
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ": " in line:
                k, v = line.split(": ", 1)
                result[k.strip("'\"")] = v.strip("'\"")
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", default="cache/concepts.yaml")
    args = parser.parse_args()
    bootstrap_from_csv(args.csv, args.output)
