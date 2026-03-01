from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def _load_metrics(path: Path) -> Dict[str, float]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check(metrics: Dict[str, float]) -> List[str]:
    errors: List[str] = []
    if metrics.get("canonicalized_value_count", 0) > metrics.get("raw_value_count", 0):
        errors.append("canonicalized_value_count should not exceed raw_value_count")
    if metrics.get("materialized_value_count", 0) < metrics.get("canonicalized_value_count", 0):
        errors.append("materialized_value_count should be >= canonicalized_value_count")
    if metrics.get("propagated_value_count", 0) < metrics.get("materialized_value_count", 0):
        errors.append("propagated_value_count should be >= materialized_value_count")
    if metrics.get("active_tag_count", 0) <= 0:
        errors.append("active_tag_count should be positive")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Path to postprocess_metrics.json. Defaults to sem_map/postprocess_metrics.json.",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    metrics_path = Path(args.metrics) if args.metrics else (here / "sem_map/postprocess_metrics.json")
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {metrics_path}. Run with --dump-intermediate first."
        )
    metrics = _load_metrics(metrics_path)
    errors = _check(metrics)
    if errors:
        print("Validation failed:")
        for err in errors:
            print(f"- {err}")
        return 1
    print("Validation passed.")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
