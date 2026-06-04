"""Tinker spend tracking for the datagen harness.

We can't see the Tinker billing dashboard (the API key comes from a funder), so
we track spend locally: every harness run records the exact sampling token
usage and the computed dollar cost to ``~/.tinker-cost/datagen_{ts}.json``. The
token counts are ground truth (captured in ``TinkerBackend.usage``); the dollar
figure depends on the per-token rates below.

Run ``python -m skunk.datagen.tinker_cost`` to aggregate every recorded run into
a single total.

PRICES: Tinker lists per-million-token prices (Prefill = input/prompt tokens,
Sample = generated tokens) on its pricing page, and MoE models are billed on
ACTIVE parameters (Qwen3.6-35B-A3B -> 3B active). The page is JS-rendered so the
numbers can't be fetched programmatically -- fill them in below (or pass
--tinker-prefill-price / --tinker-sample-price to the harness). Until set, runs
record token counts with ``cost_usd = null`` so nothing is lost; backfill by
editing the JSON or re-deriving from the recorded tokens.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import asdict, dataclass

# Per-MILLION-token USD prices, keyed by Tinker base_model. Fill from
# https://tinker-docs.thinkingmachines.ai/tinker/models/ (the "Prefill" and
# "Sample" columns). None means "unknown" -> cost is recorded as null.
TINKER_PRICING_PER_MTOK: dict[str, dict[str, float | None]] = {
    "Qwen/Qwen3.6-35B-A3B": {"prefill": 0.36, "sample": 0.89},
}

DEFAULT_COST_DIR = pathlib.Path.home() / ".tinker-cost"


@dataclass
class RunCost:
    timestamp: str
    tool: str
    base_model: str
    prefill_tokens: int
    sample_tokens: int
    n_sampling_calls: int
    prefill_price_per_mtok: float | None
    sample_price_per_mtok: float | None
    cost_usd: float | None


def resolve_prices(
    base_model: str,
    prefill_override: float | None = None,
    sample_override: float | None = None,
) -> tuple[float | None, float | None]:
    """Resolve (prefill, sample) per-Mtok prices: override > table > None."""
    table = TINKER_PRICING_PER_MTOK.get(base_model, {})
    prefill = prefill_override if prefill_override is not None else table.get("prefill")
    sample = sample_override if sample_override is not None else table.get("sample")
    return prefill, sample


def compute_cost(
    prefill_tokens: int,
    sample_tokens: int,
    prefill_price_per_mtok: float | None,
    sample_price_per_mtok: float | None,
) -> float | None:
    """USD cost, or None if either rate is unknown."""
    if prefill_price_per_mtok is None or sample_price_per_mtok is None:
        return None
    return (
        prefill_tokens / 1_000_000 * prefill_price_per_mtok
        + sample_tokens / 1_000_000 * sample_price_per_mtok
    )


def write_run_report(
    timestamp: str,
    base_model: str,
    prefill_tokens: int,
    sample_tokens: int,
    n_sampling_calls: int,
    prefill_price_per_mtok: float | None,
    sample_price_per_mtok: float | None,
    cost_dir: pathlib.Path = DEFAULT_COST_DIR,
    tool: str = "skunk.datagen.harness",
) -> pathlib.Path:
    """Write ``{cost_dir}/datagen_{timestamp}.json`` for one harness run."""
    cost_dir.mkdir(parents=True, exist_ok=True)
    record = RunCost(
        timestamp=timestamp,
        tool=tool,
        base_model=base_model,
        prefill_tokens=prefill_tokens,
        sample_tokens=sample_tokens,
        n_sampling_calls=n_sampling_calls,
        prefill_price_per_mtok=prefill_price_per_mtok,
        sample_price_per_mtok=sample_price_per_mtok,
        cost_usd=compute_cost(
            prefill_tokens, sample_tokens, prefill_price_per_mtok, sample_price_per_mtok
        ),
    )
    path = cost_dir / f"datagen_{timestamp}.json"
    with path.open("w") as f:
        json.dump(asdict(record), f, indent=2)
    return path


def aggregate(cost_dir: pathlib.Path = DEFAULT_COST_DIR) -> dict:
    """Sum token usage and cost across every recorded run."""
    runs: list[dict] = []
    for path in sorted(cost_dir.glob("*.json")):
        try:
            with path.open() as f:
                runs.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue

    total_prefill = sum(r.get("prefill_tokens", 0) or 0 for r in runs)
    total_sample = sum(r.get("sample_tokens", 0) or 0 for r in runs)
    total_calls = sum(r.get("n_sampling_calls", 0) or 0 for r in runs)
    known = [r for r in runs if r.get("cost_usd") is not None]
    total_cost = sum(r["cost_usd"] for r in known)
    n_unknown = len(runs) - len(known)
    return {
        "n_runs": len(runs),
        "total_prefill_tokens": total_prefill,
        "total_sample_tokens": total_sample,
        "total_sampling_calls": total_calls,
        "total_cost_usd": total_cost,
        "n_runs_without_price": n_unknown,
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Tinker datagen spend.")
    parser.add_argument(
        "--cost-dir",
        type=pathlib.Path,
        default=DEFAULT_COST_DIR,
        help=f"Directory of per-run cost JSON files (default: {DEFAULT_COST_DIR}).",
    )
    args = parser.parse_args()

    summary = aggregate(args.cost_dir)
    if summary["n_runs"] == 0:
        print(f"No cost reports found in {args.cost_dir}.")
        return

    print(f"Tinker datagen spend across {summary['n_runs']} run(s) in {args.cost_dir}:\n")
    print(f"  {'timestamp':<20} {'model':<26} {'prefill':>13} {'sample':>13} {'cost_usd':>10}")
    for r in summary["runs"]:
        cost = r.get("cost_usd")
        cost_str = f"${cost:,.4f}" if cost is not None else "  (no price)"
        print(
            f"  {r.get('timestamp',''):<20} {r.get('base_model',''):<26} "
            f"{r.get('prefill_tokens',0):>13,} {r.get('sample_tokens',0):>13,} {cost_str:>10}"
        )
    print(
        f"\n  TOTAL: {summary['total_prefill_tokens']:,} prefill + "
        f"{summary['total_sample_tokens']:,} sample tokens across "
        f"{summary['total_sampling_calls']:,} calls"
    )
    print(f"  TOTAL COST: ${summary['total_cost_usd']:,.4f}")
    if summary["n_runs_without_price"]:
        print(
            f"  NOTE: {summary['n_runs_without_price']} run(s) have no price set "
            "(cost excluded). Set TINKER_PRICING_PER_MTOK in tinker_cost.py or pass "
            "--tinker-prefill-price/--tinker-sample-price to the harness."
        )


if __name__ == "__main__":
    main()
