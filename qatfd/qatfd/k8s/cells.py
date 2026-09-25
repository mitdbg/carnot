"""A cell = one experiment configuration = one pod = one invocation of a qatfd driver (the stock runner or
scripts/bootstrap_enrich_upper_bound.py). `Cell.to_values()` is exactly the dict the chart's `cells:` list
takes (deploy_exp/helm/experiments/values.yaml documents the fields); the extra `system_dir` field is what
the S3 completeness check and the results pull need to find the cell's run dir."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass, field


@dataclass
class Cell:
    label: str            # run label == experiments.run_name / +ub.label; run dir is <label>_<timestamp>
    benchmark: str        # qatfd benchmark name (results/<benchmark>/...)
    system_dir: str       # results/<benchmark>/<system_dir>/ the driver writes under (e.g. search_agent, search_agent_ub)
    collection: str       # chroma collection the sidecar warms and the driver queries
    store_prefix: str     # s3://<data bucket>/<store_prefix>/ -> /data/chromadb
    data: list            # benchmark files to pull (see qatfd.k8s.benchmarks)
    argv: list[str]       # the driver command, run from /app/qatfd; the chart appends results_root + chroma host/port
    expected_rows: int    # results.jsonl rows a complete run has
    meta: dict = field(default_factory=dict)  # sweep-specific tags (x, seed, cell), for tables

    def to_values(self) -> dict:
        d = asdict(self)
        d.pop("meta")
        return d


# --- S3 completeness ---------------------------------------------------------------------------------
# A cell is complete when the newest run dir for its label in the results bucket holds >= expected_rows
# rows, the same rule the bash drivers apply locally. Run dirs are <label>_<YYYYmmdd>_<HHMMSS>/.

_RUN_DIR_RE = re.compile(r"_(\d{8})_(\d{6})/$")


def _aws(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["aws", *args], check=check, capture_output=True, text=True)


def s3_run_dirs(bucket: str, results_prefix: str, cell: Cell) -> list[str]:
    """Run-dir prefixes for `cell` in the results bucket, oldest first."""
    prefix = f"{results_prefix.strip('/')}/{cell.benchmark}/{cell.system_dir}/{cell.label}_"
    out = _aws(["s3api", "list-objects-v2", "--bucket", bucket, "--prefix", prefix, "--delimiter", "/",
                "--query", "CommonPrefixes[].Prefix", "--output", "json"]).stdout
    prefixes = json.loads(out or "null") or []
    return sorted(p for p in prefixes if _RUN_DIR_RE.search(p[len(prefix) - 1:]))


def s3_result_rows(bucket: str, run_dir_prefix: str) -> int:
    """Rows in <run dir>/results.jsonl on S3 (0 when absent)."""
    r = _aws(["s3", "cp", f"s3://{bucket}/{run_dir_prefix}results.jsonl", "-"], check=False)
    if r.returncode != 0:
        return 0
    return sum(1 for line in r.stdout.splitlines() if line.strip())


def s3_completed_rows(bucket: str, results_prefix: str, cell: Cell) -> tuple[str | None, int]:
    """(newest run dir prefix, its results.jsonl rows) for `cell`, or (None, 0) when it never ran."""
    dirs = s3_run_dirs(bucket, results_prefix, cell)
    if not dirs:
        return None, 0
    return dirs[-1], s3_result_rows(bucket, dirs[-1])
