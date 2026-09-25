"""CLI: `python -m qatfd.k8s {plan,submit,watch,pull}` — see the package docstring. Shells out to the
aws cli, helm and kubectl (all three must be on PATH and pointed at the experiments cluster / the `mit`
profile); nothing here talks to AWS or Kubernetes through an SDK, so the only Python dependency is PyYAML."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time

import yaml

from qatfd.k8s.cells import Cell, s3_completed_rows
from qatfd.k8s.sweeps import SWEEPS
from qatfd.paths import REPO_ROOT

CHART_DIR = REPO_ROOT / "deploy_exp" / "helm" / "experiments"
TERRAFORM_DIR = REPO_ROOT / "deploy_exp" / "terraform"
RESULTS_DIR = REPO_ROOT / "qatfd" / "results"
STATE_DIR = RESULTS_DIR / ".k8s"          # rendered values files + pulled status json, per release
SWEEP_LABEL = "carnot.io/sweep"           # pod label the chart sets (templates/_helpers.tpl)
_RUN_DIR_RE = re.compile(r"_(\d{8})_(\d{6})$")


def _run(cmd: list[str], *, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=capture, text=True)


def _log(msg: str) -> None:
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------
# cells
# ---------------------------------------------------------------------------------------------

def _build_cells(args: argparse.Namespace) -> list[Cell]:
    return SWEEPS[args.sweep].cells(args)


def _mark_complete(cells: list[Cell], args: argparse.Namespace) -> dict[str, tuple[str | None, int]]:
    """{label: (newest S3 run dir, rows)} for every cell; empty when the S3 check is off."""
    if args.no_skip_complete:
        return {}
    done: dict[str, tuple[str | None, int]] = {}
    for c in cells:
        done[c.label] = s3_completed_rows(args.results_bucket, args.results_prefix, c)
    return done


def _print_plan(cells: list[Cell], done: dict[str, tuple[str | None, int]]) -> list[Cell]:
    todo: list[Cell] = []
    for c in cells:
        run_dir, rows = done.get(c.label, (None, 0))
        if run_dir and rows >= c.expected_rows:
            state = f"complete ({rows} rows) in s3 {run_dir}"
        else:
            state = f"todo ({rows}/{c.expected_rows} rows{' at ' + run_dir if run_dir else ''})"
            todo.append(c)
        print(f"  {c.label:40s} {state}")
    print(f"{len(todo)}/{len(cells)} cell(s) to run")
    return todo


# ---------------------------------------------------------------------------------------------
# plan / submit
# ---------------------------------------------------------------------------------------------

def cmd_plan(args: argparse.Namespace) -> int:
    cells = _build_cells(args)
    todo = _print_plan(cells, _mark_complete(cells, args))
    if args.show_argv:
        for c in todo:
            print(f"\n{c.label}:\n  " + " \\\n  ".join(c.argv))
    return 0


def _role_arn(args: argparse.Namespace) -> str:
    if args.role_arn:
        return args.role_arn
    try:
        return _run(["terraform", f"-chdir={TERRAFORM_DIR}", "output", "-raw", "runner_role_arn"]).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        sys.exit(f"--role-arn not given and `terraform output runner_role_arn` failed ({e}); pass it explicitly")


def cmd_submit(args: argparse.Namespace) -> int:
    cells = _build_cells(args)
    todo = _print_plan(cells, _mark_complete(cells, args))
    if not todo:
        _log("nothing to run")
        return 0
    benchmarks = {c.benchmark for c in todo}
    if len(benchmarks) != 1:
        sys.exit(f"a release runs one benchmark (sizing profile); got {sorted(benchmarks)}")
    values: dict = {
        "benchmark": benchmarks.pop(),
        "image": {"tag": args.image_tag or "latest"},
        "serviceAccount": {"create": not args.no_sa_create},
        "s3": {"dataBucket": args.data_bucket, "resultsBucket": args.results_bucket, "resultsPrefix": args.results_prefix},
        "cells": [c.to_values() for c in todo],
    }
    if args.parallelism:
        values["job"] = {"parallelism": args.parallelism}
    if not args.dry_run:
        values["serviceAccount"]["roleArn"] = _role_arn(args)

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    values_path = STATE_DIR / f"{args.release}.values.yaml"
    values_path.write_text(yaml.safe_dump(values, sort_keys=False))
    helm = ["helm", "upgrade", "--install", args.release, str(CHART_DIR), "-n", args.namespace,
            "--create-namespace", "-f", str(values_path)]
    _log(f"values: {values_path}")
    if args.dry_run:
        print(" ".join(helm))
        return 0
    if not args.image_tag:
        sys.exit("--image-tag is required (the git sha build_push.sh pushed; `latest` is a foot-gun for reproducibility)")
    _log(" ".join(helm))
    r = _run(helm, check=False)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        return r.returncode
    return cmd_watch(args) if args.watch else 0


# ---------------------------------------------------------------------------------------------
# watch / pull
# ---------------------------------------------------------------------------------------------

def _kubectl_json(args: argparse.Namespace, *rest: str) -> dict:
    r = _run(["kubectl", "-n", args.namespace, *rest, "-o", "json"], check=False)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip())
    return json.loads(r.stdout)


def _release_cells(args: argparse.Namespace) -> list[Cell]:
    """The release's cells, from its ConfigMap (so `watch` / `pull` work from any machine)."""
    cm = _kubectl_json(args, "get", "configmap", f"{args.release}-cells")
    return [Cell(**{**c, "meta": {}}) for c in json.loads(cm["data"]["cells.json"])]


def _pod_stage(pod: dict) -> tuple[str, int]:
    """(human stage, restarts) from a pod's container statuses."""
    st = pod.get("status", {})
    phase = st.get("phase", "?")
    restarts = 0
    stage = phase
    for cs in st.get("initContainerStatuses", []) + st.get("containerStatuses", []):
        restarts += cs.get("restartCount", 0)
        state = cs.get("state", {})
        if "running" in state and cs["name"] != "chroma":
            stage = "fetching" if cs["name"] == "fetch-data" else "running"
        elif "waiting" in state and cs["name"] == "runner" and phase == "Pending":
            stage = "chroma starting" if any(c["name"] == "chroma" and "running" in c.get("state", {}) for c in st.get("initContainerStatuses", [])) else phase
        elif "terminated" in state and cs["name"] == "runner":
            stage = f"{phase} (exit {state['terminated'].get('exitCode')})"
    return stage, restarts


def _local_rows(cell: Cell) -> tuple[str | None, int]:
    base = RESULTS_DIR / cell.benchmark / cell.system_dir
    dirs = sorted(p for p in base.glob(f"{cell.label}_*") if _RUN_DIR_RE.search(p.name))
    if not dirs:
        return None, 0
    f = dirs[-1] / "results.jsonl"
    rows = sum(1 for line in f.read_text().splitlines() if line.strip()) if f.exists() else 0
    return dirs[-1].name, rows


def _pull(args: argparse.Namespace, cells: list[Cell]) -> None:
    """Mirror every cell's run dir(s) from the results bucket into qatfd/results/ (one sync per
    benchmark/system_dir group, include-filtered to the cells' labels)."""
    groups: dict[tuple[str, str], list[Cell]] = {}
    for c in cells:
        groups.setdefault((c.benchmark, c.system_dir), []).append(c)
    for (bench, sysdir), cs in groups.items():
        src = f"s3://{args.results_bucket}/{args.results_prefix.strip('/')}/{bench}/{sysdir}/"
        dst = RESULTS_DIR / bench / sysdir
        dst.mkdir(parents=True, exist_ok=True)
        cmd = ["aws", "s3", "sync", src, str(dst), "--only-show-errors", "--exclude", "*"]
        for c in cs:
            cmd += ["--include", f"{c.label}_*/*"]
        r = _run(cmd, check=False)
        if r.returncode != 0:
            _log(f"WARN: results pull failed for {src}: {r.stderr.strip()[:200]}")


def _status_json(args: argparse.Namespace) -> dict[int, dict]:
    """Per-index status json the runner writes to s3://<results>/<jobs>/<release>/."""
    local = STATE_DIR / args.release / "status"
    local.mkdir(parents=True, exist_ok=True)
    _run(["aws", "s3", "sync", f"s3://{args.results_bucket}/{args.jobs_prefix}/{args.release}/", str(local),
          "--only-show-errors"], check=False)
    out: dict[int, dict] = {}
    for f in local.glob("*.json"):
        try:
            d = json.loads(f.read_text())
            out[int(d["index"])] = d
        except (ValueError, KeyError):
            continue
    return out


def cmd_watch(args: argparse.Namespace) -> int:
    cells = _release_cells(args)
    _log(f"watching release {args.release}: {len(cells)} cell(s); results -> {RESULTS_DIR}")
    while True:
        job = _kubectl_json(args, "get", "job", args.release)
        pods = _kubectl_json(args, "get", "pods", "-l", f"{SWEEP_LABEL}={args.release}")["items"]
        newest: dict[int, dict] = {}
        for p in pods:
            idx = int(p["metadata"].get("annotations", {}).get("batch.kubernetes.io/job-completion-index", -1))
            if idx not in newest or p["metadata"]["creationTimestamp"] > newest[idx]["metadata"]["creationTimestamp"]:
                newest[idx] = p
        status = _status_json(args)
        if not args.no_pull:
            _pull(args, cells)

        st = job.get("status", {})
        conds = {c["type"]: c["status"] for c in st.get("conditions", [])}
        print(f"\n== {args.release}: active={st.get('active', 0)} succeeded={st.get('succeeded', 0)} "
              f"failed={st.get('failed', 0)} of {job['spec']['completions']}  ({datetime.datetime.now():%H:%M:%S})")
        print(f"  {'idx':>3} {'label':40s} {'stage':22s} {'node':28s} {'rst':>3} {'rows':>9} {'elapsed':>8}")
        for i, c in enumerate(cells):
            pod = newest.get(i)
            stage, restarts = _pod_stage(pod) if pod else ("not created", 0)
            node = (pod or {}).get("spec", {}).get("nodeName", "") or ""
            _, rows = _local_rows(c)
            el = status.get(i, {}).get("elapsed_s")
            elapsed = f"{el // 60}m" if isinstance(el, int) else ""
            print(f"  {i:>3} {c.label:40s} {stage:22s} {node[-28:]:28s} {restarts:>3} {rows:>4}/{c.expected_rows:<4} {elapsed:>8}")

        if conds.get("Complete") == "True":
            _log("job complete")
            return 0
        if conds.get("Failed") == "True":
            _log(f"job FAILED: {[c for c in st.get('conditions', []) if c['type'] == 'Failed']}")
            return 1
        time.sleep(args.interval)


def cmd_pull(args: argparse.Namespace) -> int:
    cells = _release_cells(args)
    _pull(args, cells)
    for c in cells:
        run_dir, rows = _local_rows(c)
        print(f"  {c.label:40s} {rows:>4}/{c.expected_rows:<4} {run_dir or '(no run dir yet)'}")
    return 0


# ---------------------------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------------------------

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--namespace", default=os.environ.get("QATFD_K8S_NAMESPACE", "experiments"))
    p.add_argument("--data-bucket", default="carnot-research")
    p.add_argument("--results-bucket", default="carnot-research-experiments")
    p.add_argument("--results-prefix", default="results")
    p.add_argument("--jobs-prefix", default="jobs")


def _add_sweep(p: argparse.ArgumentParser, sweep: str | None) -> None:
    p.add_argument("--sweep", required=True, choices=sorted(SWEEPS))
    p.add_argument("--no-skip-complete", action="store_true", help="do not consult S3; plan every cell")
    if sweep in SWEEPS:
        SWEEPS[sweep].add_arguments(p)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    # the sweep's own knobs depend on --sweep, so peek at it before building the full parser
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--sweep")
    sweep = pre.parse_known_args(argv)[0].sweep

    p = argparse.ArgumentParser(prog="python -m qatfd.k8s", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("plan", help="enumerate a sweep's cells and which are already complete on S3")
    _add_common(sp)
    _add_sweep(sp, sweep)
    sp.add_argument("--show-argv", action="store_true")
    sp.set_defaults(fn=cmd_plan)

    sp = sub.add_parser("submit", help="helm install a sweep as an Indexed Job (one pod per cell)")
    sp.add_argument("release", help="sweep / Helm release name (lowercase, dashes)")
    _add_common(sp)
    _add_sweep(sp, sweep)
    sp.add_argument("--image-tag", help="image tag build_push.sh pushed (git sha)")
    sp.add_argument("--role-arn", help="runner IRSA role ARN (default: terraform output runner_role_arn)")
    sp.add_argument("--parallelism", type=int, help="override the benchmark profile's parallelism")
    sp.add_argument("--no-sa-create", action="store_true", help="another release in the namespace owns the ServiceAccount")
    sp.add_argument("--dry-run", action="store_true", help="write the values file and print the helm command only")
    sp.add_argument("--watch", action="store_true", help="then watch the job and pull results as they land")
    sp.add_argument("--interval", type=int, default=60)
    sp.add_argument("--no-pull", action="store_true")
    sp.set_defaults(fn=cmd_submit)

    sp = sub.add_parser("watch", help="follow a submitted sweep; pulls results into qatfd/results/ every tick")
    sp.add_argument("release")
    _add_common(sp)
    sp.add_argument("--interval", type=int, default=60)
    sp.add_argument("--no-pull", action="store_true")
    sp.set_defaults(fn=cmd_watch)

    sp = sub.add_parser("pull", help="pull a sweep's run dirs from S3 into qatfd/results/ once")
    sp.add_argument("release")
    _add_common(sp)
    sp.set_defaults(fn=cmd_pull)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
