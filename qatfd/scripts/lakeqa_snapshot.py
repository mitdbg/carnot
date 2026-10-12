"""Snapshot the full LakeQA data lake (s3://lakeqa-yc4103-datalake, ~9.5 TB / ~40M objects) into
s3://carnot-research/lakeqa/lake/ so experiments run against a copy that never changes.

Steps (each one resumable; `all` runs them in order):
  list      list the source into per-unit TSVs (key, size, etag, last_modified) under WORK/src/
  copy      server-side copy of every listed key with s5cmd, in batches of --batch-keys keys; a batch is
            marked done only when its `s5cmd run` exits 0, so a rerun only redoes unfinished batches
  verify    list the destination into WORK/dst/ and diff it unit by unit against the source listing
  fixup     recopy missing / mismatched keys with boto3 (handles >5 GB objects and keys s5cmd can't take);
            a key whose source changed or vanished since `list` is recorded in WORK/drift.jsonl
  finalize  verify again, then upload manifest.tsv.gz (what we hold), source_listing.tsv.gz (what their
            bucket had, with its original LastModified) and COPY_REPORT.json to lakeqa/

The key space under each top prefix (datagov/, wikipedia/) is split into units that together cover every
key exactly once: one unit per printable-ASCII first character (listed with `s5cmd ls`, which is ~6x faster
than botocore's XML parsing), plus a `lo` unit (keys sorting before ' ') and a `hi` unit (keys whose first
character is past '~', i.e. non-ASCII titles) listed with boto3. First characters that are s5cmd glob
characters are listed with boto3 too. S3 lists keys in UTF-8 byte order; comparisons here use bytes.

Run (in tmux; AWS_PROFILE defaults to mit; needs ~/.local/bin/s5cmd and a venv with boto3):
    ~/.venvs/lakeqa/bin/python qatfd/scripts/lakeqa_snapshot.py all 2>&1 | tee -a ~/lakeqa_snapshot.log
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

SRC_BUCKET = "lakeqa-yc4103-datalake"
DST_BUCKET = "carnot-research"
TOPS = ("datagov/", "wikipedia/")
REGION = "us-east-1"  # both buckets
S5CMD = str(Path("~/.local/bin/s5cmd").expanduser())

_ASCII = [chr(c) for c in range(0x20, 0x7F)]
_GLOB = set("*?[]{}\\")  # s5cmd would expand these in an `ls` pattern
_HI_START = "~\U0010ffff"  # sorts after every key whose first character is ASCII
_BAD_KEY_CHARS = re.compile(r"[\t\n\r]")  # can't go in a TSV row or an s5cmd command line


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def s3_client(pool: int = 64):
    return boto3.session.Session(region_name=REGION).client(
        "s3", config=Config(max_pool_connections=pool, retries={"max_attempts": 10, "mode": "standard"})
    )


# ---------------------------------------------------------------------------------------------------------
# units and unit files


def units() -> list[tuple[str, str, str, str | None]]:
    """(unit_id, top, kind, first_char) with kind in {"lo", "prefix", "hi"}, in S3 key order."""
    out = []
    for top in TOPS:
        t = top.rstrip("/")
        out.append((f"{t}-lo", top, "lo", None))
        out += [(f"{t}-{ord(c):02x}", top, "prefix", c) for c in _ASCII]
        out.append((f"{t}-hi", top, "hi", None))
    return out


Row = tuple[str, int, str, str]  # key (relative to the listing base), size, etag, last_modified


def write_unit(path: Path, rows) -> tuple[int, int]:
    """Stream rows into a TSV.gz (+ .weird.jsonl for keys with tabs/newlines) and a .meta.json of totals."""
    tmp = path.with_name(path.name + ".tmp")
    n = nbytes = 0
    weird = []
    with gzip.open(tmp, "wt", encoding="utf-8", newline="\n") as f:
        for key, size, etag, lm in rows:
            n, nbytes = n + 1, nbytes + size
            if _BAD_KEY_CHARS.search(key):
                weird.append((key, size, etag, lm))
            else:
                f.write(f"{key}\t{size}\t{etag}\t{lm}\n")
    stem = path.name.removesuffix(".tsv.gz")
    if weird:
        (path.parent / f"{stem}.weird.jsonl").write_text("".join(json.dumps(r) + "\n" for r in weird))
    (path.parent / f"{stem}.meta.json").write_text(json.dumps({"objects": n, "bytes": nbytes}))
    os.replace(tmp, path)  # last, so an existing .tsv.gz means the unit is complete
    return n, nbytes


def iter_unit(path: Path):
    """Rows of a unit's TSV in listing order (the .weird.jsonl keys are not included)."""
    with gzip.open(path, "rt", encoding="utf-8", newline="\n") as f:
        for line in f:
            key, size, etag, lm = line.rstrip("\n").split("\t")
            yield key, int(size), etag, lm


def weird_rows(path: Path) -> list[Row]:
    p = path.parent / (path.name.removesuffix(".tsv.gz") + ".weird.jsonl")
    return [tuple(json.loads(line)) for line in p.read_text().splitlines()] if p.exists() else []


def unit_meta(work: Path, which: str, uid: str) -> dict:
    return json.loads((work / which / f"{uid}.meta.json").read_text())


def _list_s5cmd(bucket: str, base: str, top: str, c: str):
    pattern = f"s3://{bucket}/{base}{top}{c}*"
    strip = len(f"s3://{bucket}/{base}")
    proc = subprocess.Popen(
        [S5CMD, "--json", "ls", pattern],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env={**os.environ, "AWS_REGION": REGION},
    )
    for line in proc.stdout:
        o = json.loads(line)
        if o.get("type") == "file":
            yield o["key"][strip:], int(o["size"]), o["etag"].strip('"'), o["last_modified"]
    err = proc.stderr.read()
    if proc.wait() != 0 and "no object found" not in err:
        raise RuntimeError(f"s5cmd ls {pattern} failed: {err.strip()[:500]}")


def _list_boto(s3, bucket: str, base: str, top: str, kind: str, c: str | None):
    kw = {"Bucket": bucket, "Prefix": base + top + (c if kind == "prefix" else "")}
    stop = None
    if kind == "lo":
        stop = (base + top + " ").encode()
    elif kind == "hi":
        kw["StartAfter"] = base + top + _HI_START
    for page in s3.get_paginator("list_objects_v2").paginate(**kw):
        for o in page.get("Contents", []):
            if stop is not None and o["Key"].encode() >= stop:
                return
            yield o["Key"][len(base) :], o["Size"], o["ETag"].strip('"'), o["LastModified"].isoformat()


def list_all(work: Path, which: str, bucket: str, base: str, us, threads: int, relist: bool) -> None:
    """List every unit into work/<which>/<unit_id>.tsv.gz (skipping ones already listed unless `relist`)."""
    out_dir = work / which
    if relist and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [u for u in us if not (out_dir / f"{u[0]}.tsv.gz").exists()]
    log(f"list {which}: s3://{bucket}/{base}{{{','.join(TOPS)}}} ({len(todo)}/{len(us)} units to list)")
    s3 = s3_client(threads * 2)
    lock = threading.Lock()
    prog = {"units": 0, "keys": 0, "bytes": 0}
    t0 = time.time()

    def one(u):
        uid, top, kind, c = u
        if kind == "prefix" and c not in _GLOB:
            rows = _list_s5cmd(bucket, base, top, c)
        else:
            rows = _list_boto(s3, bucket, base, top, kind, c)
        n, b = write_unit(out_dir / f"{uid}.tsv.gz", rows)
        with lock:
            prog["units"] += 1
            prog["keys"] += n
            prog["bytes"] += b
            dt = time.time() - t0
            log(
                f"  {which} {uid}: {n:,} keys | {prog['units']}/{len(todo)} units, {prog['keys']:,} keys, "
                f"{prog['bytes'] / 1e12:.3f} TB, {prog['keys'] / max(dt, 1):,.0f} keys/s"
            )

    # biggest-first would balance better, but sizes are unknown until listed; common letters go first anyway
    with ThreadPoolExecutor(threads) as ex:
        for fut in as_completed([ex.submit(one, u) for u in todo]):
            fut.result()


def totals(work: Path, which: str, us) -> tuple[int, int]:
    metas = [unit_meta(work, which, u[0]) for u in us]
    return sum(m["objects"] for m in metas), sum(m["bytes"] for m in metas)


# ---------------------------------------------------------------------------------------------------------
# copy (s5cmd)


def _q(s: str) -> str:
    """Quote a URL for an `s5cmd run` line (double quotes; backslash-escape backslash and double quote)."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def copy_all(work: Path, dst_base: str, us, procs: int, workers: int, batch_keys: int) -> None:
    done_dir, cmd_dir, err_dir = work / "copied", work / "cmds", work / "copy_errors"
    for d in (done_dir, cmd_dir, err_dir):
        d.mkdir(parents=True, exist_ok=True)
    total_keys, total_bytes = totals(work, "src", us)
    already = sum(1 for _ in done_dir.iterdir())
    log(
        f"copy: {total_keys:,} keys, {total_bytes / 1e12:.3f} TB listed; {already} batches already done "
        f"({procs} s5cmd x {workers} workers, {batch_keys:,} keys per batch)"
    )

    lock = threading.Lock()
    slots = threading.Semaphore(procs * 2)  # bounds how many command files sit on disk at once
    prog = {"keys": 0, "bytes": 0, "batches": 0, "failed": 0, "skipped_keys": 0}
    t0 = time.time()
    env = {**os.environ, "AWS_REGION": REGION}

    def run_batch(bid: str, cmd_file: Path, n: int, nbytes: int) -> None:
        err_file = err_dir / f"{bid}.err"
        try:
            with open(err_file, "w") as err:
                rc = subprocess.run(
                    [
                        S5CMD,
                        "--numworkers",
                        str(workers),
                        "--retry-count",
                        "10",
                        "--log",
                        "error",
                        "run",
                        str(cmd_file),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=err,
                    env=env,
                ).returncode
            with lock:
                prog["batches"] += 1
                prog["keys"] += n
                prog["bytes"] += nbytes
                if rc == 0:
                    (done_dir / bid).touch()
                    err_file.unlink(missing_ok=True)
                else:
                    prog["failed"] += 1
                    log(f"  batch {bid} FAILED (s5cmd exit {rc}); errors in {err_file}; left for verify/fixup")
                dt = time.time() - t0
                rate = prog["keys"] / max(dt, 1)
                left = total_keys - prog["skipped_keys"] - prog["keys"]
                log(
                    f"  copy {bid}: {prog['batches']} batches this run, {prog['keys']:,} keys / "
                    f"{prog['bytes'] / 1e12:.3f} TB copied this run, {left:,} keys left, {rate:,.0f} keys/s, "
                    f"ETA {left / max(rate, 1e-9) / 3600:.1f} h, {prog['failed']} failed batches"
                )
        finally:
            cmd_file.unlink(missing_ok=True)
            slots.release()

    futs = []
    with ThreadPoolExecutor(procs) as ex:
        for uid, *_ in us:
            src = work / "src" / f"{uid}.tsv.gz"
            rows = iter_unit(src)
            j = 0
            while True:
                chunk = [r for _, r in zip(range(batch_keys), rows)]
                if not chunk:
                    break
                bid = f"{uid}.{j:05d}"
                j += 1
                if (done_dir / bid).exists():
                    prog["skipped_keys"] += len(chunk)
                    continue
                slots.acquire()
                cmd_file = cmd_dir / f"{bid}.txt"
                with open(cmd_file, "w", encoding="utf-8") as f:
                    for key, *_ in chunk:
                        f.write(
                            f"cp --raw {_q(f's3://{SRC_BUCKET}/{key}')} {_q(f's3://{DST_BUCKET}/{dst_base}{key}')}\n"
                        )
                futs.append(ex.submit(run_batch, bid, cmd_file, len(chunk), sum(r[1] for r in chunk)))
        for fut in futs:
            fut.result()
    log(f"copy: done ({prog['failed']} failed batches; the .weird.jsonl keys, if any, are left to fixup)")


# ---------------------------------------------------------------------------------------------------------
# verify / fixup


def read_drift(work: Path) -> dict[str, tuple[int, str] | None]:
    """key -> the source's (size, etag) at fixup time, or None if the key vanished from the source."""
    p = work / "drift.jsonl"
    out: dict[str, tuple[int, str] | None] = {}
    if p.exists():
        for line in p.read_text().splitlines():
            d = json.loads(line)
            out[d["key"]] = None if d["now"] is None else (d["now"][0], d["now"][1])
    return out


def _same(a: tuple[int, str], b: tuple[int, str]) -> bool:
    """Size must match; ETags only when neither is a multipart ETag (a copy re-chunks, so '-N' ETags differ)."""
    if a[0] != b[0]:
        return False
    return "-" in a[1] or "-" in b[1] or a[1] == b[1]


def _merge(src_rows, dst_rows):
    """Walk two key-sorted row streams together, yielding (key, src_row | None, dst_row | None)."""
    s, d = next(src_rows, None), next(dst_rows, None)
    while s is not None or d is not None:
        sk = s[0].encode() if s is not None else None
        dk = d[0].encode() if d is not None else None
        if d is None or (s is not None and sk < dk):
            yield s[0], s, None
            s = next(src_rows, None)
        elif s is None or dk < sk:
            yield d[0], None, d
            d = next(dst_rows, None)
        else:
            yield s[0], s, d
            s, d = next(src_rows, None), next(dst_rows, None)


def _sorted_check(rows):
    """Pass rows through, failing loudly if they are not in strictly increasing UTF-8 key order."""
    prev = None
    for r in rows:
        k = r[0].encode()
        if prev is not None and k <= prev:
            raise RuntimeError(f"listing out of order at {r[0]!r}; verify assumes S3 key order")
        prev = k
        yield r


def verify(work: Path, dst_base: str, us, threads: int, relist: bool = True) -> dict:
    if relist or not (work / "dst").exists():
        list_all(work, "dst", DST_BUCKET, dst_base, us, threads, relist=True)
    drift = read_drift(work)
    counts = {"ok": 0, "missing": 0, "mismatch": 0, "extra": 0, "drifted": len(drift)}
    with open(work / "problems.tsv", "w", encoding="utf-8") as out:
        for uid, *_ in us:
            src_path, dst_path = work / "src" / f"{uid}.tsv.gz", work / "dst" / f"{uid}.tsv.gz"
            # streamed: the biggest unit has >3M keys, which as a list OOM'd the 7.5 GB dev box
            ws, wd = {r[0]: r for r in weird_rows(src_path)}, {r[0]: r for r in weird_rows(dst_path)}
            pairs = itertools.chain(
                _merge(_sorted_check(iter_unit(src_path)), _sorted_check(iter_unit(dst_path))),
                ((k, ws.get(k), wd.get(k)) for k in ws.keys() | wd.keys()),
            )
            for key, s, d in pairs:
                if s is None:
                    counts["extra"] += 1
                    out.write(f"extra\t{json.dumps(key)}\t{d[1]}\t{d[2]}\n")
                    continue
                want = (s[1], s[2])
                if key in drift:
                    if drift[key] is None:  # vanished from the source before we copied it: nothing to hold
                        continue
                    want = drift[key]
                if d is None or not _same(want, (d[1], d[2])):
                    kind = "missing" if d is None else "mismatch"
                    counts[kind] += 1
                    out.write(f"{kind}\t{json.dumps(key)}\t{s[1]}\t{s[2]}\n")
                else:
                    counts["ok"] += 1
    log(f"verify: {counts} (details in {work / 'problems.tsv'})")
    return counts


def fixup(work: Path, dst_base: str, threads: int) -> None:
    """Recopy every missing/mismatched key with boto3's managed copy, recording source drift."""
    todo = []
    p = work / "problems.tsv"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            kind, key, size, etag = line.split("\t")
            if kind in ("missing", "mismatch"):
                todo.append((json.loads(key), (int(size), etag)))
    log(f"fixup: {len(todo):,} keys to recopy")
    if not todo:
        return
    s3 = s3_client(threads * 2)
    tc = TransferConfig(multipart_threshold=4 * 1024**3, multipart_chunksize=512 * 1024**2, max_concurrency=8)
    lock = threading.Lock()
    failed = []

    def one(key: str, listed: tuple[int, str]) -> None:
        try:
            head = s3.head_object(Bucket=SRC_BUCKET, Key=key)
        except s3.exceptions.ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                with lock, open(work / "drift.jsonl", "a") as f:
                    f.write(json.dumps({"key": key, "listed": listed, "now": None}) + "\n")
                return
            raise
        now = (head["ContentLength"], head["ETag"].strip('"'))
        s3.copy({"Bucket": SRC_BUCKET, "Key": key}, DST_BUCKET, dst_base + key, Config=tc)
        if not _same(listed, now):
            with lock, open(work / "drift.jsonl", "a") as f:
                f.write(json.dumps({"key": key, "listed": listed, "now": now}) + "\n")

    with ThreadPoolExecutor(threads) as ex:
        futs = {ex.submit(one, k, listed): k for k, listed in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001 - report and keep going; finalize refuses to publish
                failed.append(futs[fut])
                log(f"  fixup FAILED {futs[fut]!r}: {e}")
            if i % 1000 == 0 or i == len(todo):
                log(f"  fixup: {i:,}/{len(todo):,} ({len(failed)} failed)")


# ---------------------------------------------------------------------------------------------------------
# finalize


def concat(work: Path, which: str, us, header: str, out: Path) -> None:
    """One gzip of every unit file in key order, with a header row. Keys with tabs/newlines are appended at
    the end with the key column JSON-quoted."""
    weird = []
    with open(out, "wb") as f:
        f.write(gzip.compress(header.encode()))
        for uid, *_ in us:
            p = work / which / f"{uid}.tsv.gz"
            f.write(p.read_bytes())  # concatenated gzip members are one valid gzip stream
            weird += weird_rows(p)
        if weird:
            f.write(gzip.compress("".join(f"{json.dumps(k)}\t{s}\t{e}\t{lm}\n" for k, s, e, lm in weird).encode()))


def finalize(work: Path, dst_base: str, us, threads: int, started: str | None) -> None:
    counts = verify(work, dst_base, us, threads)
    if counts["missing"] or counts["mismatch"]:
        sys.exit(
            f"finalize: {counts['missing']} missing / {counts['mismatch']} mismatched keys; run `fixup` "
            f"(see {work / 'problems.tsv'}) and then `finalize` again"
        )
    src_n, src_b = totals(work, "src", us)
    dst_n, dst_b = totals(work, "dst", us)
    drift_p = work / "drift.jsonl"
    drift = [json.loads(line) for line in drift_p.read_text().splitlines()] if drift_p.exists() else []
    extra = [
        line.split("\t")[1]
        for line in (work / "problems.tsv").read_text(encoding="utf-8").splitlines()
        if line.startswith("extra\t")
    ]
    concat(work, "dst", us, "key\tsize\tetag\tcopied_at\n", work / "manifest.tsv.gz")
    concat(work, "src", us, "key\tsize\tetag\tsource_last_modified\n", work / "source_listing.tsv.gz")
    report = {
        "source_uri": f"s3://{SRC_BUCKET}/{{{','.join(TOPS)}}}",
        "destination_uri": f"s3://{DST_BUCKET}/{dst_base}",
        "started_at": started,
        "finalized_at": datetime.now(timezone.utc).isoformat(),
        "source_totals": {"objects": src_n, "bytes": src_b},
        "destination_totals": {"objects": dst_n, "bytes": dst_b},
        "verify": counts,
        "drift": drift,  # keys whose source changed or vanished between `list` and the copy
        "extra_in_destination": [json.loads(k) for k in extra],
        "s5cmd": subprocess.run([S5CMD, "version"], capture_output=True, text=True).stdout.strip(),
        "notes": "manifest keys are relative to destination_uri; ETags are unquoted, and a multipart source "
        "ETag ('-N') differs from its copy's, so those were verified by size only",
    }
    (work / "COPY_REPORT.json").write_text(json.dumps(report, indent=2))
    s3 = s3_client()
    snap_prefix = "lakeqa/"
    for name in ("manifest.tsv.gz", "source_listing.tsv.gz", "COPY_REPORT.json"):
        s3.upload_file(str(work / name), DST_BUCKET, snap_prefix + name)
        log(f"uploaded s3://{DST_BUCKET}/{snap_prefix}{name}")
    log(
        f"finalize: {dst_n:,} objects, {dst_b / 1e12:.3f} TB at s3://{DST_BUCKET}/{dst_base} "
        f"({len(drift)} drifted, {len(extra)} extra)"
    )


# ---------------------------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["list", "copy", "verify", "fixup", "finalize", "all"])
    ap.add_argument("--work", type=Path, default=Path("~/lakeqa_snapshot").expanduser(), help="local state dir")
    ap.add_argument("--list-threads", type=int, default=24)
    ap.add_argument("--procs", type=int, default=4, help="concurrent s5cmd copy processes")
    ap.add_argument("--workers", type=int, default=128, help="s5cmd --numworkers per copy process")
    ap.add_argument("--batch-keys", type=int, default=100_000, help="keys per s5cmd copy batch")
    ap.add_argument(
        "--reuse-dst-listing", action="store_true", help="verify: diff against the existing WORK/dst/ listing"
    )
    ap.add_argument("--fixup-threads", type=int, default=32)
    ap.add_argument("--unit-filter", default=None, help="regex on unit ids, smoke tests only (e.g. 'wikipedia-2[0-3]')")
    args = ap.parse_args()

    os.environ.setdefault("AWS_PROFILE", "mit")
    work: Path = args.work
    work.mkdir(parents=True, exist_ok=True)
    state_path = work / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    state.setdefault("started_at", datetime.now(timezone.utc).isoformat())
    state_path.write_text(json.dumps(state, indent=2))

    dst_base = "lakeqa/lake/"
    us = units()
    if args.unit_filter:
        us = [u for u in us if re.search(args.unit_filter, u[0])]
    log(f"{args.step}: -> s3://{DST_BUCKET}/{dst_base}, {len(us)} units, work dir {work}")

    if args.step in ("list", "all"):
        list_all(work, "src", SRC_BUCKET, "", us, args.list_threads, relist=False)
        n, b = totals(work, "src", us)
        log(f"source listing: {n:,} objects, {b / 1e12:.3f} TB")
    if args.step in ("copy", "all"):
        copy_all(work, dst_base, us, args.procs, args.workers, args.batch_keys)
    if args.step in ("verify", "all"):
        verify(work, dst_base, us, args.list_threads, relist=not args.reuse_dst_listing)
    if args.step in ("fixup", "all"):
        fixup(work, dst_base, args.fixup_threads)
    if args.step in ("finalize", "all"):
        if args.unit_filter:
            sys.exit("finalize publishes the manifest; refusing with --unit-filter")
        finalize(work, dst_base, us, args.list_threads, state["started_at"])


if __name__ == "__main__":
    main()
