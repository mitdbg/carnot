"""Generate dev/test split files for every benchmark, co-located with the benchmark data at
qatfd/benchmarks/<benchmark>/<benchmark>_splits.json (each benchmark's resolved `splits_path`).

Each file is {"dev": [qids], "test": [qids]} consumed by the runner via `benchmarks.splits_path`
(the runner selects `experiments.split` from these explicit lists). Re-run to regenerate; the output
is fully deterministic. Regenerating also needs two source files that aren't required to RUN a
benchmark: benchmarks/qampari/train_data.jsonl (source of the dev sample) and
benchmarks/browsecomp-plus/karl_bcp_test_ids.json (the KARL test id list).

Split spec:
  officeqa         dev = first 25% of queries (CSV order),         test = last 75%
  browsecomp_plus  test = KARL's calibrated subset,                dev = 50 sampled (seed 0) from the rest
  trec_biogen      shuffle(seed 0): dev = first 10,                test = last 30
  financebench     shuffle(seed 0): dev = first 50,                test = last 100
  qampari          test = ALL 1000 test_data qids (KARL's set),    dev = 50 sampled (seed 0) from train_data
  freshstack       dev = all laravel queries,                      test = all langchain queries

"shuffle(seed 0)" = sort the qids for a stable base order, then random.Random(0).shuffle, then slice —
so the dev/test membership depends only on the seed, not on file/load order.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from qatfd.env import load_env

load_env()  # populate skunk/.env before importing skunk-backed modules

from hydra import compose, initialize_config_dir  # noqa: E402
from qatfd.config import benchmark_config_factory  # noqa: E402
from qatfd.paths import resolve_under_benchmarks  # noqa: E402
from qatfd.registry import build_benchmark  # noqa: E402

CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")


def _benchmark(overrides: list[str]):
    cfg = compose(config_name="config", overrides=overrides)
    return build_benchmark(benchmark_config_factory(cfg))


def _ordered_qids(bench) -> list[str]:
    return [q.qid for q in bench.load_questions()]


def _shuffle_slice(qids: list[str], seed: int, n_dev: int) -> tuple[list[str], list[str]]:
    ordered = sorted(qids)
    random.Random(seed).shuffle(ordered)
    return ordered[:n_dev], ordered[n_dev:]


def _read_jsonl_by_qid(path) -> dict[str, str]:
    """qid -> raw JSONL line (kept verbatim), for benchmarks whose split is drawn from a file the
    benchmark loader doesn't read (e.g. QAMPARI's train_data, source of the dev extract)."""
    by_qid: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                by_qid[str(json.loads(line)["qid"])] = line if line.endswith("\n") else line + "\n"
    return by_qid


def _write_jsonl_extract(path, lines: list[str]) -> None:
    """Materialize a small subset of raw JSONL records (e.g. the 50 QAMPARI dev questions) so a
    benchmark's dev_questions_path can be loaded alongside its main question file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(lines)
    print(f"{'':16s} wrote {len(lines)} dev records -> {path}")


def _write(path, name: str, dev: list[str], test: list[str]) -> None:
    """Write {"dev", "test"} to `path` (the benchmark's resolved, co-located splits_path)."""
    dev_s, test_s = sorted(set(dev)), sorted(set(test))
    overlap = set(dev_s) & set(test_s)
    assert not overlap, f"{name}: dev/test overlap ({len(overlap)}): {sorted(overlap)[:5]}"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"dev": dev_s, "test": test_s}, f, indent=2)
    print(f"{name:16s} dev={len(dev_s):4d} test={len(test_s):4d} -> {path}")


def main() -> None:
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        # Each benchmark's split file is written to its resolved, co-located `splits_path`
        # (qatfd/benchmarks/<benchmark>/<benchmark>_splits.json).

        # officeqa: first 25% (CSV order) is dev, last 75% is test (not shuffled).
        bench = _benchmark(["benchmarks=officeqa"])
        q = _ordered_qids(bench)
        k = round(0.25 * len(q))
        print(f"[officeqa] n={len(q)}, dev=first {k} (25%)")
        _write(bench.config.splits_path, "officeqa", q[:k], q[k:])

        # shuffle(seed 0) then slice.
        for name, n_dev in [("trec_biogen", 10), ("financebench", 50)]:
            bench = _benchmark([f"benchmarks={name}"])
            q = _ordered_qids(bench)
            dev, test = _shuffle_slice(q, 0, n_dev)
            _write(bench.config.splits_path, name, dev, test)

        # qampari: test = ALL 1000 test_data qids (KARL's exact eval set — confirmed by the appendix
        # query "What did James B. Longacre design?" living only in test_data.jsonl). dev = 50 sampled
        # (seed 0) from train_data.jsonl, DISJOINT from test (different source file), so dev never
        # leaks into the KARL test set. We also materialize the 50 dev records into a small extract
        # (dev_questions_path) so the runner can load them alongside the test file.
        bench = _benchmark(["benchmarks=qampari"])
        train_recs = _read_jsonl_by_qid(resolve_under_benchmarks("qampari/train_data.jsonl"))
        dev, _ = _shuffle_slice(list(train_recs), 0, 50)
        # dev_questions_path is already resolved (absolute) by QampariBenchmark.__init__.
        _write_jsonl_extract(bench.config.dev_questions_path, [train_recs[q] for q in dev])
        # test = every qid in test_data.jsonl (read directly, so the just-written dev extract can't leak in).
        test = list(_read_jsonl_by_qid(bench.config.questions_path))
        _write(bench.config.splits_path, "qampari", dev, test)

        # browsecomp_plus: test = KARL's subset; dev = 50 sampled (seed 0) from the DISJOINT rest.
        bench = _benchmark(["benchmarks=browsecomp_plus"])
        all_q = set(_ordered_qids(bench))
        # KARL's calibrated test subset — the source of the BCP test split. Referenced directly here
        # (this generator is its only consumer; it's no longer a BrowseCompPlusConfig field).
        with open(resolve_under_benchmarks("browsecomp-plus/karl_bcp_test_ids.json")) as f:
            karl = {str(x) for x in json.load(f)["query_ids"]}
        test = sorted(all_q & karl)
        complement = sorted(all_q - karl)
        random.Random(0).shuffle(complement)
        print(f"[browsecomp_plus] n={len(all_q)}, KARL test={len(test)}, dev=50 from {len(complement)} non-test")
        _write(bench.config.splits_path, "browsecomp_plus", complement[:50], test)

        # freshstack: dev = all laravel queries, test = all langchain queries (separate corpora). Both
        # topics resolve the SAME topic-independent splits_path (freshstack/freshstack_splits.json).
        dev_bench = _benchmark(["benchmarks=freshstack", "benchmarks.topic=laravel"])
        dev = _ordered_qids(dev_bench)
        test = _ordered_qids(_benchmark(["benchmarks=freshstack", "benchmarks.topic=langchain"]))
        _write(dev_bench.config.splits_path, "freshstack", dev, test)


if __name__ == "__main__":
    main()
