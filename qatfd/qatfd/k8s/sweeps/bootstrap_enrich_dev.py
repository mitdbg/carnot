"""Bootstrap / enrich on the real benchmark: the STOCK runner (`python -m qatfd.runner`), one pass over a split
(default: the officeqa dev set), sequentially, in one long-lived system per cell. One cell = one (system cell,
seed); the seed is `experiments.shuffle_seed`, i.e. the order the questions arrive in, which is what the
Enrich agent's workload window sees. Results land in results/<benchmark>/search_agent/<prefix>_<cell>_s<seed>_<ts>/
with the stock report.csv (one row per question), so every existing eval / plot / trace-viewer path applies.

The cell table is the upper-bound sweep's (same five systems, same knobs); the one knob it relies on that the
stock configs did not have, `systems.retrieve.trajectory_working_sets`, is now a stock option.

    python -m qatfd.k8s plan --sweep bootstrap_enrich_dev --seeds 0 1 2 --enrich-batch 5
"""

from __future__ import annotations

import argparse

from qatfd.k8s.benchmarks import benchmark_data
from qatfd.k8s.cells import Cell
from qatfd.k8s.sweeps._common import add_model_arguments, model_overrides
from qatfd.k8s.sweeps.bootstrap_enrich_ub import CELLS, DEFAULT_CELLS

NAME = "bootstrap_enrich_dev"
SYSTEM_DIR = "search_agent"   # the stock runner's results/<benchmark>/<systems.name>/


def add_arguments(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group(f"{NAME} sweep")
    g.add_argument("--benchmark", default="officeqa")
    g.add_argument("--split", default="dev")
    g.add_argument("--num-questions", type=int, default=33,
                   help="questions in the split (only used to decide whether a run dir on S3 is complete)")
    g.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="experiments.shuffle_seed: question order")
    g.add_argument("--cells", nargs="+", default=DEFAULT_CELLS, choices=sorted(CELLS), metavar="CELL",
                   help=f"cells from the upper-bound table (default: {' '.join(DEFAULT_CELLS)})")
    g.add_argument("--enrich-batch", type=int, default=5, help="EnrichAgent cadence in questions")
    add_model_arguments(g)
    g.add_argument("--base-collection", default=None, help="corpus collection (default: the benchmark's v1 copy)")
    g.add_argument("--baseline-collection-off", default="true", choices=["true", "false"],
                   help="exp1's working_set_collection_off")
    g.add_argument("--run-prefix", default="dev", help="labels are <prefix>_<cell>_s<seed>")
    g.add_argument("--extra-overrides", nargs="*", default=[], help="hydra overrides appended to every cell")
    g.add_argument("--with-pdfs", action="store_true", help="pull the corpus PDFs and keep pdf_dir (the figure tool)")


def cells(args: argparse.Namespace) -> list[Cell]:
    bd = benchmark_data(args.benchmark, with_pdfs=args.with_pdfs)
    base_collection = args.base_collection or bd.collection
    system_overrides = [
        "systems=search_agent",
        *model_overrides(args),
        "systems.retrieve.include_search_corpus=true",
        "systems.retrieve.include_grep_corpus=true",
        "systems.retrieve.include_semantic_filter=false",
        "systems.retrieve.id_tracking_off=false",
        "systems.retrieve.fetch_related_working_sets=true",
        f"benchmarks={args.benchmark}",
        f"benchmarks.collection_name={base_collection}",
        f"experiments.split={args.split}",
        "experiments.run_mode=sequential",   # the Enrich cadence and the workload window need arrival order
        "experiments.workers=1",
    ]
    if not args.with_pdfs:
        system_overrides.append("benchmarks.pdf_dir=null")

    out: list[Cell] = []
    for seed in args.seeds:
        for cell in args.cells:
            traj, coll_off, mode, hide = CELLS[cell]
            if coll_off is None:
                coll_off = args.baseline_collection_off
            label = f"{args.run_prefix}_{cell}_s{seed}"
            overrides = [
                *system_overrides,
                f"systems.retrieve.trajectory_working_sets={traj}",
                f"systems.retrieve.working_set_collection_off={coll_off}",
                f"systems.retrieve.hide_and_clear_working_sets={hide}",
                f"systems.retrieve.enrich_working_sets={mode}",
                f"systems.retrieve.enrich_query_batch_size={args.enrich_batch}",
                f"experiments.shuffle_seed={seed}",
                f"experiments.run_name={label}",
                *args.extra_overrides,
            ]
            out.append(Cell(
                label=label, benchmark=args.benchmark, system_dir=SYSTEM_DIR,
                collection=base_collection, store_prefix=bd.store_prefix, data=list(bd.data),
                argv=["python", "-m", "qatfd.runner", *overrides],
                expected_rows=args.num_questions,
                meta={"seed": seed, "cell": cell},
            ))
    return out
