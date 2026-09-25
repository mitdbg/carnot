"""The bootstrap / enrich upper-bound sweep as cells: a port of scripts/run_bootstrap_enrich_upper_bound.sh
(its cell table, the fixed search-agent tool configuration, and the X x seed x cell enumeration). One cell
= one (X, cell, seed) = one invocation of scripts/bootstrap_enrich_upper_bound.py answering the same X dev
questions `passes` times in one long-lived system, so a complete run has X * passes rows.

Differences from the bash driver, all consequences of every pod having its own pristine store: no server
management, no corpus-metadata reset, no orphan cleanup, and `+ub.wipe_collections=false` (there is nothing
to wipe). `keep_collections` is still passed so the driver's own guard sees the same list.

    python -m qatfd.k8s plan --sweep bootstrap_enrich_ub --xs 1 5 10 --seeds 0 1 2
    python -m qatfd.k8s plan --sweep bootstrap_enrich_ub --qids UID0001 --xs 1 --run-prefix smoke
"""

from __future__ import annotations

import argparse

from qatfd.k8s.benchmarks import benchmark_data
from qatfd.k8s.cells import Cell

NAME = "bootstrap_enrich_ub"
# results/<benchmark>/<SYSTEM_DIR>/ — scripts/bootstrap_enrich_upper_bound.py's SYSTEM_DIR
SYSTEM_DIR = "search_agent_ub"

# cell -> (trajectory_working_sets, working_set_collection_off, enrich_working_sets, hide_and_clear_working_sets);
# the table in the bash driver's header. exp1's collection_off is the BASELINE_COLLECTION_OFF knob.
CELLS: dict[str, tuple[str, str | None, str, str]] = {
    "exp1_baseline": ("false", None, "null", "true"),
    "exp2_ws": ("true", "false", "null", "false"),
    "exp3_bs": ("false", "false", "before", "true"),
    "exp4_bs_ws": ("true", "false", "before", "true"),
    "exp5_en": ("false", "false", "after", "true"),
    "exp6_en_ws": ("true", "false", "after", "true"),
    "exp7_bs_en": ("false", "false", "both", "true"),
    "exp8_bs_en_ws": ("true", "false", "both", "true"),
}
DEFAULT_CELLS = ["exp1_baseline", "exp2_ws", "exp3_bs", "exp5_en", "exp7_bs_en"]


def add_arguments(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group(f"{NAME} sweep")
    g.add_argument("--benchmark", default="officeqa")
    g.add_argument("--split", default="dev")
    g.add_argument("--xs", type=int, nargs="+", default=[1, 5, 10], help="questions per pass (one cell per X)")
    g.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="dev-split permutation seeds")
    g.add_argument("--cells", nargs="+", default=DEFAULT_CELLS, choices=sorted(CELLS), metavar="CELL",
                   help=f"cells from the table (default: {' '.join(DEFAULT_CELLS)})")
    g.add_argument("--passes", type=int, default=2)
    g.add_argument("--enrich-batch", type=int, default=None, help="EnrichAgent cadence in questions; default X")
    g.add_argument("--qids", nargs="+", default=None, help="explicit qids (overrides X / seed; X is a label only)")
    g.add_argument("--model", default="openai/gpt-5.6-luna", help="search agent model")
    g.add_argument("--compute-model", default="openai/gpt-5.6-terra")
    g.add_argument("--agent-model", default="openai/gpt-5.6-terra", help="Bootstrap / Enrich agents' own model")
    g.add_argument("--map-model", default="openai/gpt-5.6-luna", help="semantic_map judge model")
    g.add_argument("--base-collection", default=None, help="corpus collection (default: the benchmark's v1 copy)")
    g.add_argument("--keep-collections", nargs="*", default=["officeqa-qwen-8b", "officeqa-qwen-8b-v1"])
    g.add_argument("--baseline-collection-off", default="true", choices=["true", "false"],
                   help="exp1's working_set_collection_off (see the bash driver's header)")
    g.add_argument("--run-prefix", default="ub", help="labels are <prefix>_x<X>_<cell>_s<seed>")
    g.add_argument("--extra-overrides", nargs="*", default=[], help="hydra overrides appended to every cell")
    g.add_argument("--with-pdfs", action="store_true", help="pull the corpus PDFs and keep pdf_dir (the figure tool)")


def cells(args: argparse.Namespace) -> list[Cell]:
    bd = benchmark_data(args.benchmark, with_pdfs=args.with_pdfs)
    base_collection = args.base_collection or bd.collection
    keep = ",".join(sorted(set([base_collection, *args.keep_collections])))
    system_overrides = [
        "systems=search_agent",
        f"inference.llm_model={args.model}",
        f"systems.compute.llm_model={args.compute_model}",
        "systems.retrieve.include_search_corpus=true",
        "systems.retrieve.include_grep_corpus=true",
        "systems.retrieve.include_semantic_filter=false",
        "systems.retrieve.id_tracking_off=false",
        "systems.retrieve.fetch_related_working_sets=true",
        f"systems.retrieve.bootstrap_config.llm_model={args.agent_model}",
        f"systems.retrieve.bootstrap_config.semantic_map_llm_model={args.map_model}",
        f"systems.retrieve.enrich_config.llm_model={args.agent_model}",
        f"systems.retrieve.enrich_config.semantic_map_llm_model={args.map_model}",
        f"benchmarks={args.benchmark}",
        f"benchmarks.collection_name={base_collection}",
        f"experiments.split={args.split}",
        "experiments.run_mode=sequential",
        "experiments.workers=1",
    ]
    if not args.with_pdfs:
        system_overrides.append("benchmarks.pdf_dir=null")

    out: list[Cell] = []
    for x in args.xs:
        for seed in args.seeds:
            for cell in args.cells:
                traj, coll_off, mode, hide = CELLS[cell]
                if coll_off is None:
                    coll_off = args.baseline_collection_off
                label = f"{args.run_prefix}_x{x}_{cell}_s{seed}"
                batch = args.enrich_batch or x
                n_q = len(args.qids) if args.qids else x
                overrides = [
                    *system_overrides,
                    f"systems.retrieve.working_set_collection_off={coll_off}",
                    f"systems.retrieve.hide_and_clear_working_sets={hide}",
                    f"systems.retrieve.enrich_working_sets={mode}",
                    f"systems.retrieve.enrich_query_batch_size={batch}",
                    f"experiments.run_name={label}",
                    f"+ub.label={label}",
                    f"+ub.num_queries={x}",
                    f"+ub.sample_seed={seed}",
                    f"+ub.passes={args.passes}",
                    f"+ub.trajectory_working_sets={traj}",
                    "+ub.wipe_collections=false",
                    f"+ub.keep_collections=[{keep}]",
                ]
                if args.qids:
                    overrides.append(f"experiments.qids=[{','.join(args.qids)}]")
                overrides.extend(args.extra_overrides)
                out.append(Cell(
                    label=label, benchmark=args.benchmark, system_dir=SYSTEM_DIR,
                    collection=base_collection, store_prefix=bd.store_prefix, data=list(bd.data),
                    argv=["python", "scripts/bootstrap_enrich_upper_bound.py", *overrides],
                    expected_rows=n_q * args.passes,
                    meta={"x": x, "seed": seed, "cell": cell},
                ))
    return out
