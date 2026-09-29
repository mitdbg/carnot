"""The Codex ablation on the cluster: the stock runner with `systems=codex`, one pod per (scenario, seed), on the
v1 (no-metadata) corpus copy. The scenario table is scripts/run_codex_ablation.sh's, minus `par`: questions run
in arrival order in every cell, so the isolation baseline is `seq_iso` (a fresh ephemeral session per question,
no shell, no resume) and the Bootstrap / Enrich cells differ from it only by the collection agents.

    scenario           isolation  shell  resume  collection agents
    seq_iso            yes        no     no      none
    seq_iso_bs         yes        no     no      Bootstrap before the first question
    seq_iso_en         yes        no     no      Enrich after every --enrich-batch questions
    seq_iso_bsen       yes        no     no      both
    seq_resume         no         no     yes     none   (one codex thread carried across all questions)
    seq_resume_c40     no         no     yes     none   (same, auto-compact at --c40-token-limit)
    seq_shell          no         yes    no      none   (persistent workspace + AGENTS.md)
    seq_shell_resume   no         yes    yes     none

Every pod has its own pristine store copy, so there is nothing to wipe or reset between runs (the local driver's
seq_iso* wipe / metadata reset is a shared-server problem). Every cell runs inside the codex sandbox (values
codexSandbox, on unless --no-sandbox): codex runs as an unprivileged uid that cannot read /data (gold answers, raw
corpus, chroma store) or /results, and can only reach the MCP server and an egress proxy that tunnels to the model
provider alone; see deploy_exp/README.md "Codex cells". The corpus PDFs are pulled (the MCP server's view_figure
tool) unless --no-pdfs.

Labels are codex_<scenario>_<variant>_<split>_s<seed> (seq_resume_c40 -> codex_seq_resume_<variant>c40_...), i.e.
the names scripts/plot_codex_ablation.py parses. --variant defaults to `v1r2` (v1 corpus, round 2: the collection-
aware MCP tools + view_figure + the sandbox) so these runs never supersede, or silently mix with, the 2026-09-22
local `v1` runs, which used the previous tool surface.

    python -m qatfd.k8s plan --sweep codex_ablation --seeds 0 1 2
    python -m qatfd.k8s submit codex-v1r2 --sweep codex_ablation --image-tag <sha> --watch
"""

from __future__ import annotations

import argparse
import re

from qatfd.k8s.benchmarks import benchmark_data
from qatfd.k8s.cells import Cell

NAME = "codex_ablation"
SYSTEM_DIR = "codex"   # the stock runner's results/<benchmark>/<systems.name>/

# must agree with the chart's codexSandbox block (deploy_exp/helm/experiments/values.yaml)
SANDBOX_UID = 10001
SANDBOX_PROXY_PORT = 3128

ISO = [
    "experiments.run_mode=sequential", "experiments.workers=1",
    "systems.isolation=true", "systems.codex_shell=false", "systems.session_resume=false",
]
# scenario -> (overrides, collection agents on?, label variant suffix)
SCENARIOS: dict[str, tuple[list[str], bool, str]] = {
    "seq_iso": ([*ISO, "systems.enrich_working_sets=null"], False, ""),
    "seq_iso_bs": ([*ISO, "systems.enrich_working_sets=before"], True, ""),
    "seq_iso_en": ([*ISO, "systems.enrich_working_sets=after"], True, ""),
    "seq_iso_bsen": ([*ISO, "systems.enrich_working_sets=both"], True, ""),
    "seq_resume": (["experiments.run_mode=sequential", "systems.isolation=false", "systems.codex_shell=false",
                    "systems.session_resume=true"], False, ""),
    "seq_resume_c40": (["experiments.run_mode=sequential", "systems.isolation=false", "systems.codex_shell=false",
                        "systems.session_resume=true"], False, "c40"),
    "seq_shell": (["experiments.run_mode=sequential", "systems.isolation=false", "systems.codex_shell=true",
                   "systems.session_resume=false"], False, ""),
    "seq_shell_resume": (["experiments.run_mode=sequential", "systems.isolation=false", "systems.codex_shell=true",
                          "systems.session_resume=true"], False, ""),
}
DEFAULT_SCENARIOS = list(SCENARIOS)


def add_arguments(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group(f"{NAME} sweep")
    g.add_argument("--benchmark", default="officeqa")
    g.add_argument("--split", default="dev")
    g.add_argument("--num-questions", type=int, default=33,
                   help="questions in the split (only used to decide whether a run dir on S3 is complete)")
    g.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="experiments.shuffle_seed: question order")
    g.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS, choices=list(SCENARIOS), metavar="SCENARIO",
                   help=f"default: {' '.join(DEFAULT_SCENARIOS)}")
    g.add_argument("--variant", default="v1r2", help="label variant tag, [a-z0-9]+ (see the module docstring)")
    g.add_argument("--base-collection", default=None, help="corpus collection (default: the benchmark's v1 copy)")
    g.add_argument("--model", default="openai/gpt-5.6-luna",
                   help="inference.llm_model: the fallback for any agent model left null (codex's own model is its config.toml's)")
    g.add_argument("--agent-model", default="openai/gpt-5.6-terra", help="Bootstrap / Enrich agents' own model")
    g.add_argument("--map-model", default="openai/gpt-5.6-luna", help="Bootstrap / Enrich semantic_map judge model")
    g.add_argument("--enrich-batch", type=int, default=5, help="EnrichAgent cadence in questions")
    g.add_argument("--enrich-max-previous-queries", type=int, default=50, help="EnrichAgent workload window")
    g.add_argument("--c40-token-limit", type=int, default=420000,
                   help="seq_resume_c40's systems.auto_compact_token_limit (40%% of the 1.05M context window)")
    g.add_argument("--no-pdfs", action="store_true", help="skip the corpus PDFs (benchmarks.pdf_dir=null: no view_figure)")
    g.add_argument("--no-sandbox", action="store_true",
                   help="run codex as root with the pod's full view (NOT for shell scenarios: they can read the gold answers)")
    g.add_argument("--extra-overrides", nargs="*", default=[], help="hydra overrides appended to every cell")


def chart_values(args: argparse.Namespace) -> dict:
    """Release-level chart values this sweep needs (merged into the values file by `submit`)."""
    return {"codexSandbox": {"enabled": not args.no_sandbox, "uid": SANDBOX_UID, "proxyPort": SANDBOX_PROXY_PORT}}


def cells(args: argparse.Namespace) -> list[Cell]:
    if not re.fullmatch(r"[a-z0-9]+", args.variant):
        raise SystemExit(f"--variant {args.variant!r} must match [a-z0-9]+ (the plot scripts parse it out of the label)")
    if args.no_sandbox and any(SCENARIOS[s][0].count("systems.codex_shell=true") for s in args.scenarios):
        raise SystemExit("--no-sandbox with a shell scenario: codex could read the gold answers; drop one of the two")
    bd = benchmark_data(args.benchmark, with_pdfs=not args.no_pdfs)
    base_collection = args.base_collection or bd.collection

    common = [
        "systems=codex",
        f"benchmarks={args.benchmark}",
        f"benchmarks.collection_name={base_collection}",
        f"experiments.split={args.split}",
        f"inference.llm_model={args.model}",
    ]
    if args.no_pdfs:
        common.append("benchmarks.pdf_dir=null")
    if not args.no_sandbox:
        common += [
            f"systems.sandbox_uid={SANDBOX_UID}",
            "systems.codex_home_dir=/codex/home",
            "systems.codex_scratch_dir=/codex/work",
            f"systems.egress_proxy=http://127.0.0.1:{SANDBOX_PROXY_PORT}",
        ]
    collection_agents = [
        f"systems.bootstrap_config.llm_model={args.agent_model}",
        f"systems.bootstrap_config.semantic_map_llm_model={args.map_model}",
        f"systems.enrich_config.llm_model={args.agent_model}",
        f"systems.enrich_config.semantic_map_llm_model={args.map_model}",
        f"systems.enrich_config.max_previous_queries={args.enrich_max_previous_queries}",
        f"systems.enrich_query_batch_size={args.enrich_batch}",
    ]

    out: list[Cell] = []
    for seed in args.seeds:
        for scenario in args.scenarios:
            overrides, agents, suffix = SCENARIOS[scenario]
            base_scenario = scenario.removesuffix("_c40")
            label = f"codex_{base_scenario}_{args.variant}{suffix}_{args.split}_s{seed}"
            argv_overrides = [*common, *overrides]
            if agents:
                argv_overrides += collection_agents
            if suffix == "c40":
                argv_overrides.append(f"systems.auto_compact_token_limit={args.c40_token_limit}")
            argv_overrides += [f"experiments.shuffle_seed={seed}", f"experiments.run_name={label}", *args.extra_overrides]
            out.append(Cell(
                label=label, benchmark=args.benchmark, system_dir=SYSTEM_DIR,
                collection=base_collection, store_prefix=bd.store_prefix, data=list(bd.data),
                argv=["python", "-m", "qatfd.runner", *argv_overrides],
                expected_rows=args.num_questions,
                meta={"seed": seed, "scenario": scenario},
            ))
    return out
