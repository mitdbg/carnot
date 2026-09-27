"""Knobs shared by the bootstrap / enrich sweeps: which model each agent runs on, the OpenRouter provider
pin, and reasoning-off per agent."""

from __future__ import annotations

import argparse

# agents whose own LLM steps `--disable-reasoning` can switch thinking off on -> their config path
REASONING_AGENTS: dict[str, str] = {
    "search": "systems.retrieve",
    "compute": "systems.compute",
    "bootstrap": "systems.retrieve.bootstrap_config",
    "enrich": "systems.retrieve.enrich_config",
}


def add_model_arguments(g) -> None:
    g.add_argument("--model", default="openai/gpt-5.6-luna", help="search agent model")
    g.add_argument("--compute-model", default="openai/gpt-5.6-terra")
    g.add_argument("--agent-model", default="openai/gpt-5.6-terra", help="Bootstrap / Enrich agents' own model")
    g.add_argument("--map-model", default="openai/gpt-5.6-luna", help="semantic_map judge model")
    g.add_argument("--provider", "--providers", dest="provider", default=None,
                   help="OpenRouter provider pin for every LLM call (inference.llm_provider_order): a comma-separated "
                        "ORDER of slugs, tried first-to-last, never anyone outside the list, e.g. parasail,akashml,reka,venice")
    g.add_argument("--disable-reasoning", nargs="*", default=None, choices=sorted(REASONING_AGENTS), metavar="AGENT",
                   help="no thinking tokens on these agents' own steps (bare flag = all of "
                        f"{' '.join(sorted(REASONING_AGENTS))}; the semantic_map judge already runs with reasoning "
                        "disabled). Qwen3-style models only: endpoints that mandate reasoning reject it")
    g.add_argument("--enrich-max-previous-queries", type=int, default=None,
                   help="EnrichAgent's query-workload window (enrich_config.max_previous_queries; config default 20)")


def model_overrides(args: argparse.Namespace) -> list[str]:
    """Hydra overrides for the model / provider / reasoning knobs above."""
    out = [
        f"inference.llm_model={args.model}",
        f"systems.compute.llm_model={args.compute_model}",
        f"systems.retrieve.bootstrap_config.llm_model={args.agent_model}",
        f"systems.retrieve.bootstrap_config.semantic_map_llm_model={args.map_model}",
        f"systems.retrieve.enrich_config.llm_model={args.agent_model}",
        f"systems.retrieve.enrich_config.semantic_map_llm_model={args.map_model}",
    ]
    if args.provider:
        order = ",".join(p.strip() for p in args.provider.split(",") if p.strip())
        out.append(f"inference.llm_provider_order=[{order}]")
    if args.disable_reasoning is not None:  # [] (bare flag) means every agent
        for agent in args.disable_reasoning or sorted(REASONING_AGENTS):
            out.append(f"{REASONING_AGENTS[agent]}.disable_reasoning=true")
    if args.enrich_max_previous_queries is not None:
        out.append(f"systems.retrieve.enrich_config.max_previous_queries={args.enrich_max_previous_queries}")
    return out
