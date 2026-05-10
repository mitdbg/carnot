from skunk.subagents import retrieve, extract, read_visual, lookup_external, compute

SUBAGENT_REGISTRY = {
    "retrieve": retrieve.run,
    "extract": extract.run,
    "read_visual": read_visual.run,
    "lookup_external": lookup_external.run,
    "compute": compute.run,
}
