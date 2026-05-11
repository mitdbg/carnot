from skunk.subagents import compute, extract, lookup_external, retrieve
from skunk.subagents.base import SubagentFn

SUBAGENT_REGISTRY: dict[str, SubagentFn] = {
    "retrieve": retrieve.run,
    "extract": extract.run,
    "lookup_external": lookup_external.run,
    "compute": compute.run,
}
