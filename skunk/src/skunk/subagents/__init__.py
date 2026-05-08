from skunk.subagents.retrieve import RetrieveSubagent
from skunk.subagents.extract import ExtractSubagent
from skunk.subagents.read_visual import ReadVisualSubagent
from skunk.subagents.lookup_external import LookupExternalSubagent
from skunk.subagents.compute import ComputeSubagent
from skunk.subagents.format import FormatSubagent

SUBAGENT_REGISTRY = {
    "retrieve": RetrieveSubagent,
    "extract": ExtractSubagent,
    "read_visual": ReadVisualSubagent,
    "lookup_external": LookupExternalSubagent,
    "compute": ComputeSubagent,
    "format": FormatSubagent,
}

__all__ = [
    "RetrieveSubagent", "ExtractSubagent",
    "ReadVisualSubagent", "LookupExternalSubagent",
    "ComputeSubagent", "FormatSubagent",
    "SUBAGENT_REGISTRY",
]
