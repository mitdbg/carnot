from qatfd.agents.collection_agent import CollectionAgent
from qatfd.prompts import load_qatfd_prompts


class EnrichAgent(CollectionAgent):
    """Curates the existing collections (working sets) given the observed query workload. Tools,
    rendering, and the agent loop live in `CollectionAgent`; only the prompts differ."""

    _PROMPTS = load_qatfd_prompts("enrich_agent")
    _AGENT_TYPE = "EnrichAgent"
