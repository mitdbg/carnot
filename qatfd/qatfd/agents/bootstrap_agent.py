from qatfd.agents.collection_agent import CollectionAgent
from qatfd.prompts import load_qatfd_prompts


class BootstrapAgent(CollectionAgent):
    """Creates an initial set of collections (working sets) over a corpus before any question is
    asked. Tools, rendering, and the agent loop live in `CollectionAgent`; only the prompts differ."""

    _PROMPTS = load_qatfd_prompts("bootstrap_agent")
    _AGENT_TYPE = "BootstrapAgent"
