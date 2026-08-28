from __future__ import annotations

from jinja2 import Environment, StrictUndefined

from skunk.agents.lookup_agent.lookup_tools import resolve_lookup_tools
from skunk.agents.multi_turn_agent import MultiTurnAgent, StepOutput, parse_step
from skunk.config import LookupAgentConfig
from skunk.errors import ParseError
from skunk.prompts import load_prompts
from skunk.sandbox.local_python_executor import BASE_BUILTIN_MODULES

_ENV = Environment(
    autoescape=False, keep_trailing_newline=True, undefined=StrictUndefined
)
_PROMPTS = load_prompts("lookup_agent")


class LookupAgent(MultiTurnAgent):

    @staticmethod
    def parse_step(text: str) -> StepOutput:
        """Apply the basic answer parser and validate that a final answer has a `description` and `value` field."""
        step_output = parse_step(text)

        if step_output.is_final and ("description" not in step_output.result or "value" not in step_output.result):
            raise ParseError(
                detail='Final answer must be a JSON object with "description" and "value" keys, e.g. {"description": "...", "value": ...}.'
            )

        return step_output

    # TODO: should LookupAgent override _blocks_from_output()?
    def __init__(self, config: LookupAgentConfig, *, agent_id: str | None = None, additional_notes: str | None = None):
        # override the default agent_id if one is provided
        config.agent_id = config.agent_id if agent_id is None else agent_id

        # resolve the set of lookup tools based on the config
        tools = resolve_lookup_tools(config)

        # construct the system prompt
        system_prompt_template = _PROMPTS["system_prompt"]
        system_prompt = _ENV.from_string(system_prompt_template).render(
            authorized_imports=list(set(BASE_BUILTIN_MODULES) | set(config.authorized_imports)),
            tools="\n\n".join(t.doc for t in tools),
            max_steps=config.max_steps,
            cost_budget=config.cost_budget,
            latency_budget=config.latency_budget,
            additional_notes=additional_notes,
        )

        # construct the terminal prompt
        terminal_prompt_template = _PROMPTS["terminal_prompt"]
        terminal_prompt = _ENV.from_string(terminal_prompt_template).render()

        # construct the agent
        super().__init__(config, tools, system_prompt=system_prompt, terminal_prompt=terminal_prompt, parse=LookupAgent.parse_step)
