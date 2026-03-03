import time
from concurrent.futures import ThreadPoolExecutor, wait
from importlib import resources

import yaml

from carnot.agents.base import populate_template
from carnot.agents.memory import (
    ActionStep,
    AgentMemory,
    SemAggOperatorStep,
    SystemPromptStep,
    Timing,
)
from carnot.agents.models import ChatMessage, LiteLLMModel
from carnot.agents.monitoring import AgentLogger, LogLevel
from carnot.agents.utils import (
    AgentError,
    AgentGenerationError,
    AgentParsingError,
    parse_json_output,
)
from carnot.data.dataset import Dataset


class SemAggOperator:
    """Semantic aggregation operator — reduces a dataset to a single output row.

    All items are sent to the LLM in a single call.  The LLM produces JSON
    with the requested ``agg_fields``.  Missing fields default to ``None``.

    Representation invariant:
        - ``agg_fields`` is a non-empty list of dicts, each with a ``'name'``
          key.
        - ``output_tags`` is ``["```json", "```"]``.
        - ``max_steps >= 1``.

    Abstraction function:
        An instance of this class is a callable that, given a dataset, returns
        a new single-item dataset containing the LLM-produced aggregation of
        all input items.
    """
    def __init__(self, task: str, agg_fields: list[dict], output_dataset_id: str, model_id: str, llm_config: dict, max_workers: int, max_steps: int = 3):
        self.task = task
        self.output_dataset_id = output_dataset_id
        self.agg_fields = agg_fields
        self.model = LiteLLMModel(model_id=model_id, api_key=llm_config.get("OPENAI_API_KEY"))
        self.max_workers = max_workers
        self.prompt_templates = yaml.safe_load(
            resources.files("carnot.agents.prompts").joinpath("sem_agg.yaml").read_text()
        )
        self.memory = AgentMemory("")
        self.logger = AgentLogger(level=LogLevel.INFO)
        self.output_tags = ["```json", "```"]
        self.max_steps = max_steps

    def _finalize_step(self, memory_step: ActionStep):
        memory_step.timing.end_time = time.time()

    def write_memory_to_messages(
        self,
        memory: AgentMemory,
        summary_mode: bool = False,
    ) -> list[ChatMessage]:
        """
        Reads past llm_outputs, actions, and observations or errors from the memory into a series of messages
        that can be used as input to the LLM. Adds a number of keywords (such as PLAN, error, etc) to help
        the LLM.
        """
        messages = memory.system_prompt.to_messages(summary_mode=summary_mode)
        for memory_step in memory.steps:
            messages.extend(memory_step.to_messages(summary_mode=summary_mode))
        return messages

    def _sem_agg(self, items: list[dict], system_prompt: str) -> dict | None:
        """Aggregate all *items* via a single LLM call.

        Requires:
            - *items* is a non-empty list of dicts.
            - *system_prompt* is a pre-populated prompt string.

        Returns:
            A single dict with keys from ``agg_fields``.  Missing fields
            are set to ``None``.

        Raises:
            AgentGenerationError: If the LLM call itself fails.
        """
        memory = AgentMemory("")
        memory.system_prompt = SystemPromptStep(system_prompt=system_prompt)
        memory.steps.append(SemAggOperatorStep(task=self.task, agg_fields=self.agg_fields, items=items))

        output_json, step_number = None, 0
        while output_json is None and step_number < self.max_steps:
            memory_step = ActionStep(step_number=1, timing=Timing(start_time=time.time()))
            try:
                # convert the steps to messages
                memory_messages = self.write_memory_to_messages(memory)
                input_messages = memory_messages.copy()

                ### Generate model output ###
                memory_step.model_input_messages = input_messages
                stop_sequences = []
                try:
                    chat_message: ChatMessage = self.model.generate(input_messages, stop_sequences=stop_sequences)
                    memory_step.model_output_message = chat_message
                    memory_step.token_usage = chat_message.token_usage
                    memory_step.model_output = chat_message.content
                except Exception as e:
                    raise AgentGenerationError(f"Error in generating model output:\n{e}", self.logger) from e

                ### Parse output ###
                try:
                    output_json = parse_json_output(memory_step.model_output, self.output_tags)
                except Exception as e:
                    error_msg = f"Error in filter output parsing:\n{e}\nMake sure to return a properly formatted json output."
                    raise AgentParsingError(error_msg, self.logger) from e

            except AgentError as e:
                memory_step.error = e
            
            finally:
                self._finalize_step(memory_step)
                memory.steps.append(memory_step)
                step_number += 1

        # ensure all output fields are present
        for field in self.agg_fields:
            field_name = field['name']
            if field_name not in output_json:
                output_json[field_name] = None

        return output_json

    def __call__(self, dataset_id: str, input_datasets: dict[str, Dataset]) -> dict[str, Dataset]:
        """Execute the semantic aggregation over the input dataset.

        Requires:
            - *dataset_id* is a key in *input_datasets*.

        Returns:
            A new ``dict[str, Dataset]`` that is a copy of *input_datasets*
            with an additional entry keyed by ``self.output_dataset_id``
            containing a single aggregated item.

        Raises:
            KeyError: If *dataset_id* is not in *input_datasets*.
        """
        # retrieve items from the input dataset
        items = input_datasets[dataset_id].items

        # pre-populate system prompt
        system_prompt = populate_template(
            self.prompt_templates["sem_agg_prompt"],
            variables={
                "output_opening_tag": self.output_tags[0],
                "output_closing_tag": self.output_tags[1],
            },
        )

        # construct futures; NOTE: we should parallelize if possible in the future
        futures = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future = executor.submit(self._sem_agg, items, system_prompt)
            futures.append(future)

        # block until futures complete
        done_futures, _ = wait(futures)
        results = [fut.result() for fut in done_futures]

        # create new dataset and return it with the input datasets
        output_dataset = Dataset(name=self.output_dataset_id, annotation=f"Sem agg operator output for task: {self.task}", items=results)
        output_datasets = {**input_datasets, output_dataset.name: output_dataset}

        return output_datasets
