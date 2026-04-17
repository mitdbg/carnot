from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import resources

import yaml

from carnot.agents.base import populate_template
from carnot.agents.memory import (
    ActionStep,
    AgentMemory,
    SemFlatMapOperatorStep,
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
from carnot.core.models import LLMCallStats, OperatorStats
from carnot.data.dataset import Dataset
from carnot.operators.physical import PhysicalOperator
from carnot.optimizer.model_ids import get_api_key_for_model
from carnot.utils.model_helpers import count_tokens, truncate_item_to_fit

# SemFlatMapOperatorStep templates prompt as:
# - f"Map Instruction: \"{self.task}\"\n\nOutput Fields:\n{output_fields_str}\n\nInput:\n{input_str}"
# this means the overhead tokens needs to cover the text excluding the task, output_fields_str, and input_str,
# which we estimate to be around 50 tokens just to be safe
_FLAT_MAP_PROMPT_OVERHEAD_TOKENS = 50

class SemFlatMapOperator(PhysicalOperator):
    """Semantic flat-map operator — expands each item into zero or more output items.

    For every item the LLM is asked to produce a JSON *list* of new items
    with the specified ``output_fields``.  All per-item lists are flattened
    into a single output dataset.  Missing fields default to ``None``.

    Representation invariant:
        - ``output_fields`` is a non-empty list of dicts, each with a
          ``'name'`` key.
        - ``output_tags`` is ``["```json", "```"]``.
        - ``max_steps >= 1``.

    Abstraction function:
        An instance of this class is a callable that, given a dataset, returns
        a new dataset where each original item has been expanded into zero or
        more items according to the LLM's response.
    """
    def __init__(
            self,
            task: str,
            output_fields: list[dict],
            dataset_id: str,
            model_id: str,
            llm_config: dict,
            max_workers: int,
            max_steps: int = 3,
            logical_op_id: str | None = None,
            logical_op_class_name: str | None = None,
        ):
        super().__init__(logical_op_id=logical_op_id, logical_op_class_name=logical_op_class_name)
        self.task = task
        self.dataset_id = dataset_id
        self.output_fields = output_fields
        self.model_id = model_id
        self.llm_config = llm_config
        self.model = LiteLLMModel(model_id=model_id, api_key=get_api_key_for_model(model_id, llm_config))
        self.max_workers = max_workers
        self.prompt_templates = yaml.safe_load(
            resources.files("carnot.agents.prompts").joinpath("sem_flat_map.yaml").read_text()
        )
        self.memory = AgentMemory("")
        self.logger = AgentLogger(level=LogLevel.INFO)
        self.output_tags = ["```json", "```"]
        self.max_steps = max_steps

    def get_id_params(self):
        id_params = super().get_id_params()
        id_params = {
            "task": self.task,
            "output_fields": self.output_fields,
            "dataset_id": self.dataset_id,
            "model_id": self.model_id,
            **id_params,
        }

        return id_params

    def get_op_params(self):
        op_params = super().get_op_params()
        op_params = {
            "task": self.task,
            "output_fields": self.output_fields,
            "dataset_id": self.dataset_id,
            "model_id": self.model_id,
            "llm_config": self.llm_config,
            "max_workers": self.max_workers,
            "max_steps": self.max_steps,
            **op_params,
        }

        return op_params

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

    def _sem_flat_map(self, item: dict, system_prompt: str) -> tuple[list[dict] | None, list[LLMCallStats]]:
        """Expand a single item into a list of output dicts via the LLM.

        Requires:
            - *item* is a dict representing one dataset row.
            - *system_prompt* is a pre-populated prompt string.

        Returns:
            A tuple ``(expanded_items, llm_call_stats_list)`` where
            *expanded_items* is a ``list[dict]`` of items with
            ``output_fields`` keys (missing fields default to ``None``),
            and *llm_call_stats_list* is a list of :class:`LLMCallStats`
            from each LLM call made (including retries).

        Raises:
            AgentGenerationError: If the LLM call itself fails.
        """
        # Truncate item if it would exceed the model's context window.
        output_fields_str = "\n".join([
            f"- {field['name']}" + (f" ({field['type']})" if 'type' in field else "") + f": {field['description']}"
            for field in self.output_fields
        ])
        overhead = (
            count_tokens(system_prompt, self.model_id)
            + count_tokens(self.task, self.model_id)
            + count_tokens(output_fields_str, self.model_id)
            + _FLAT_MAP_PROMPT_OVERHEAD_TOKENS
        )
        item, embed_stats = truncate_item_to_fit(item, self.task, self.model_id, self.llm_config, overhead_tokens=overhead)

        memory = AgentMemory("")
        memory.system_prompt = SystemPromptStep(system_prompt=system_prompt)
        memory.steps.append(SemFlatMapOperatorStep(task=self.task, output_fields=self.output_fields, item=item))

        output_json, step_number = None, 0
        call_stats: list[LLMCallStats] = list(embed_stats)
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
                    if chat_message.llm_call_stats is not None:
                        call_stats.append(chat_message.llm_call_stats)
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
        for field in self.output_fields:
            field_name = field['name']
            for item in output_json:
                if field_name not in item:
                    item[field_name] = None

        return output_json, call_stats

    def __call__(self, dataset_id: str, input_datasets: dict[str, Dataset], on_item_complete: Callable[[], None] | None = None) -> tuple[dict[str, Dataset], OperatorStats]:
        """Execute the semantic flat-map over every item in the input dataset.

        Requires:
            - *dataset_id* is a key in *input_datasets*.

        Returns:
            A tuple ``(output_datasets, stats)`` where *output_datasets*
            is a new ``dict[str, Dataset]`` with an additional entry keyed
            by ``self.dataset_id`` containing the flattened
            expansion of all items, and *stats* is an
            :class:`OperatorStats` summarising all LLM calls made.

        Raises:
            KeyError: If *dataset_id* is not in *input_datasets*.
        """
        op_start = time.perf_counter()

        # retrieve items from the input dataset
        items = input_datasets[dataset_id].items

        # pre-populate system prompt
        system_prompt = populate_template(
            self.prompt_templates["sem_flat_map_prompt"],
            variables={
                "output_opening_tag": self.output_tags[0],
                "output_closing_tag": self.output_tags[1],
            },
        )

        # construct futures
        futures = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for item in items:
                future = executor.submit(self._sem_flat_map, item, system_prompt)
                futures.append(future)

        # collect results as futures complete
        all_call_stats: list[LLMCallStats] = []
        results = []
        for fut in as_completed(futures):
            expanded_items, item_stats = fut.result()
            all_call_stats.extend(item_stats)
            results.extend(expanded_items)
            if on_item_complete is not None:
                on_item_complete()

        # create new dataset and return it with the input datasets
        output_dataset = Dataset(name=self.dataset_id, annotation=f"Sem flat map operator output for task: {self.task}", items=results)
        output_datasets = {**input_datasets, output_dataset.name: output_dataset}

        op_stats = OperatorStats(
            operator_name="SemFlatMap",
            operator_id=self.dataset_id,
            wall_clock_secs=time.perf_counter() - op_start,
            llm_calls=all_call_stats,
            items_in=len(items),
            items_out=len(results),
        )

        return output_datasets, op_stats
