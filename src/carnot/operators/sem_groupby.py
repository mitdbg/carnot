import time
from concurrent.futures import ThreadPoolExecutor, wait
from importlib import resources

import yaml

from carnot.agents.base import populate_template
from carnot.agents.memory import (
    ActionStep,
    AgentMemory,
    SemAggOperatorStep,
    SemGroupByGroupOperatorStep,
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


class SemGroupByOperator(PhysicalOperator):
    """Semantic group-by operator — groups items then aggregates per group.

    Execution proceeds in two phases:

    1. **Grouping** (``_sem_group``): The LLM assigns group-by field values
       to each item.  Deterministic fields could skip the LLM call (not yet
       optimised).
    2. **Aggregation**: Built-in aggregations (``min``, ``max``, ``mean``,
       ``sum``, ``count``) are computed locally.  Any remaining aggregation
       functions are delegated to the LLM via ``_sem_agg``.

    Representation invariant:
        - ``group_by_fields`` is a non-empty list of dicts with ``'name'``
          keys.
        - ``agg_fields`` is a non-empty list of dicts with ``'name'`` and
          ``'func'`` keys.
        - ``_sem_agg_fields`` ⊆ ``agg_fields`` and contains only fields
          whose ``'func'`` is not in ``{min, max, count, sum, mean}``.
        - ``max_steps >= 1``.

    Abstraction function:
        An instance of this class is a callable that, given a dataset, returns
        a new dataset with one row per group, each containing the computed
        aggregation values.
    """
    def __init__(
            self,
            task: str,
            group_by_fields: list[dict],
            agg_fields: list[dict],
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
        self.group_by_fields = group_by_fields
        self.agg_fields = agg_fields
        self.model_id = model_id
        self.llm_config = llm_config
        self._sem_agg_fields = [field for field in agg_fields if field['func'] not in ["min", "max", "count", "sum", "mean"]]
        self.model = LiteLLMModel(model_id=model_id, api_key=get_api_key_for_model(model_id, llm_config))
        self.max_workers = max_workers
        self.group_by_prompt_templates = yaml.safe_load(
            resources.files("carnot.agents.prompts").joinpath("sem_groupby.yaml").read_text()
        )
        self.agg_prompt_templates = yaml.safe_load(
            resources.files("carnot.agents.prompts").joinpath("sem_agg.yaml").read_text()
        )
        self.memory = AgentMemory("")
        self.logger = AgentLogger(level=LogLevel.INFO)
        self.output_tags = ["```json", "```"]
        self.max_steps = max_steps

    def get_id_params(self):
        id_params = super().get_id_params()
        id_params = {
            "task": self.task,
            "group_by_fields": self.group_by_fields,
            "agg_fields": self.agg_fields,
            "dataset_id": self.dataset_id,
            "model_id": self.model_id,
            **id_params,
        }

        return id_params

    def get_op_params(self):
        op_params = super().get_op_params()
        op_params = {
            "task": self.task,
            "group_by_fields": self.group_by_fields,
            "agg_fields": self.agg_fields,
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

    def _sem_group(self, item: dict, system_prompt: str) -> tuple[dict | None, list[LLMCallStats]]:
        """Compute the group-by field values for a single item via the LLM.

        Requires:
            - *item* is a dict representing one dataset row.

        Returns:
            A tuple ``(item, llm_call_stats_list)`` where *item* is the
            dict mutated in-place with the group-by field values (missing
            fields are set to ``None``), and *llm_call_stats_list* is a
            list of :class:`LLMCallStats` from each LLM call made.

        Raises:
            AgentGenerationError: If the LLM call itself fails.
        """
        memory = AgentMemory("")
        memory.system_prompt = SystemPromptStep(system_prompt=system_prompt)

        # NOTE: we should avoid this step if the group by fields are all deterministic
        # NOTE: we could use a semantic aggregation to first compute the unique group values and then classify
        # first, compute the value of each group by field for each input
        memory.steps.append(SemGroupByGroupOperatorStep(group_by_fields=self.group_by_fields, item=item))

        output_json, step_number = None, 0
        call_stats: list[LLMCallStats] = []
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

                # update item
                item.update(output_json)

            except AgentError as e:
                memory_step.error = e
            
            finally:
                self._finalize_step(memory_step)
                memory.steps.append(memory_step)
                step_number += 1

        # ensure all output fields are present
        for field in self.group_by_fields:
            field_name = field['name']
            if field_name not in item:
                item[field_name] = None

        return item, call_stats

    def _sem_agg(self, items: list[dict], system_prompt: str, group_key: tuple[str]) -> tuple[tuple[str], dict | None, list[LLMCallStats]]:
        """Aggregate items within a single group via the LLM.

        Only the fields in ``_sem_agg_fields`` are aggregated here; built-in
        aggregations are handled in ``__call__``.

        Requires:
            - *items* is a non-empty list of dicts belonging to the same group.
            - *group_key* identifies the group.

        Returns:
            A tuple ``(group_key, output_dict, llm_call_stats_list)`` where
            *output_dict* contains the aggregated field values and
            *llm_call_stats_list* is a list of :class:`LLMCallStats`.

        Raises:
            AgentGenerationError: If the LLM call itself fails.
        """
        memory = AgentMemory("")
        memory.system_prompt = SystemPromptStep(system_prompt=system_prompt)
        memory.steps.append(SemAggOperatorStep(task="Apply the aggregation(s) specified in Output Fields over the Input.", agg_fields=self._sem_agg_fields, items=items))

        output_json, step_number = None, 0
        call_stats: list[LLMCallStats] = []
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
        for field in self._sem_agg_fields:
            field_name = field['name']
            if field_name not in output_json:
                output_json[field_name] = None

        return group_key, output_json, call_stats

    def __call__(self, dataset_id: str, input_datasets: dict[str, Dataset]) -> tuple[dict[str, Dataset], OperatorStats]:
        """Execute the two-phase semantic group-by over the input dataset.

        Phase 1 computes group-by field values (potentially via LLM).
        Phase 2 computes aggregations per group — built-in functions are
        evaluated locally; semantic aggregations are delegated to the LLM.

        Requires:
            - *dataset_id* is a key in *input_datasets*.

        Returns:
            A tuple ``(output_datasets, stats)`` where *output_datasets*
            is a new ``dict[str, Dataset]`` with an additional entry keyed
            by ``self.dataset_id`` containing one row per group,
            and *stats* is an :class:`OperatorStats` summarising all LLM
            calls made across both phases.

        Raises:
            KeyError: If *dataset_id* is not in *input_datasets*.
        """
        op_start = time.perf_counter()
        all_call_stats: list[LLMCallStats] = []

        # retrieve items from the input dataset
        items = input_datasets[dataset_id].items

        # first, compute the group by field value(s) for each input item
        group_prompt = populate_template(
            self.group_by_prompt_templates["compute_groupby_fields_prompt"],
            variables={
                "output_opening_tag": self.output_tags[0],
                "output_closing_tag": self.output_tags[1],
            },
        )

        # construct futures for computing the group for each input item
        futures = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for item in items:
                future = executor.submit(self._sem_group, item, group_prompt)
                futures.append(future)

        # block until futures complete
        done_futures, _ = wait(futures)
        items: list[dict] = []
        for fut in done_futures:
            result_item, item_stats = fut.result()
            all_call_stats.extend(item_stats)
            items.append(result_item)

        # compute the non-semantic agregations for each group
        agg_state = {}
        for item in items:
            group_key = tuple(item[field['name']] for field in self.group_by_fields)
            if group_key not in agg_state:
                agg_state[group_key] = {}
                for agg_field in self.agg_fields:
                    agg_func = agg_field['func']
                    state = {}
                    if agg_func == "min":
                        state = {"min": None}
                    elif agg_func == "max":
                        state = {"max": None}
                    elif agg_func == "count":
                        state = {"count": 0}
                    elif agg_func == "sum":
                        state = {"total": 0.0}
                    elif agg_func == "mean":
                        state = {"count": 0, "total": 0.0}
                    agg_state[group_key][agg_func] = state

            # update aggregation state for each agg field
            for agg_field in self.agg_fields:
                agg_field_name = agg_field['name']
                agg_func = agg_field['func']
                value = item.get(agg_field_name, None)
                state = agg_state[group_key][agg_func]

                if agg_func == "min":
                    if state["min"] is None or (value is not None and value < state["min"]):
                        state["min"] = value
                elif agg_func == "max":
                    if state["max"] is None or (value is not None and value > state["max"]):
                        state["max"] = value
                elif agg_func == "count":
                    state["count"] += 1
                elif agg_func == "sum":
                    if value is not None:
                        state["total"] += value
                elif agg_func == "mean" and value is not None:
                    state["total"] += value
                    state["count"] += 1

        # compute the semantic aggregations for each group
        agg_prompt = populate_template(
            self.agg_prompt_templates["sem_agg_prompt"],
            variables={
                "output_opening_tag": self.output_tags[0],
                "output_closing_tag": self.output_tags[1],
            },
        )

        # construct futures for computing the aggregation for each group
        if len(self._sem_agg_fields) > 0:
            futures = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for group_key in agg_state:
                    group_items = [item for item in items if all(item[field['name']] == group_key[i] for i, field in enumerate(self.group_by_fields))]
                    future = executor.submit(self._sem_agg, group_items, agg_prompt, group_key)
                    futures.append(future)

            # block until futures complete
            done_futures, _ = wait(futures)
            group_sem_agg_outputs = [fut.result() for fut in done_futures]

            # update aggregation state with semantic aggregation results
            for group_key, sem_agg_output, agg_stats in group_sem_agg_outputs:
                all_call_stats.extend(agg_stats)
                for agg_field in self._sem_agg_fields:
                    agg_name = agg_field['name']
                    agg_func = agg_field['func']
                    if agg_func not in ["min", "max", "count", "sum", "mean"]:
                        agg_state[group_key][agg_name] = sem_agg_output.get(agg_name, None)

        # construct final results
        results = []
        for group_key, aggs in agg_state.items():
            result = {}
            for i, field in enumerate(self.group_by_fields):
                result[field['name']] = group_key[i]
            for agg_field in self.agg_fields:
                agg_func = agg_field['func']
                state = aggs[agg_func]
                if agg_func == "min":
                    result[agg_field['name']] = state["min"]
                elif agg_func == "max":
                    result[agg_field['name']] = state["max"]
                elif agg_func == "count":
                    result[agg_field['name']] = state["count"]
                elif agg_func == "sum":
                    result[agg_field['name']] = state["total"]
                elif agg_func == "mean":
                    mean_value = state["total"] / state["count"] if state["count"] > 0 else None
                    result[agg_field['name']] = mean_value
                else:
                    result[agg_field['name']] = state
            results.append(result)

        # create new dataset and return it with the input datasets
        output_dataset = Dataset(name=self.dataset_id, annotation=f"Sem group by operator output for task: {self.task}", items=results)
        output_datasets = {**input_datasets, output_dataset.name: output_dataset}

        op_stats = OperatorStats(
            operator_name="SemGroupBy",
            operator_id=self.dataset_id,
            wall_clock_secs=time.perf_counter() - op_start,
            llm_calls=all_call_stats,
            items_in=len(input_datasets[dataset_id].items),
            items_out=len(results),
        )

        return output_datasets, op_stats
