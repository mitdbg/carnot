import json
import time
from concurrent.futures import ThreadPoolExecutor, wait
from importlib import resources

import yaml

from carnot.agents.base import populate_template
from carnot.agents.memory import (
    ActionStep,
    AgentMemory,
    SemFilterBatchOperatorStep,
    SemFilterOperatorStep,
    SystemPromptStep,
    Timing,
)
from carnot.agents.models import ChatMessage, LiteLLMModel
from carnot.agents.monitoring import AgentLogger, LogLevel
from carnot.agents.utils import (
    AgentError,
    AgentGenerationError,
    AgentParsingError,
    parse_boolean_output,
    parse_json_output,
)
from carnot.core.models import LLMCallStats, OperatorStats
from carnot.data.dataset import Dataset


def _strip_sem_filter_now_begin_suffix(system_prompt: str) -> str:
    """Remove trailing ``Now Begin!`` so a batch-specific footer can end the prompt once."""
    trimmed = system_prompt.rstrip()
    marker = "Now Begin!"
    idx = trimmed.rfind(marker)
    if idx == -1:
        return trimmed
    return trimmed[:idx].rstrip()


class SemFilterOperator:
    """Semantic filter operator — retains or discards items via an LLM boolean judgement.

    For every item in the input dataset the operator asks the LLM whether the
    item satisfies *task*.  Items for which the LLM answers ``True`` are kept;
    all others are dropped. When ``batch_size > 1``, the operator groups items
    into batches and asks the LLM to return the matching item indices as JSON.
    Retries up to *max_steps* times per item or batch on parse errors; batched
    calls fall back to smaller batches when parsing repeatedly fails or when a
    batch is estimated to be too large.

    Representation invariant:
        - ``model`` is a ready-to-call ``LiteLLMModel`` instance.
        - ``boolean_output_tags`` and ``json_output_tags`` are two-element
          lists ``[open_tag, close_tag]``.
        - ``batch_size >= 1``.
        - ``max_steps >= 1``.

    Abstraction function:
        An instance of this class is a callable that, given a dataset, returns
        a new dataset containing only the items for which the LLM answers ``True``
        to the natural-language predicate ``task``.
    """
    _MAX_BATCH_INPUT_CHARS = 32000

    def __init__(
        self,
        task: str,
        output_dataset_id: str,
        model_id: str,
        llm_config: dict,
        max_workers: int,
        max_steps: int = 3,
        batch_size: int = 1,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.task = task
        self.output_dataset_id = output_dataset_id
        self.model = LiteLLMModel(model_id=model_id, api_key=llm_config.get("OPENAI_API_KEY"))
        self.max_workers = max_workers
        self.prompt_templates = yaml.safe_load(
            resources.files("carnot.agents.prompts").joinpath("sem_filter.yaml").read_text()
        )
        self.memory = AgentMemory("")
        self.logger = AgentLogger(level=LogLevel.INFO)
        self.boolean_output_tags = ["```text", "```"]
        self.json_output_tags = ["```json", "```"]
        self.max_steps = max_steps
        self.batch_size = batch_size

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

    def _sem_filter(self, item: dict, system_prompt: str) -> tuple[dict | None, list[LLMCallStats]]:
        """Apply the semantic filter to a single item.

        Requires:
            - *item* is a dict (or ``DataItem``) representing one dataset row.
            - *system_prompt* is a pre-populated prompt string.

        Returns:
            A tuple ``(result, llm_call_stats_list)`` where *result* is the
            original *item* if the LLM judges it passes the filter, otherwise
            ``None``, and *llm_call_stats_list* is a list of
            :class:`LLMCallStats` from each LLM call made (including retries).

        Raises:
            AgentGenerationError: If the LLM call itself fails.
        """
        memory = AgentMemory("")
        memory.system_prompt = SystemPromptStep(system_prompt=system_prompt)
        memory.steps.append(SemFilterOperatorStep(task=self.task, item=item))
        passes_filter, step_number = None, 0
        call_stats: list[LLMCallStats] = []
        while passes_filter is None and step_number < self.max_steps:
            memory_step = ActionStep(step_number=1, timing=Timing(start_time=time.time()))
            try:
                # convert the steps to messages
                memory_messages = self.write_memory_to_messages(memory)
                input_messages = memory_messages.copy()

                ### Generate model output ###
                memory_step.model_input_messages = input_messages
                stop_sequences = []
                try:
                    #print("DOCID:", item.get("docid"))
                    #print("TEXT_LEN:", len(item.get("text", "") or ""))
                    #print("TEXT_PREVIEW_REPR:", repr((item.get("text", "") or "")[:200]))
                    chat_message: ChatMessage = self.model.generate(input_messages, stop_sequences=stop_sequences)
                    memory_step.model_output_message = chat_message
                    memory_step.token_usage = chat_message.token_usage
                    memory_step.model_output = chat_message.content
                    if chat_message.llm_call_stats is not None:
                        call_stats.append(chat_message.llm_call_stats)
                except Exception as e:
                    import traceback
                    print("GENERATION EXCEPTION TYPE:", type(e))
                    print("GENERATION EXCEPTION REPR:", repr(e))
                    print("GENERATION EXCEPTION STR:", str(e))
                    traceback.print_exc()
                    raise AgentGenerationError(f"Error in generating model output:\n{e}", self.logger) from e

                ### Parse output ###
                try:
                    passes_filter = parse_boolean_output(memory_step.model_output, self.boolean_output_tags)
                except Exception as e:
                    error_msg = f"Error in filter output parsing:\n{e}\nMake sure to return a properly formatted boolean output."
                    raise AgentParsingError(error_msg, self.logger) from e

            except AgentError as e:
                memory_step.error = e
            
            finally:
                self._finalize_step(memory_step)
                memory.steps.append(memory_step)
                step_number += 1

        return (item if passes_filter else None), call_stats

    def _batch_input_chars(self, items: list[dict]) -> int:
        """Estimate prompt size for a batch using its JSON payload length."""
        return len(json.dumps(items, ensure_ascii=True))

    def _is_oversized_batch(self, items: list[dict]) -> bool:
        return len(items) > 1 and self._batch_input_chars(items) > self._MAX_BATCH_INPUT_CHARS

    def _split_batch(self, items: list[dict]) -> tuple[list[dict], list[dict]]:
        midpoint = max(1, len(items) // 2)
        return items[:midpoint], items[midpoint:]

    def _build_batch_items(self, items: list[dict]) -> list[dict]:
        return [{**item, "batch_index": index} for index, item in enumerate(items)]

    def _parse_batch_output(self, model_output: str, batch_size: int) -> list[int]:
        output_json = parse_json_output(model_output, self.json_output_tags)
        matching_indices = output_json.get("matching_indices")
        if not isinstance(matching_indices, list):
            raise ValueError("Batch output must contain a list field named 'matching_indices'.")

        normalized_indices: list[int] = []
        seen_indices: set[int] = set()
        for index in matching_indices:
            if not isinstance(index, int) or isinstance(index, bool):
                raise ValueError(f"All matching indices must be integers. Got: {index!r}")
            if index < 0 or index >= batch_size:
                raise ValueError(
                    f"Batch output index {index} is out of range for batch size {batch_size}."
                )
            if index not in seen_indices:
                normalized_indices.append(index)
                seen_indices.add(index)
        return normalized_indices

    def _sem_filter_batch_once(
        self,
        items: list[dict],
        system_prompt: str,
    ) -> tuple[list[dict] | None, list[LLMCallStats]]:
        """Apply the semantic filter to a batch of items in one LLM call."""
        batched_items = self._build_batch_items(items)
        memory = AgentMemory("")
        memory.system_prompt = SystemPromptStep(system_prompt=system_prompt)
        memory.steps.append(SemFilterBatchOperatorStep(task=self.task, items=batched_items))

        matching_indices, step_number = None, 0
        call_stats: list[LLMCallStats] = []
        while matching_indices is None and step_number < self.max_steps:
            memory_step = ActionStep(step_number=1, timing=Timing(start_time=time.time()))
            try:
                memory_messages = self.write_memory_to_messages(memory)
                input_messages = memory_messages.copy()

                memory_step.model_input_messages = input_messages
                stop_sequences = []
                try:
                    chat_message: ChatMessage = self.model.generate(
                        input_messages, stop_sequences=stop_sequences
                    )
                    memory_step.model_output_message = chat_message
                    memory_step.token_usage = chat_message.token_usage
                    memory_step.model_output = chat_message.content
                    if chat_message.llm_call_stats is not None:
                        call_stats.append(chat_message.llm_call_stats)
                except Exception as e:
                    raise AgentGenerationError(
                        f"Error in generating model output:\n{e}", self.logger
                    ) from e

                try:
                    matching_indices = self._parse_batch_output(
                        memory_step.model_output,
                        batch_size=len(items),
                    )
                except Exception as e:
                    error_msg = (
                        f"Error in batch filter output parsing:\n{e}\n"
                        "Make sure to return a properly formatted JSON output."
                    )
                    raise AgentParsingError(error_msg, self.logger) from e

            except AgentError as e:
                memory_step.error = e

            finally:
                self._finalize_step(memory_step)
                memory.steps.append(memory_step)
                step_number += 1

        if matching_indices is None:
            return None, call_stats

        selected_indices = set(matching_indices)
        selected_items = [
            item for item_index, item in enumerate(items) if item_index in selected_indices
        ]
        return selected_items, call_stats

    def _sem_filter_batch(
        self,
        items: list[dict],
        single_system_prompt: str,
        batch_system_prompt: str,
    ) -> tuple[list[dict], list[LLMCallStats]]:
        """Filter a batch, falling back to smaller batches when needed."""
        if not items:
            return [], []
        if len(items) == 1:
            result, call_stats = self._sem_filter(items[0], single_system_prompt)
            return ([result] if result is not None else []), call_stats
        if self._is_oversized_batch(items):
            left_items, right_items = self._split_batch(items)
            left_results, left_stats = self._sem_filter_batch(
                left_items, single_system_prompt, batch_system_prompt
            )
            right_results, right_stats = self._sem_filter_batch(
                right_items, single_system_prompt, batch_system_prompt
            )
            return left_results + right_results, left_stats + right_stats

        batch_results, call_stats = self._sem_filter_batch_once(items, batch_system_prompt)
        if batch_results is not None:
            return batch_results, call_stats

        left_items, right_items = self._split_batch(items)
        left_results, left_stats = self._sem_filter_batch(
            left_items, single_system_prompt, batch_system_prompt
        )
        right_results, right_stats = self._sem_filter_batch(
            right_items, single_system_prompt, batch_system_prompt
        )
        return left_results + right_results, call_stats + left_stats + right_stats

    def _chunk_items(self, items: list[dict]) -> list[list[dict]]:
        if self.batch_size <= 1:
            return [[item] for item in items]
        return [items[index : index + self.batch_size] for index in range(0, len(items), self.batch_size)]

    def __call__(self, dataset_id: str, input_datasets: dict[str, Dataset]) -> tuple[dict[str, Dataset], OperatorStats]:
        """Execute the semantic filter over every item in the input dataset.

        Requires:
            - *dataset_id* is a key in *input_datasets*.
            - ``input_datasets[dataset_id].items`` is an iterable of dicts.

        Returns:
            A tuple ``(output_datasets, stats)`` where *output_datasets* is
            a **new** ``dict[str, Dataset]`` that is a copy of
            *input_datasets* with an additional entry keyed by
            ``self.output_dataset_id`` containing only the items that
            passed the filter, and *stats* is an :class:`OperatorStats`
            summarising the LLM calls made.

        Raises:
            KeyError: If *dataset_id* is not in *input_datasets*.
        """
        op_start = time.perf_counter()

        # retrieve items from the input dataset
        items = list(input_datasets[dataset_id].items)

        # pre-populate system prompt
        single_system_prompt = populate_template(
            self.prompt_templates["sem_filter_prompt"],
            variables={
                "output_opening_tag": self.boolean_output_tags[0],
                "output_closing_tag": self.boolean_output_tags[1],
            },
        )
        batch_reference = _strip_sem_filter_now_begin_suffix(single_system_prompt)
        batch_system_prompt = (
            populate_template(
                self.prompt_templates["sem_filter_batch_header"],
                variables={
                    "output_opening_tag": self.json_output_tags[0],
                    "output_closing_tag": self.json_output_tags[1],
                },
            )
            + "\n\n"
            + batch_reference
            + "\n\n"
            + populate_template(
                self.prompt_templates["sem_filter_batch_footer"],
                variables={
                    "output_opening_tag": self.json_output_tags[0],
                    "output_closing_tag": self.json_output_tags[1],
                },
            )
        )

        batches = self._chunk_items(items)
        futures = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for batch in batches:
                if self.batch_size <= 1:
                    future = executor.submit(self._sem_filter, batch[0], single_system_prompt)
                else:
                    future = executor.submit(
                        self._sem_filter_batch,
                        batch,
                        single_system_prompt,
                        batch_system_prompt,
                    )
                futures.append(future)

        wait(futures)
        all_call_stats: list[LLMCallStats] = []
        results = []
        for fut in futures:
            batch_results, item_stats = fut.result()
            all_call_stats.extend(item_stats)
            if self.batch_size <= 1:
                if batch_results is not None:
                    results.append(batch_results)
            else:
                results.extend(batch_results)

        # create new dataset and return it with the input datasets
        output_dataset = Dataset(name=self.output_dataset_id, annotation=f"Sem filter operator output for task: {self.task}", items=results)
        output_datasets = {**input_datasets, output_dataset.name: output_dataset}

        op_stats = OperatorStats(
            operator_name="SemFilter",
            operator_id=self.output_dataset_id,
            wall_clock_secs=time.perf_counter() - op_start,
            llm_calls=all_call_stats,
            items_in=len(items),
            items_out=len(results),
        )

        return output_datasets, op_stats
