import time
from concurrent.futures import ThreadPoolExecutor, wait
from importlib import resources

import yaml

from carnot.agents.base import populate_template
from carnot.agents.memory import (
    ActionStep,
    AgentMemory,
    SemJoinOperatorStep,
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
)
from carnot.data.dataset import Dataset


class SemJoinOperator:
    """Semantic join operator — cross-product join filtered by an LLM boolean predicate.

    For every ``(left, right)`` pair the LLM is asked whether the pair
    satisfies *task*.  Pairs that pass are merged into a single output
    dict.  Shared keys are prefixed with ``left_`` / ``right_``.

    Representation invariant:
        - ``output_tags`` is ``["```text", "```"]``.
        - ``max_steps >= 1``.

    Abstraction function:
        An instance of this class is a callable that, given a left and right dataset,
        returns a new dataset containing the cross-product of items that the LLM judges
        as matching under ``task``.
    """
    def __init__(self, task: str, model_id: str, llm_config: dict, output_dataset_id: str, max_workers: int, max_steps: int = 3):
        self.task = task
        self.output_dataset_id = output_dataset_id
        self.model = LiteLLMModel(model_id=model_id, api_key=llm_config.get("OPENAI_API_KEY"))
        self.max_workers = max_workers
        self.prompt_templates = yaml.safe_load(
            resources.files("carnot.agents.prompts").joinpath("sem_join.yaml").read_text()
        )
        self.memory = AgentMemory("")
        self.logger = AgentLogger(level=LogLevel.INFO)
        self.output_tags = ["```text", "```"]
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

    def _sem_join(self, left_item: dict, right_item: dict, system_prompt: str) -> dict | None:
        """Evaluate whether a single ``(left, right)`` pair should be joined.

        Requires:
            - *left_item* and *right_item* are dicts.
            - *system_prompt* is a pre-populated prompt string.

        Returns:
            A merged dict if the LLM judges the pair as matching, otherwise
            ``None``.  Shared keys receive ``left_`` / ``right_`` prefixes.

        Raises:
            AgentGenerationError: If the LLM call itself fails.
        """
        memory = AgentMemory("")
        memory.system_prompt = SystemPromptStep(system_prompt=system_prompt)
        memory.steps.append(SemJoinOperatorStep(task=self.task, left_item=left_item, right_item=right_item))

        is_joined, step_number = None, 0
        while is_joined is None and step_number < self.max_steps:
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
                    is_joined = parse_boolean_output(memory_step.model_output, self.output_tags)
                except Exception as e:
                    error_msg = f"Error in filter output parsing:\n{e}\nMake sure to return a properly formatted boolean output."
                    raise AgentParsingError(error_msg, self.logger) from e

            except AgentError as e:
                memory_step.error = e
            
            finally:
                self._finalize_step(memory_step)
                memory.steps.append(memory_step)
                step_number += 1

        if not is_joined:
            return None
        
        output_dict = {}
        shared_keys = set(left_item.keys()).intersection(set(right_item.keys()))
        for key, value in left_item.items():
            if key in shared_keys:
                output_dict[f"left_{key}"] = value
            else:
                output_dict[key] = value
        for key, value in right_item.items():
            if key in shared_keys:
                output_dict[f"right_{key}"] = value
            else:
                output_dict[key] = value

        return output_dict

    def __call__(self, left_dataset_id: str, right_dataset_id: str, input_datasets: dict[str, Dataset]) -> dict[str, Dataset]:
        """Execute the semantic join over the cross-product of two datasets.

        Requires:
            - *left_dataset_id* and *right_dataset_id* are keys in
              *input_datasets*.

        Returns:
            A new ``dict[str, Dataset]`` that is a copy of *input_datasets*
            with an additional entry keyed by ``self.output_dataset_id``
            containing the joined rows.

        Raises:
            KeyError: If either dataset id is not in *input_datasets*.
        """
        # retrieve left and right items from the input datasets
        left_items = input_datasets[left_dataset_id].items
        right_items = input_datasets[right_dataset_id].items

        # pre-populate system prompt
        system_prompt = populate_template(
            self.prompt_templates["sem_join_prompt"],
            variables={
                "output_opening_tag": self.output_tags[0],
                "output_closing_tag": self.output_tags[1],
            },
        )

        # construct futures
        futures = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for left_item in left_items:
                for right_item in right_items:
                    future = executor.submit(self._sem_join, left_item, right_item, system_prompt)
                    futures.append(future)

        # block until futures complete
        done_futures, _ = wait(futures)
        results = [fut.result() for fut in done_futures]
        results = list(filter(None, results))

        # create new dataset and return it with the input datasets
        output_dataset = Dataset(name=self.output_dataset_id, annotation=f"Sem join operator output for task: {self.task}", items=results)
        output_datasets = {**input_datasets, output_dataset.name: output_dataset}

        return output_datasets
