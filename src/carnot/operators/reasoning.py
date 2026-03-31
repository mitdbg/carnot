import time
from collections.abc import Generator
from importlib import resources

import yaml
from rich.console import Group
from rich.text import Text

from carnot.agents.base import populate_template
from carnot.agents.local_python_executor import fix_final_answer_code
from carnot.agents.memory import (
    ActionStep,
    CodeOperatorStep,
    SystemPromptStep,
    Timing,
    ToolCall,
)
from carnot.agents.models import ChatMessage
from carnot.agents.monitoring import YELLOW_HEX, LogLevel
from carnot.agents.tools import Tool
from carnot.agents.utils import (
    AgentError,
    AgentExecutionError,
    AgentGenerationError,
    AgentParsingError,
    parse_code_blobs,
    truncate_content,
)
from carnot.core.models import LLMCallStats, OperatorStats
from carnot.data.dataset import Dataset
from carnot.operators.code import CodeActionOutput, CodeOperator, FinalAnswerStep


class ReasoningOperator(CodeOperator):
    """Reasoning operator — a ``CodeOperator`` with a prompt specialised for reasoning.

    Inherits the full agentic generate/execute loop from ``CodeOperator``
    but loads ``reasoning_operator.yaml`` prompts instead of
    ``code_operator.yaml``.  The final answer is expected to include a
    ``final_items`` key in its code state.

    Representation invariant:
        Same as ``CodeOperator``, plus ``prompt_templates`` are loaded from
        ``reasoning_operator.yaml``.

    Abstraction function:
        An instance of this class is a ``CodeOperator`` whose prompts guide the LLM to
        reason over the input datasets and produce structured output items.
    """
    def __init__(self, task: str, output_dataset_id: str, model_id: str, llm_config: dict, tools: list[Tool] | None = None, additional_authorized_imports: list[str] | None = None, max_steps: int = 20):
        super().__init__(task, output_dataset_id, model_id, llm_config, tools, additional_authorized_imports, max_steps)
        self.prompt_templates = yaml.safe_load(
            resources.files("carnot.agents.prompts").joinpath("reasoning_operator.yaml").read_text()
        )

    def _run_stream(self, input_datasets: dict[str, Dataset]) -> Generator[ActionStep | FinalAnswerStep]:
        self.step_number = 1
        final_answer, final_code, final_execution_logs = None, None, None
        returned_final_answer = False
        while not returned_final_answer and self.step_number <= self.max_steps:
            # Start action step!
            action_step = ActionStep(step_number=self.step_number, timing=Timing(start_time=time.time()))
            self.logger.log_rule(f"Step {self.step_number}", level=LogLevel.INFO)
            try:
                for output in self._step_generate_code(action_step, input_datasets):
                    # Yield all
                    yield output

                    if isinstance(output, CodeActionOutput) and output.is_final_answer:
                        final_answer = output.output
                        final_code = output.code
                        final_execution_logs = output.execution_logs
                        self.logger.log(
                            Text(f"Final answer: {final_answer}", style=f"bold {YELLOW_HEX}"),
                            level=LogLevel.INFO,
                        )
                        returned_final_answer = True

            # Agent generation errors are not caused by a Model error but an implementation error: so we should raise them and exit.
            except AgentGenerationError as e:
                raise e

            # Other AgentError types are caused by the Model, so we should log them and iterate.
            except AgentError as e:
                action_step.error = e

            finally:
                self._finalize_step(action_step)
                self.memory.steps.append(action_step)
                yield action_step
                self.step_number += 1

        if not returned_final_answer and self.step_number == self.max_steps + 1:
            final_answer = self._handle_max_steps_reached()
            final_code = ""
            final_execution_logs = ""
            yield action_step
        yield FinalAnswerStep(final_answer, final_execution_logs, final_code)

    def _step_generate_code(self, memory_step: ActionStep, input_datasets: dict[str, Dataset]) -> Generator[ToolCall | CodeActionOutput]:
        """
        Generate code to execute the given task on the input datasets.
        """
        system_prompt = populate_template(
            self.prompt_templates["reasoning_prompt"],
            variables={
                "code_block_opening_tag": self.code_block_tags[0],
                "code_block_closing_tag": self.code_block_tags[1],
            },
        )
        self.memory.system_prompt = SystemPromptStep(system_prompt=system_prompt)
        self.memory.steps.append(CodeOperatorStep(task=self.task, tools=self.tools, input_datasets=input_datasets, code_block_tags=self.code_block_tags))

        # convert the steps to messages
        memory_messages = self.write_memory_to_messages()
        input_messages = memory_messages.copy()

        ### Generate model output ###
        memory_step.model_input_messages = input_messages
        stop_sequences = []
        try:
            for msg in input_messages:
                self.logger.log(msg.render_as_markdown(), level=LogLevel.INFO)
            chat_message: ChatMessage = self.model.generate(input_messages, stop_sequences=stop_sequences)
            memory_step.model_output_message = chat_message
            output_text = chat_message.content

            # This adds the end code sequence (i.e. the closing code block tag) to the history.
            # This will nudge subsequent LLM calls to finish with this end code sequence, thus efficiently stopping generation.
            if output_text and not output_text.strip().endswith(self.code_block_tags[1]):
                output_text += self.code_block_tags[1]
                memory_step.model_output_message.content = output_text

            memory_step.token_usage = chat_message.token_usage
            memory_step.model_output = output_text
        except Exception as e:
            raise AgentGenerationError(f"Error in generating model output:\n{e}", self.logger) from e

        ### Parse output ###
        try:
            code_action = parse_code_blobs(output_text, self.code_block_tags)
            code_action = fix_final_answer_code(code_action)
            memory_step.code_action = code_action
        except Exception as e:
            error_msg = f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
            raise AgentParsingError(error_msg, self.logger) from e

        # Check for anti-patterns before executing
        self._check_for_anti_patterns(code_action)

        tool_call = ToolCall(
            name="python_interpreter",
            arguments=code_action,
            id=f"call_{len(self.memory.steps)}",
        )
        yield tool_call
        memory_step.tool_calls = [tool_call]

        ### Execute action ###
        self.logger.log_code(title="Executing parsed code:", content=code_action, level=LogLevel.INFO)
        try:
            code_output = self.python_executor(code_action)
            execution_outputs_console = []
            if len(code_output.logs) > 0:
                execution_outputs_console += [
                    Text("Execution logs:", style="bold"),
                    Text(code_output.logs),
                ]
            observation = "Execution logs:\n" + code_output.logs
        except Exception as e:
            if hasattr(self.python_executor, "state") and "_print_outputs" in self.python_executor.state:
                execution_logs = str(self.python_executor.state["_print_outputs"])
                if len(execution_logs) > 0:
                    execution_outputs_console = [
                        Text("Execution logs:", style="bold"),
                        Text(execution_logs),
                    ]
                    memory_step.observations = "Execution logs:\n" + execution_logs
                    self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
            error_msg = str(e)
            if "Import of " in error_msg and " is not allowed" in error_msg:
                self.logger.log(
                    "[bold red]Warning to user: Code execution failed due to an unauthorized import - Consider passing said import under `additional_authorized_imports` when initializing your CodeAgent.",
                    level=LogLevel.INFO,
                )
            raise AgentExecutionError(error_msg, self.logger) from e

        truncated_output = truncate_content(str(code_output.output))
        observation += "Last output from code snippet:\n" + truncated_output
        memory_step.observations = observation

        if not code_output.is_final_answer:
            execution_outputs_console += [
                Text(
                    f"Out: {truncated_output}",
                ),
            ]
        self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
        memory_step.code_action_output = code_output.output
        yield CodeActionOutput(
            code=code_action,
            output=code_output.output,
            execution_logs=observation,
            is_final_answer=code_output.is_final_answer,
        )

    def __call__(self, input_datasets: dict[str, Dataset]) -> tuple[dict[str, Dataset], OperatorStats]:
        """Execute the reasoning loop and return the resulting datasets.

        Requires:
            - *input_datasets* is a non-empty ``dict[str, Dataset]``.

        Returns:
            A tuple ``(output_datasets, stats)`` where *output_datasets*
            is a new ``dict[str, Dataset]`` with an additional entry keyed
            by ``self.output_dataset_id`` whose ``items`` are the
            ``final_items`` from the code state, and *stats* is an
            :class:`OperatorStats` summarising all LLM calls made.

        Raises:
            AgentGenerationError: If the LLM fails on the first step.
            KeyError: If ``final_items`` is absent from the code state.
        """
        op_start = time.perf_counter()

        self.python_executor.send_variables(variables={"input_datasets": input_datasets})
        self.python_executor.send_tools({**self.tools})

        # use an LLM to generate and execute code based on the task and input datasets.
        steps = list(self._run_stream(input_datasets))
        assert isinstance(steps[-1], FinalAnswerStep)
        output_state = steps[-1].output # TODO: enforce that this is a proper state dictionary

        # Collect LLM call stats from all action steps
        all_call_stats: list[LLMCallStats] = []
        for step in steps:
            if (
                isinstance(step, ActionStep)
                and step.model_output_message is not None
                and hasattr(step.model_output_message, "llm_call_stats")
                and step.model_output_message.llm_call_stats is not None
            ):
                all_call_stats.append(step.model_output_message.llm_call_stats)

        # create new dataset and return it with the input datasets
        output_dataset = Dataset(
            name=self.output_dataset_id,
            annotation=f"Reasoning operator output for task: {self.task}",
            items=output_state["final_items"],
            code_state=output_state,
        )
        output_datasets = {**input_datasets, output_dataset.name: output_dataset}

        op_stats = OperatorStats(
            operator_name="Reasoning",
            operator_id=self.output_dataset_id,
            wall_clock_secs=time.perf_counter() - op_start,
            llm_calls=all_call_stats,
            items_in=sum(len(ds.items) for ds in input_datasets.values()),
            items_out=len(output_dataset.items) if output_dataset.items else 0,
        )

        return output_datasets, op_stats
