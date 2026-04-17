import re
import time
from collections.abc import Generator
from dataclasses import dataclass
from importlib import resources
from typing import Any

import yaml
from rich.console import Group
from rich.text import Text

from carnot.agents.base import populate_template
from carnot.agents.default_tools import FinalAnswerTool
from carnot.agents.local_python_executor import (
    BASE_BUILTIN_MODULES,
    LocalPythonExecutor,
    fix_final_answer_code,
)
from carnot.agents.memory import (
    ActionStep,
    AgentMemory,
    CodeOperatorStep,
    MemoryStep,
    SystemPromptStep,
    Timing,
    ToolCall,
)
from carnot.agents.models import ChatMessage, LiteLLMModel, MessageRole
from carnot.agents.monitoring import YELLOW_HEX, AgentLogger, LogLevel
from carnot.agents.tools import BaseTool, Tool
from carnot.agents.utils import (
    AgentError,
    AgentExecutionError,
    AgentGenerationError,
    AgentMaxStepsError,
    AgentParsingError,
    parse_code_blobs,
    truncate_content,
)
from carnot.core.models import LLMCallStats, OperatorStats
from carnot.data.dataset import Dataset
from carnot.operators.physical import PhysicalOperator
from carnot.optimizer.model_ids import get_api_key_for_model

# Default imports that are always authorized for the CodeOperator beyond BASE_BUILTIN_MODULES.
# These cover common data-science and data-manipulation workflows that generated code
# routinely relies on.  Security-sensitive modules (subprocess, socket, shutil, etc.) are
# deliberately excluded.
CODE_OPERATOR_DEFAULT_IMPORTS = [
    "copy",        # Deep copying data structures
    "csv",         # CSV parsing (in-memory)
    "functools",   # Functional programming utilities
    "io",          # StringIO / BytesIO for in-memory streams
    "json",        # JSON parsing / formatting
    "numpy",       # Numerical computing
    "operator",    # Operator functions
    "pandas",      # DataFrames and tabular data manipulation
    "string",      # String constants and helpers
    "textwrap",    # Text formatting utilities
    "typing",      # Type hint helpers
]


@dataclass
class FinalAnswerStep(MemoryStep):
    output: Any
    execution_logs: str
    code: str

@dataclass
class CodeActionOutput:
    code: str
    output: Any
    execution_logs: str
    is_final_answer: bool

class CodeOperator(PhysicalOperator):
    """Multi-step agentic code execution operator.

    The operator uses an LLM to generate Python code, executes it in a
    sandboxed ``LocalPythonExecutor``, observes the output, and iterates
    until the LLM calls ``final_answer()`` or ``max_steps`` is reached.

    Representation invariant:
        - ``tools`` always contains a ``"final_answer"`` key mapping to a
          ``FinalAnswerTool`` instance.
        - ``python_executor`` is a ``LocalPythonExecutor`` instance.
        - ``max_steps >= 1``.

    Abstraction function:
        An instance of this class is a callable that, given input datasets, uses an LLM to
        iteratively generate and execute Python code until a final answer is produced, then
        returns the result wrapped in a new ``Dataset``.
    """
    def __init__(
            self,
            task: str,
            dataset_id: str,
            model_id: str,
            llm_config: dict,
            tools: list[Tool] | None = None,
            additional_authorized_imports: list[str] | None = None,
            max_steps: int = 20,
            logical_op_id: str | None = None,
            logical_op_class_name: str | None = None,
        ):
        super().__init__(logical_op_id=logical_op_id, logical_op_class_name=logical_op_class_name)
        self.task = task
        self.dataset_id = dataset_id
        self.model_id = model_id
        self.llm_config = llm_config
        self._tools = tools or []
        self.model = LiteLLMModel(model_id=model_id, api_key=get_api_key_for_model(model_id, llm_config))
        self.additional_authorized_imports = (
            sorted(set(CODE_OPERATOR_DEFAULT_IMPORTS) | set(additional_authorized_imports or []))
        )
        self.authorized_imports = sorted(set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports))
        self.code_block_tags = ["```python", "```"]
        self.python_executor = self.create_python_executor()
        self.prompt_templates = yaml.safe_load(
            resources.files("carnot.agents.prompts").joinpath("code_operator.yaml").read_text()
        )
        self.memory = AgentMemory("")
        self.logger = AgentLogger(level=LogLevel.INFO)
        self._setup_tools(self._tools)
        self._validate_tools(self._tools)
        self.max_steps = max_steps

    def get_id_params(self):
        id_params = super().get_id_params()
        id_params = {
            "task": self.task,
            "dataset_id": self.dataset_id,
            "model_id": self.model_id,
            **id_params,
        }

        return id_params

    def get_op_params(self):
        op_params = super().get_op_params()
        op_params = {
            "task": self.task,
            "dataset_id": self.dataset_id,
            "model_id": self.model_id,
            "llm_config": self.llm_config,
            "tools": self._tools,
            "additional_authorized_imports": self.additional_authorized_imports,
            "max_steps": self.max_steps,
            **op_params,
        }

        return op_params

    def _setup_tools(self, tools: list[BaseTool]):
        assert all(isinstance(tool, BaseTool) for tool in tools), (
            "All elements must be instance of BaseTool (or a subclass)"
        )
        self.tools = {tool.name: tool for tool in tools}
        self.tools.setdefault("final_answer", FinalAnswerTool())

    def _validate_tools(self, tools: list[BaseTool]):
        tool_names = [tool.name for tool in tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError(
                "Each tool should have a unique name! You passed these duplicate names: "
                f"{[name for name in tool_names if tool_names.count(name) > 1]}"
            )

    def _check_for_anti_patterns(self, code_action: str) -> None:
        """Check for anti-patterns in generated code and raise errors with guidance.

        Requires:
            - *code_action* is a non-empty string of parsed Python code.

        Returns:
            None — returns normally when no anti-patterns are detected.

        Raises:
            AgentParsingError: If the code contains both ``print()`` and
            ``final_answer()`` calls in the same block.
        """
        has_print = bool(re.search(r'\bprint\s*\(', code_action))
        has_final_answer = bool(re.search(r'\bfinal_answer\s*\(', code_action))

        if has_print and has_final_answer:
            error_msg = (
                "Anti-pattern detected: You cannot call both 'print()' and 'final_answer()' "
                "in the same code block.\n\n"
                "You should:\n"
                "1. First use print() to explore and inspect dataset items\n"
                "2. Observe the printed output to understand the data structure\n"
                "3. Then in a SEPARATE step, call final_answer() using the actual keys/values you discovered\n\n"
                "Please split these into separate steps."
            )
            raise AgentParsingError(error_msg, self.logger)

    def _finalize_step(self, memory_step: ActionStep):
        memory_step.timing.end_time = time.time()

    def __enter__(self):
        """Method used to initialize resources when entering a `with` context."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Method used to clean up resources when exiting a `with` context."""
        self.cleanup()

    def cleanup(self):
        """Release resources held by the Python executor.

        Safe to call multiple times.
        """
        if hasattr(self.python_executor, "cleanup"):
            self.python_executor.cleanup()

    def create_python_executor(self) -> LocalPythonExecutor:
        return LocalPythonExecutor(self.additional_authorized_imports)

    def write_memory_to_messages(
        self,
        summary_mode: bool = False,
    ) -> list[ChatMessage]:
        """
        Reads past llm_outputs, actions, and observations or errors from the memory into a series of messages
        that can be used as input to the LLM. Adds a number of keywords (such as PLAN, error, etc) to help
        the LLM.
        """
        messages = self.memory.system_prompt.to_messages(summary_mode=summary_mode)
        for memory_step in self.memory.steps:
            messages.extend(memory_step.to_messages(summary_mode=summary_mode))
        return messages

    def provide_final_answer(self, task: str) -> ChatMessage:
        """Ask the LLM for a final answer based on the accumulated memory.

        Called when ``max_steps`` is reached without an explicit
        ``final_answer()`` call in generated code.

        Requires:
            - ``self.memory`` has at least one step recorded.

        Returns:
            A ``ChatMessage`` containing the LLM's final-answer response.

        Raises:
            None — generation errors are caught and returned as a
            ``ChatMessage`` with the error text.
        """
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=[
                    {
                        "type": "text",
                        "text": self.prompt_templates["final_answer"]["pre_messages"],
                    }
                ],
            )
        ]
        messages += self.write_memory_to_messages()[1:]
        messages.append(
            ChatMessage(
                role=MessageRole.USER,
                content=[
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["final_answer"]["post_messages"], variables={"task": task}
                        ),
                    }
                ],
            )
        )
        try:
            chat_message: ChatMessage = self.model.generate(messages)
            return chat_message
        except Exception as e:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content=[{"type": "text", "text": f"Error in generating final LLM output: {e}"}],
            )

    def _handle_max_steps_reached(self):
        action_step_start_time = time.time()
        final_answer = self.provide_final_answer(self.task)
        final_memory_step = ActionStep(
            step_number=self.step_number,
            error=AgentMaxStepsError("Reached max steps.", self.logger),
            timing=Timing(start_time=action_step_start_time, end_time=time.time()),
            token_usage=final_answer.token_usage,
        )
        final_memory_step.final_output = final_answer.content
        self._finalize_step(final_memory_step)
        self.memory.steps.append(final_memory_step)
        return final_answer.content
    
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
        """Run one generate → parse → execute cycle.

        Requires:
            - *memory_step* is a fresh ``ActionStep`` for this iteration.
            - *input_datasets* is the current dataset dict.

        Returns:
            Yields a ``ToolCall`` (the parsed code) followed by a
            ``CodeActionOutput`` (execution result).

        Raises:
            AgentGenerationError: If the LLM call fails.
            AgentParsingError: If the output cannot be parsed as code.
            AgentExecutionError: If the generated code raises at runtime.
        """
        system_prompt = populate_template(
            self.prompt_templates["code_gen_prompt"],
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
        """Execute the agentic code loop and return the resulting datasets.

        The operator injects *input_datasets* and its tools into the
        sandboxed executor, then iterates through generate/execute steps
        until the LLM calls ``final_answer()`` or ``max_steps`` is reached.

        Requires:
            - *input_datasets* is a non-empty ``dict[str, Dataset]``.

        Returns:
            A tuple ``(output_datasets, stats)`` where *output_datasets*
            is a new ``dict[str, Dataset]`` with an additional entry keyed
            by ``self.dataset_id``, and *stats* is an
            :class:`OperatorStats` summarising all LLM calls made across
            the agentic loop.

        Raises:
            AgentGenerationError: If the LLM fails on the first step.
            AssertionError: If the run stream does not end with a
            ``FinalAnswerStep``.
        """
        op_start = time.perf_counter()
        
        self.python_executor.send_variables(variables={"input_datasets": input_datasets})
        self.python_executor.send_tools({**self.tools})

        # use an LLM to generate and execute code based on the task and input datasets.
        steps = list(self._run_stream(input_datasets))
        assert isinstance(steps[-1], FinalAnswerStep)
        output_state = steps[-1].output # TODO: enforce that this is a proper state dictionary
        output_code = steps[-1].code
        # TODO: ? output_execution_logs = steps[-1].execution_logs

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

        # If the code operator produced a list of items under the "final_items" key,
        # surface them as the Dataset's items so downstream operators can consume them.
        output_items = None
        if isinstance(output_state, dict) and "final_items" in output_state:
            candidate = output_state["final_items"]
            if isinstance(candidate, list) and all(isinstance(item, dict) for item in candidate):
                output_items = candidate

        # create new dataset and return it with the input datasets
        output_dataset = Dataset(
            name=self.dataset_id,
            annotation=f"Code operator output for task: {self.task}",
            items=output_items,
            code=output_code,
            code_state=output_state,
        )
        output_datasets = {**input_datasets, output_dataset.name: output_dataset}

        op_stats = OperatorStats(
            operator_name="Code",
            operator_id=self.dataset_id,
            wall_clock_secs=time.perf_counter() - op_start,
            llm_calls=all_call_stats,
            items_in=sum(len(ds.items) for ds in input_datasets.values()),
            items_out=len(output_dataset.items) if output_dataset.items else 0,
        )

        return output_datasets, op_stats
