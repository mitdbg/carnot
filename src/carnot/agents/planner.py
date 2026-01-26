import time
from collections.abc import Generator
from importlib import resources

import yaml
from rich.console import Group
from rich.text import Text

from carnot.agents.base import ActionOutput, BaseAgent, populate_template
from carnot.agents.local_python_executor import (
    BASE_BUILTIN_MODULES,
    LocalPythonExecutor,
    PythonExecutor,
    fix_final_answer_code,
)
from carnot.agents.memory import (
    ActionStep,
    CompilerTaskStep,
    FinalAnswerStep,
    PlannerTaskStep,
    PlanningStep,
    SystemPromptStep,
    ToolCall,
)
from carnot.agents.models import ChatMessage, ChatMessageStreamDelta
from carnot.agents.monitoring import YELLOW_HEX, LogLevel, Timing
from carnot.agents.utils import (
    AgentError,
    AgentExecutionError,
    AgentGenerationError,
    AgentParsingError,
    parse_code_blobs,
    parse_plan,
)
from carnot.data.dataset import Dataset
from carnot.operators import LOGICAL_OPERATORS

# from carnot.plan.logical import LogicalPlan


class Planner(BaseAgent):
    """
    An agent specialized in generating and compiling logical execution plans.
    """
    def __init__(self, *args, **kwargs):
        prompt_templates = yaml.safe_load(
            resources.files("carnot.agents.prompts").joinpath("planner_agent.yaml").read_text()
        )
        self.plan_tags = ["<begin_plan>", "<end_plan>"]
        self.code_block_tags = ["```python", "```"]
        self.additional_authorized_imports = ["carnot"]
        self.authorized_imports = sorted(set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports))
        super().__init__(*args, prompt_templates=prompt_templates, **kwargs)
        self.python_executor = self.create_python_executor()
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()

    def cleanup(self):
        """Clean up resources used by the agent, such as the remote Python executor."""
        if hasattr(self.python_executor, "cleanup"):
            self.python_executor.cleanup()

    def create_python_executor(self) -> PythonExecutor:
        return LocalPythonExecutor(self.additional_authorized_imports)

    def initialize_system_prompt(self) -> str:
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={
                "plan_opening_tag": self.plan_tags[0],
                "plan_closing_tag": self.plan_tags[1],
            },
        )
        return system_prompt

    def _run_stream(
        self, phase: str = "planning",
    ) -> Generator[ActionStep | PlanningStep | FinalAnswerStep | ChatMessageStreamDelta]:
        self.step_number = 1
        returned_final_answer = False
        while not returned_final_answer:
            # Start action step!
            action_step = PlanningStep(model_input_messages=[], model_output_message="", plan="", timing=Timing(start_time=time.time()))
            generator = self._step_generating_logical_plan_stream(action_step)
            if phase == "logical-compilation":
                action_step = ActionStep(step_number=self.step_number, timing=Timing(start_time=time.time()))
                generator = self._step_compiling_logical_plan_stream(action_step)
            elif phase == "physical-compilation":
                action_step = None
                generator = None

            self.logger.log_rule(f"Step {self.step_number}", level=LogLevel.INFO)
            try:
                for output in generator:
                    # Yield all
                    yield output

                    if isinstance(output, ActionOutput) and output.is_final_answer:
                        final_answer = output.output
                        self.logger.log(
                            Text(f"Final answer: {final_answer}", style=f"bold {YELLOW_HEX}"),
                            level=LogLevel.INFO,
                        )

                        if self.final_answer_checks:
                            self._validate_final_answer(final_answer)
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

        yield FinalAnswerStep(final_answer)

    def _step_generating_logical_plan_stream(self, memory_step: PlanningStep) -> Generator[ActionOutput]:
        # convert the steps to messages
        memory_messages = self.write_memory_to_messages()
        input_messages = memory_messages.copy()

        ### Generate model output ###
        memory_step.model_input_messages = input_messages
        stop_sequences = [self.plan_tags[1]]
        try:
            chat_message: ChatMessage = self.model.generate(input_messages, stop_sequences=stop_sequences)
            memory_step.model_output_message = chat_message
            output_text = chat_message.content
            self.logger.log_markdown(
                content=output_text,
                title="Output message of the LLM:",
                level=LogLevel.DEBUG,
            )

            # This adds the end code sequence (i.e. the closing plan tag) to the history.
            # This will nudge subsequent LLM calls to finish with this end code sequence, thus efficiently stopping generation.
            if output_text and not output_text.strip().endswith(self.plan_tags[1]):
                output_text += self.plan_tags[1]
                memory_step.model_output_message.content = output_text

            memory_step.token_usage = chat_message.token_usage
        except Exception as e:
            raise AgentGenerationError(f"Error in generating model output:\n{e}", self.logger) from e

        ### Parse output ###
        is_final_answer = False
        try:
            memory_step.plan = parse_plan(output_text, self.plan_tags)
            is_final_answer = True
        except Exception as e:
            error_msg = f"Error in code parsing:\n{e}\nMake sure to provide a correctly formatted plan."
            raise AgentParsingError(error_msg, self.logger) from e
        
        yield ActionOutput(output=memory_step.plan, is_final_answer=is_final_answer)

    def _step_compiling_logical_plan_stream(self, memory_step: ActionStep) -> Generator[ActionOutput]:
        # convert the steps to messages
        memory_messages = self.write_memory_to_messages()
        input_messages = memory_messages.copy()

        ### Generate model output ###
        memory_step.model_input_messages = input_messages
        stop_sequences = []
        try:
            chat_message: ChatMessage = self.model.generate(input_messages, stop_sequences=stop_sequences)
            memory_step.model_output_message = chat_message
            output_text = chat_message.content
            self.logger.log_markdown(
                content=output_text,
                title="Output message of the LLM:",
                level=LogLevel.DEBUG,
            )

            # This adds the end code sequence (i.e. the closing code block tag) to the history.
            # This will nudge subsequent LLM calls to finish with this end code sequence, thus efficiently stopping generation.
            if output_text and not output_text.strip().endswith(self.code_block_tags[1]):
                output_text += self.code_block_tags[1]
                memory_step.model_output_message.content = output_text

            memory_step.token_usage = chat_message.token_usage
            memory_step.model_output = output_text
        except Exception as e:
            raise AgentGenerationError(f"Error in generating model output:\n{e}", self.logger) from e

        # TODO: in theory we should check to make sure the logical plan compiles
        ### Parse output ###
        try:
            code_action = parse_code_blobs(output_text, self.code_block_tags)
            code_action = fix_final_answer_code(code_action)
            memory_step.code_action = code_action
        except Exception as e:
            error_msg = f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
            raise AgentParsingError(error_msg, self.logger) from e

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

        truncated_output = str(code_output.output)
        observation += "Last output from code snippet:\n" + truncated_output
        memory_step.observations = observation

        if not code_output.is_final_answer:
            execution_outputs_console += [
                Text(
                    f"Out: {truncated_output}",
                ),
            ]
        self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
        memory_step.action_output = code_output.output
        yield ActionOutput(output=code_output.output, is_final_answer=code_output.is_final_answer)

    def _step_compiling_physical_plan_stream(self, memory_step: ActionStep) -> Generator[ActionOutput]:
        pass  # To be implemented in the future for physical plan compilation

    def generate_logical_plan(self, query: str, datasets: list[Dataset], indices: list, tools: list, memories: list, conversation_context: str) -> str:
        """
        Generate a logical execution plan in natural language.
        """
        # create system prompt and task prompt for the planner
        self.memory.system_prompt = SystemPromptStep(system_prompt=self.system_prompt)
        self.logger.log_task(
            content=query.strip(),
            subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
            level=LogLevel.INFO,
            title=self.name if hasattr(self, "name") else None,
        )
        self.memory.steps.append(PlannerTaskStep(task=query, datasets=datasets))

        # steps = list(self._run_stream(generating_logical_plan=True))
        steps = list(self._run_stream(phase="planning"))
        assert isinstance(steps[-1], FinalAnswerStep)
        output = steps[-1].output

        return output

    def compile_logical_plan(self, query: str, datasets: list[Dataset], nl_plan: str) -> dict:
        """
        Compile a natural language plan into a LogicalPlan object.
        """
        # create system prompt and task prompt for the planner
        system_prompt = populate_template(
            self.prompt_templates["logical_compiler_prompt"],
            variables={
                "logical_operators": {op: op.desc() for op in LOGICAL_OPERATORS},
                "plan_opening_tag": self.code_block_tags[0],
                "plan_closing_tag": self.code_block_tags[1],
            },
        )
        self.memory.system_prompt = SystemPromptStep(system_prompt=system_prompt)
        self.logger.log_task(
            content=nl_plan.strip(),
            subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
            level=LogLevel.INFO,
            title=self.name if hasattr(self, "name") else None,
        )
        self.memory.steps.append(CompilerTaskStep(task=query, datasets=datasets, nl_plan=nl_plan))

        self.python_executor.send_variables(variables=self.state)
        self.python_executor.send_tools({**self.tools})

        # steps = list(self._run_stream(generating_logical_plan=False))
        steps = list(self._run_stream(phase="logical-compilation"))
        assert isinstance(steps[-1], FinalAnswerStep)
        output = steps[-1].output

        return output

    def compile_physical_plan(self, logical_plan: dict):
        """
        Compile a logical plan into a physical execution plan.
        """
        logical_plan = LogicalPlan.from_dict(logical_plan)
