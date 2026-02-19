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
    AgentMemory,
    CompilerTaskStep,
    DataDiscoveryTaskStep,
    FinalAnswerStep,
    PlannerTaskStep,
    PlanningStep,
    ToolCall,
)
from carnot.agents.models import ChatMessageStreamDelta
from carnot.agents.monitoring import YELLOW_HEX, LogLevel, Timing
from carnot.agents.utils import (
    AgentError,
    AgentExecutionError,
    AgentGenerationError,
    AgentParsingError,
    parse_code_blobs,
    parse_plan,
)
from carnot.conversation.conversation import Conversation
from carnot.data.dataset import Dataset
from carnot.operators import LOGICAL_OPERATORS


class Planner(BaseAgent):
    """
    An agent specialized in generating and compiling logical execution plans.
    Each phase (data discovery, NL planning, compilation) uses its own isolated memory.
    """
    def __init__(self, *args, **kwargs):
        prompt_templates = yaml.safe_load(
            resources.files("carnot.agents.prompts").joinpath("planner_agent.yaml").read_text()
        )
        self.plan_tags = ["<begin_plan>", "<end_plan>"]
        self.report_tags = ["<begin_report>", "<end_report>"]
        self.code_block_tags = ["```python", "```"]
        self.additional_authorized_imports = ["carnot"]
        self.authorized_imports = sorted(set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports))
        super().__init__(*args, prompt_templates=prompt_templates, **kwargs)

        # phase-specific memories to prevent context bleeding between phases
        self.data_discovery_memory = None
        self.planning_memory = None
        self.compilation_memory = None

        self.python_executor = self.create_python_executor()
        self.max_steps = 5
    
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
        # NOTE: we won't end up using this system prompt (stored at self.system_prompt);
        #       instead, we will template the prompt(s) dynamically within the relevant functions
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={
                "code_opening_tag": self.code_block_tags[0],
                "code_closing_tag": self.code_block_tags[1],
                "plan_opening_tag": self.plan_tags[0],
                "plan_closing_tag": self.plan_tags[1],
                "report_opening_tag": self.report_tags[0],
                "report_closing_tag": self.report_tags[1],
                "data_discovery_report": "",
                "has_conversation": False,
            },
        )
        return system_prompt

    def _run_stream(
        self,
        phase: str = "planning",
        memory: AgentMemory | None = None,
    ) -> Generator[ActionStep | PlanningStep | FinalAnswerStep | ChatMessageStreamDelta]:
        """
        Execute a planning phase using the provided memory.
        
        Args:
            phase: Phase name ("planning", "logical-compilation", "data-discovery")
            memory: AgentMemory instance specific to this phase. If None, uses self.memory.
        """
        # use provided memory or fall back to self.memory
        if memory is None:
            memory = self.memory

        # temporarily swap memory for helper method compatibility
        original_memory = self.memory
        self.memory = memory

        try:
            self.step_number = 1
            returned_final_answer = False
            while not returned_final_answer and self.step_number <= self.max_steps:
                # Start action step based on phase
                if phase == "planning":
                    action_step = ActionStep(step_number=self.step_number, timing=Timing(start_time=time.time()))
                    generator = self._step_generating_logical_plan_stream(action_step)
                elif phase == "logical-compilation":
                    action_step = ActionStep(step_number=self.step_number, timing=Timing(start_time=time.time()))
                    generator = self._step_compiling_logical_plan_stream(action_step)
                elif phase == "data-discovery":
                    action_step = ActionStep(step_number=self.step_number, timing=Timing(start_time=time.time()))
                    generator = self._step_data_discovery_stream(action_step)
                else:
                    raise ValueError(f"Unknown phase: {phase}")

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

            if not returned_final_answer:
                final_answer = "The agent did not return a final answer within the maximum number of steps."

            yield FinalAnswerStep(final_answer)
        finally:
            # restore original memory
            self.memory = original_memory

    def _generate_model_output(
        self,
        memory_step: ActionStep,
        stop_sequences: list[str] | None = None
    ) -> str:
        """Generate model output and store in memory step."""
        memory_messages = self.write_memory_to_messages()
        memory_step.model_input_messages = memory_messages.copy()

        try:
            chat_message = self.model.generate(
                memory_messages,
                stop_sequences=stop_sequences or []
            )
            memory_step.model_output_message = chat_message
            output_text = chat_message.content
            
            self.logger.log_markdown(
                content=output_text,
                title="Output message of the LLM:",
                level=LogLevel.DEBUG,
            )

            memory_step.token_usage = chat_message.token_usage
            memory_step.model_output = output_text
            return output_text
        except Exception as e:
            raise AgentGenerationError(
                f"Error in generating model output:\n{e}", 
                self.logger
            ) from e

    def _try_parse_final_answer(
        self, 
        output_text: str, 
        tags: list[str], 
        memory_step: ActionStep
    ) -> tuple[bool, str | None]:
        """
        Try to parse output as final answer using provided tags.
        
        Returns:
            (success, parsed_content): Tuple indicating if parsing succeeded
        """
        try:
            final_answer = parse_plan(output_text, tags)
            memory_step.action_output = final_answer
            return True, final_answer
        except Exception:
            return False, None

    def _parse_and_prepare_code(
        self, 
        output_text: str, 
        memory_step: ActionStep,
        error_context: str = "code blobs"
    ) -> tuple[str, ToolCall]:
        """
        Parse code from output text and create tool call.
        
        Args:
            output_text: Raw LLM output
            memory_step: Current action step
            error_context: Context string for error messages
            
        Returns:
            (code_action, tool_call): Parsed code and corresponding tool call
        """
        try:
            code_action = parse_code_blobs(output_text, self.code_block_tags)
            code_action = fix_final_answer_code(code_action)
            memory_step.code_action = code_action
        except Exception as e:
            error_msg = f"Error in parsing output:\n{e}\nMake sure to provide {error_context}."
            raise AgentParsingError(error_msg, self.logger) from e

        tool_call = ToolCall(
            name="python_interpreter",
            arguments=code_action,
            id=f"call_{len(self.memory.steps)}",
        )
        memory_step.tool_calls = [tool_call]

        return code_action, tool_call

    def _execute_code_and_collect_output(
        self, 
        code_action: str, 
        memory_step: ActionStep
    ) -> tuple:
        """
        Execute code and collect execution logs and output.
        
        Returns:
            (code_output, observation, execution_outputs_console): Tuple of code output,
            observation string, and console outputs for logging
        """
        self.logger.log_code(
            title="Executing parsed code:", 
            content=code_action, 
            level=LogLevel.INFO
        )

        try:
            code_output = self.python_executor(code_action)
            execution_outputs_console = []

            if len(code_output.logs) > 0:
                execution_outputs_console += [
                    Text("Execution logs:", style="bold"),
                    Text(code_output.logs),
                ]

            observation = "Execution logs:\n" + code_output.logs

            return code_output, observation, execution_outputs_console

        except Exception as e:
            # Handle print outputs if available
            execution_outputs_console = []
            if hasattr(self.python_executor, "state") and "_print_outputs" in self.python_executor.state:
                execution_logs = str(self.python_executor.state["_print_outputs"])
                if len(execution_logs) > 0:
                    execution_outputs_console = [
                        Text("Execution logs:", style="bold"),
                        Text(execution_logs),
                    ]
                    memory_step.observations = "Execution logs:\n" + execution_logs
                    self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)

            # Handle import errors with helpful message
            error_msg = str(e)
            if "Import of " in error_msg and " is not allowed" in error_msg:
                self.logger.log(
                    "[bold red]Warning: Code execution failed due to an unauthorized import - "
                    "Consider passing said import under `additional_authorized_imports`.",
                    level=LogLevel.INFO,
                )

            raise AgentExecutionError(error_msg, self.logger) from e

    def _finalize_code_execution_step(
        self,
        code_output,
        observation: str,
        execution_outputs_console: list,
        memory_step: ActionStep,
        is_final_answer: bool | None = None
    ) -> ActionOutput:
        """
        Finalize code execution step with observations and output.
        
        Args:
            code_output: Output from python executor
            observation: Accumulated observation string
            execution_outputs_console: Console outputs for logging
            memory_step: Current action step
            is_final_answer: Override for is_final_answer. If None, use code_output.is_final_answer
        
        Returns:
            ActionOutput for yielding
        """
        truncated_output = str(code_output.output)
        observation += "Last output from code snippet:\n" + truncated_output
        memory_step.observations = observation

        # Determine finality
        if is_final_answer is None:
            use_final_answer = getattr(code_output, 'is_final_answer', False)
        else:
            use_final_answer = is_final_answer

        # Only add output to console if not final answer
        if not use_final_answer:
            execution_outputs_console += [
                Text(f"Out: {truncated_output}"),
            ]

        self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
        memory_step.action_output = code_output.output

        return ActionOutput(
            output=code_output.output, 
            is_final_answer=use_final_answer
        )

    def _step_with_code_and_final_answer(
        self,
        memory_step: ActionStep,
        final_answer_tags: list[str] | None = None,
        parse_code_only: bool = False,
        add_closing_tag: bool = False
    ) -> Generator[ActionOutput | ToolCall]:
        """
        Generic streaming step that supports:
        - Parsing final answer (plan/report) OR code
        - Code execution with error handling
        
        Args:
            memory_step: Current action step
            final_answer_tags: Tags for parsing final answer (None means code-only)
            parse_code_only: If True, skip final answer parsing
            add_closing_tag: If True, add closing code tag to output
        """
        # Generate model output
        output_text = self._generate_model_output(memory_step)

        # Add closing tag if needed (compilation phase)
        if add_closing_tag and output_text and not output_text.strip().endswith(self.code_block_tags[1]):
            output_text += self.code_block_tags[1]
            memory_step.model_output_message.content = output_text

        # Try to parse as final answer (if applicable)
        if not parse_code_only and final_answer_tags:
            success, final_answer = self._try_parse_final_answer(
                output_text, 
                final_answer_tags, 
                memory_step
            )
            if success:
                yield ActionOutput(output=final_answer, is_final_answer=True)
                return

        # Parse as code
        error_context = "correct code blobs" if parse_code_only else "either code blobs or a final answer"
        code_action, tool_call = self._parse_and_prepare_code(
            output_text, 
            memory_step,
            error_context
        )
        yield tool_call

        # Execute code
        code_output, observation, execution_outputs_console = self._execute_code_and_collect_output(
            code_action, 
            memory_step
        )

        # Finalize and yield result
        # Use explicit False for planning/discovery, use code_output.is_final_answer for compilation
        is_final_override = None if parse_code_only else False
        action_output = self._finalize_code_execution_step(
            code_output,
            observation,
            execution_outputs_console,
            memory_step,
            is_final_answer=is_final_override
        )
        yield action_output

    def _step_generating_logical_plan_stream(self, memory_step: ActionStep) -> Generator[ActionOutput]:
        """Handle natural language planning with code execution support."""
        yield from self._step_with_code_and_final_answer(
            memory_step=memory_step,
            final_answer_tags=self.plan_tags,
            parse_code_only=False,
            add_closing_tag=False
        )

    def _step_compiling_logical_plan_stream(self, memory_step: ActionStep) -> Generator[ActionOutput]:
        """Handle logical plan compilation - code only."""
        yield from self._step_with_code_and_final_answer(
            memory_step=memory_step,
            final_answer_tags=None,
            parse_code_only=True,
            add_closing_tag=True
        )

    def _step_data_discovery_stream(self, memory_step: ActionStep) -> Generator[ActionOutput]:
        """Handle data discovery with code execution support."""
        yield from self._step_with_code_and_final_answer(
            memory_step=memory_step,
            final_answer_tags=self.report_tags,
            parse_code_only=False,
            add_closing_tag=False
        )

    def _setup_phase_execution(
        self,
        phase_type: str,
        prompt_template_key: str,
        template_vars: dict,
        task_step_class: type,
        task_kwargs: dict,
        conversation: Conversation | None = None,
        plan_type_for_history: str | None = None,
        log_title: str | None = None,
        log_content_key: str = "task"
    ) -> AgentMemory:
        """
        Setup system prompt, memory, and Python executor for phase execution.
        Creates and returns a fresh AgentMemory instance specific to this phase.
        
        Args:
            phase_type: Phase identifier ('data-discovery', 'planning', 'compilation')
            prompt_template_key: Key in self.prompt_templates
            template_vars: Variables for template population
            task_step_class: Memory step class (e.g., PlannerTaskStep)
            task_kwargs: Kwargs for task step initialization
            conversation: Optional conversation history
            plan_type_for_history: Type of plan to retrieve from conversation (e.g., 'natural-language-plan')
            log_title: Optional title for task logging
            log_content_key: Key in task_kwargs to use for log content
            
        Returns:
            AgentMemory instance specific to this phase
        """
        # create system prompt
        system_prompt = populate_template(
            self.prompt_templates[prompt_template_key],
            variables=template_vars,
        )

        # create fresh memory for this phase
        phase_memory = AgentMemory(system_prompt)

        # store reference based on phase type
        if phase_type == "data-discovery":
            self.data_discovery_memory = phase_memory
        elif phase_type == "planning":
            self.planning_memory = phase_memory
        elif phase_type == "compilation":
            self.compilation_memory = phase_memory

        # add previous plan from conversation if applicable
        if conversation and plan_type_for_history:
            latest_plan = conversation.get_latest_agent_plan(plan_type_for_history)
            if latest_plan:
                phase_memory.steps.append(latest_plan)

        # log task
        log_content = task_kwargs.get(log_content_key, "")
        self.logger.log_task(
            content=str(log_content).strip(),
            subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
            level=LogLevel.INFO,
            title=log_title or (self.name if hasattr(self, "name") else None),
        )

        # add task step to phase memory
        task_step = task_step_class(**task_kwargs)
        phase_memory.steps.append(task_step)

        # setup Python executor state
        datasets_dict = {dataset.name: dataset for dataset in task_kwargs.get("datasets", [])}
        conversation_list = conversation.to_dict_list() if conversation else []
        state = {
            "datasets": datasets_dict,
            "conversation": conversation_list,
            **self.state
        }
        self.python_executor.send_variables(variables=state)
        self.python_executor.send_tools({**self.tools})

        return phase_memory

    def generate_logical_plan(
        self, 
        query: str, 
        datasets: list[Dataset], 
        indices: list, 
        tools: list, 
        memories: list, 
        data_discovery_report: str | None = None,
        conversation: Conversation | None = None
    ) -> str:
        """Generate a logical execution plan in natural language."""
        # setup execution with phase-specific memory
        planning_memory = self._setup_phase_execution(
            phase_type="planning",
            prompt_template_key="system_prompt",
            template_vars={
                "code_opening_tag": self.code_block_tags[0],
                "code_closing_tag": self.code_block_tags[1],
                "plan_opening_tag": self.plan_tags[0],
                "plan_closing_tag": self.plan_tags[1],
                "data_discovery_report": str(data_discovery_report),
                "has_conversation": conversation is not None,
            },
            task_step_class=PlannerTaskStep,
            task_kwargs={"task": query, "datasets": datasets},
            conversation=conversation,
            plan_type_for_history="natural-language-plan"
        )

        # run and return with isolated memory
        steps = list(self._run_stream(phase="planning", memory=planning_memory))
        assert isinstance(steps[-1], FinalAnswerStep)
        return steps[-1].output

    def compile_logical_plan(
        self, 
        query: str, 
        datasets: list[Dataset], 
        nl_plan: str, 
        data_discovery_report: str | None = None,
        conversation: Conversation | None = None,
    ) -> dict:
        """Compile a natural language plan into a LogicalPlan object."""
        # setup execution with phase-specific memory
        compilation_memory = self._setup_phase_execution(
            phase_type="compilation",
            prompt_template_key="logical_compiler_prompt",
            template_vars={
                "logical_operators": {op: op.desc() for op in LOGICAL_OPERATORS},
                "code_opening_tag": self.code_block_tags[0],
                "code_closing_tag": self.code_block_tags[1],
                "data_discovery_report": str(data_discovery_report),
                "has_conversation": conversation is not None,
            },
            task_step_class=CompilerTaskStep,
            task_kwargs={"task": query, "datasets": datasets, "nl_plan": nl_plan},
            conversation=conversation,
            plan_type_for_history="logical-plan",
            log_content_key="nl_plan"
        )

        # run and return with isolated memory
        steps = list(self._run_stream(phase="logical-compilation", memory=compilation_memory))
        assert isinstance(steps[-1], FinalAnswerStep)
        return steps[-1].output

    def search_for_relevant_data(
        self, 
        query: str, 
        datasets: list[Dataset], 
        indices: list, 
        tools: list, 
        memories: list, 
        conversation: Conversation | None = None
    ) -> str:
        """Perform preliminary data discovery to identify relevant data for the query."""
        # setup execution with phase-specific memory
        data_discovery_memory = self._setup_phase_execution(
            phase_type="data-discovery",
            prompt_template_key="data_discovery_prompt",
            template_vars={
                "code_opening_tag": self.code_block_tags[0],
                "code_closing_tag": self.code_block_tags[1],
                "report_opening_tag": self.report_tags[0],
                "report_closing_tag": self.report_tags[1],
                "has_conversation": conversation is not None,
            },
            task_step_class=DataDiscoveryTaskStep,
            task_kwargs={"task": query, "datasets": datasets},
            conversation=conversation,
            plan_type_for_history=None,
            log_title="Data Discovery"
        )

        # run and return with isolated memory
        steps = list(self._run_stream(phase="data-discovery", memory=data_discovery_memory))
        assert isinstance(steps[-1], FinalAnswerStep)
        return steps[-1].output
