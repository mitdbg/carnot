import re
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
from carnot.execution.progress import PlanningProgress
from carnot.index import INDEX_TYPES
from carnot.operators import LOGICAL_OPERATORS

# Number of steps remaining at which to warn the agent to wrap up
MAX_STEPS_WARNING_THRESHOLD = 3


class Planner(BaseAgent):
    """An agent that generates logical execution plans from natural language queries.

    The Planner generates logical plans (as code) and can delegate data
    discovery tasks to a managed ``DataDiscoveryAgent``.  The flow is:

    1. ``generate_logical_plan()`` — creates a code-based logical plan
       using semantic operators.  May call the ``DataDiscoveryAgent``
       to understand dataset schemas before planning.
    2. ``paraphrase_logical_plan()`` — translates the code-based plan
       into a natural-language summary for user presentation.

    Each phase operates with its own isolated ``AgentMemory`` to prevent
    context bleeding between planning and paraphrasing.

    Representation invariant:
        - ``_datasets`` is a non-empty list of ``Dataset`` objects.
        - ``_data_discovery_agent`` is a ``DataDiscoveryAgent`` managed
          by this planner.
        - ``plan_tags`` is ``["<begin_plan>", "<end_plan>"]``.
        - ``code_block_tags`` is ``["```python", "\\n```"]``.

    Abstraction function:
        An instance of this class is an agent that, given a natural-language query
        and a set of datasets, produces (1) a code-based logical plan and (2) a
        natural-language paraphrase of that plan.

    Args:
        datasets: List of ``Dataset`` objects available for planning.
        model: The LLM model to use.
        tools: Optional list of additional tools.
        managed_agents: Optional list of additional managed agents.
        **kwargs: Additional arguments passed to ``BaseAgent``.
    """
    def __init__(
        self,
        datasets: list[Dataset],
        *args,
        managed_agents: list | None = None,
        **kwargs
    ):
        # Store datasets for use in planning
        self._datasets = datasets

        # Load prompt templates
        prompt_templates = yaml.safe_load(
            resources.files("carnot.agents.prompts").joinpath("planner_agent.yaml").read_text()
        )

        self.plan_tags = ["<begin_plan>", "<end_plan>"]
        # Use code block tags where closing tag is NOT contained in opening tag
        # This allows us to use the closing tag as a stop sequence
        self.code_block_tags = ["```python", "\n```"]
        self.additional_authorized_imports = ["carnot"]
        self.authorized_imports = sorted(set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports))
        
        # Create the DataDiscoveryAgent as a managed agent
        # Import here to avoid circular imports
        from carnot.agents.data_discovery import DataDiscoveryAgent
        
        # Get the model from kwargs (required for creating managed agent)
        model = kwargs.get("model") or (args[1] if len(args) > 1 else None)
        if model is None:
            raise ValueError("Planner requires a 'model' argument")
        
        self._data_discovery_agent = DataDiscoveryAgent(
            datasets=datasets,
            model=model,
            max_steps=10,
        )
        
        # Combine with any user-provided managed agents
        all_managed_agents = [self._data_discovery_agent]
        if managed_agents:
            all_managed_agents.extend(managed_agents)
        
        super().__init__(*args, prompt_templates=prompt_templates, managed_agents=all_managed_agents, **kwargs)

        # Phase-specific memories to prevent context bleeding between phases
        self.planning_memory = None
        self.paraphrase_memory = None

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
        """Initialize the system prompt with template variables including managed agents."""
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={
                "code_opening_tag": self.code_block_tags[0],
                "code_closing_tag": self.code_block_tags[1],
                "plan_opening_tag": self.plan_tags[0],
                "plan_closing_tag": self.plan_tags[1],
                "logical_operators": {op: op.desc() for op in LOGICAL_OPERATORS},
                "managed_agents": self.managed_agents,
                "has_conversation": False,
                "index_types": {cls.__name__: cls.description for cls in INDEX_TYPES},
            },
        )
        return system_prompt

    def _run_stream(
        self,
        phase: str = "planning",
        memory: AgentMemory | None = None,
    ) -> Generator[ActionStep | PlanningStep | FinalAnswerStep | ChatMessageStreamDelta]:
        """Execute a planning phase, yielding intermediate steps.

        Requires:
            - *phase* is ``"planning"`` or ``"paraphrase"``.
            - *memory* (if supplied) is a properly initialised
              ``AgentMemory`` with system prompt and task step.

        Returns:
            A generator of step objects, always ending with a
            ``FinalAnswerStep``.

        Raises:
            AgentGenerationError: If the LLM fails to produce valid output.
            ValueError: If *phase* is not recognised.
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
                elif phase == "paraphrase":
                    action_step = ActionStep(step_number=self.step_number, timing=Timing(start_time=time.time()))
                    generator = self._step_paraphrase_stream(action_step)
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
                    
                    # Countdown warning for the last few steps
                    # After incrementing, step_number is the NEXT step to run
                    # steps_remaining = max_steps - (next_step - 1) = max_steps - step_number + 1
                    steps_remaining = self.max_steps - self.step_number + 1
                    
                    # Debug logging
                    self.logger.log(
                        f"[Planner] Step {self.step_number - 1} completed, {steps_remaining} steps remaining (max={self.max_steps})",
                        level=LogLevel.INFO,
                    )
                    
                    if 1 <= steps_remaining <= MAX_STEPS_WARNING_THRESHOLD:
                        step_word = "step" if steps_remaining == 1 else "steps"
                        warning_msg = (
                            f"\n\n⚠️ Warning: You have {steps_remaining} {step_word} remaining. "
                            f"Return your final answer with final_answer() soon."
                        )
                        # Modify the action step's observations directly
                        action_step.observations = (action_step.observations or "") + warning_msg
                        # Log that we're adding the warning
                        self.logger.log(
                            "[Planner] Added max_steps warning to observations",
                            level=LogLevel.INFO,
                        )

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
        """
        Generate model output and store in memory step.
        
        Uses stop sequences to prevent the model from continuing after a code block.
        For models that don't support stop sequences (e.g., gpt-5 series), we truncate
        the output after the first complete code block.
        """
        memory_messages = self.write_memory_to_messages()
        memory_step.model_input_messages = memory_messages.copy()

        # Build stop sequences - include the code block closing tag
        # since it's not contained in the opening tag
        default_stop_sequences = ["Observation:", "Calling tools:"]
        if self.code_block_tags[1] not in self.code_block_tags[0]:
            default_stop_sequences.append(self.code_block_tags[1])

        try:
            chat_message = self.model.generate(
                memory_messages,
                stop_sequences=stop_sequences or default_stop_sequences
            )
            memory_step.model_output_message = chat_message
            output_text = chat_message.content
            
            # Truncate after first code block for models that don't support stop sequences
            output_text = self._truncate_after_first_code_block(output_text)
            
            # Append the closing tag if it was used as a stop sequence
            # This helps subsequent LLM calls learn to close code blocks properly
            if (
                output_text
                and self.code_block_tags[1] not in self.code_block_tags[0]
                and not output_text.strip().endswith(self.code_block_tags[1].strip())
            ):
                output_text += self.code_block_tags[1]
            
            # Update memory with possibly truncated output
            memory_step.model_output_message.content = output_text
            
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

    def _truncate_after_first_code_block(self, output_text: str) -> str:
        """
        Truncate output after the first complete code block.
        
        This is necessary for models that don't support stop sequences (e.g., gpt-5 series),
        which may generate multiple code blocks in a single response.
        
        Returns the text up to and including the first complete code block.
        """
        if not output_text:
            return output_text
        
        open_tag = self.code_block_tags[0]
        close_tag = self.code_block_tags[1].strip()  # Remove leading newline for matching
        
        # Find the first code block opening
        open_idx = output_text.find(open_tag)
        if open_idx == -1:
            return output_text  # No code block found
        
        # Find the closing tag after the opening
        search_start = open_idx + len(open_tag)
        close_idx = output_text.find(close_tag, search_start)
        if close_idx == -1:
            return output_text  # No closing tag found, return as-is
        
        # Truncate after the closing tag
        truncated = output_text[:close_idx + len(close_tag)]
        return truncated

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

        # check for anti-pattern: data_discovery and final_answer in same code block
        self._check_for_anti_patterns(code_action)

        tool_call = ToolCall(
            name="python_interpreter",
            arguments=code_action,
            id=f"call_{len(self.memory.steps)}",
        )
        memory_step.tool_calls = [tool_call]

        return code_action, tool_call

    def _check_for_anti_patterns(self, code_action: str) -> None:
        """
        Check for anti-patterns in generated code and raise errors with guidance.
        
        Args:
            code_action: The parsed code to check
            
        Raises:
            AgentParsingError: If an anti-pattern is detected
        """
        # Check if code contains both data_discovery and final_answer calls
        has_data_discovery = bool(re.search(r'\bdata_discovery\s*\(', code_action))
        has_final_answer = bool(re.search(r'\bfinal_answer\s*\(', code_action))
        
        if has_data_discovery and has_final_answer:
            error_msg = (
                "Anti-pattern detected: You cannot call both 'data_discovery' and 'final_answer' "
                "in the same code block.\n\n"
                "This is because data discovery should be completed BEFORE building the final plan. "
                "You need to:\n"
                "1. First call data_discovery() to explore the datasets\n"
                "2. Observe the results from data discovery\n"
                "3. Then in a SEPARATE step, build and return your logical plan with final_answer()\n\n"
                "Please split these into separate steps."
            )
            raise AgentParsingError(error_msg, self.logger)

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
        # Note: _generate_model_output may have already added it, but we check before adding
        if add_closing_tag and output_text and not output_text.strip().endswith(self.code_block_tags[1].strip()):
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
        """Handle logical plan generation with code execution and managed agent calls."""
        yield from self._step_with_code_and_final_answer(
            memory_step=memory_step,
            final_answer_tags=None,
            parse_code_only=True,
            add_closing_tag=True
        )

    def _step_paraphrase_stream(self, memory_step: ActionStep) -> Generator[ActionOutput]:
        """Handle paraphrasing a logical plan to natural language."""
        yield from self._step_with_code_and_final_answer(
            memory_step=memory_step,
            final_answer_tags=self.plan_tags,
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
        """Initialise system prompt, memory, and executor for a phase.

        Creates a fresh ``AgentMemory`` instance, populates it with the
        system prompt and optional conversation history, and configures
        the Python executor with datasets and tools.

        Requires:
            - *prompt_template_key* exists in ``self.prompt_templates``.
            - *task_step_class* is a ``MemoryStep`` subclass whose
              ``__init__`` accepts ``**task_kwargs``.

        Returns:
            A new ``AgentMemory`` instance ready for ``_run_stream``.

        Raises:
            KeyError: If *prompt_template_key* is missing.
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
        self.python_executor.send_tools({**self.tools, **self.managed_agents})

        return phase_memory

    def generate_logical_plan(
        self, 
        query: str, 
        datasets: list[Dataset], 
        conversation: Conversation | None = None,
        cost_budget: float | None = None,
    ) -> dict:
        """Generate a logical execution plan as code.

        Uses the managed ``DataDiscoveryAgent`` to explore datasets and then
        produces a code-based logical plan composed of semantic operators.

        Requires:
            - *query* is a non-empty string.
            - *datasets* is a non-empty list of ``Dataset`` objects.

        Returns:
            A dict (or structured object) representing the logical plan.

        Raises:
            AgentGenerationError: If the LLM fails to produce valid output.
            AssertionError: If the run stream does not terminate with a
            ``FinalAnswerStep``.
        """
        # TODO: Use cost_budget to inform query optimization decisions
        # For now, we accept the parameter but do not use it
        _ = cost_budget
        
        # Update datasets if provided (also updates managed agent)
        if datasets:
            self._datasets = datasets
            if hasattr(self, 'data_discovery_agent'):
                self.data_discovery_agent._datasets = datasets
        
        # setup execution with phase-specific memory
        planning_memory = self._setup_phase_execution(
            phase_type="planning",
            prompt_template_key="system_prompt",
            template_vars={
                "logical_operators": {op: op.desc() for op in LOGICAL_OPERATORS},
                "code_opening_tag": self.code_block_tags[0],
                "code_closing_tag": self.code_block_tags[1],
                "plan_opening_tag": self.plan_tags[0],
                "plan_closing_tag": self.plan_tags[1],
                "has_conversation": conversation is not None,
                "managed_agents": self.managed_agents,
            },
            task_step_class=PlannerTaskStep,
            task_kwargs={"task": query, "datasets": datasets},
            conversation=conversation,
            plan_type_for_history="logical-plan"
        )

        # run and return with isolated memory
        steps = list(self._run_stream(phase="planning", memory=planning_memory))
        assert isinstance(steps[-1], FinalAnswerStep)
        return steps[-1].output

    def paraphrase_logical_plan(
        self, 
        query: str, 
        logical_plan: dict, 
        datasets: list[Dataset],
        conversation: Conversation | None = None,
        cost_budget: float | None = None,
    ) -> str:
        """Translate a logical plan into a natural-language description.

        Requires:
            - *logical_plan* is a non-empty dict produced by
              ``generate_logical_plan``.
            - *datasets* lists the datasets referenced in the plan.

        Returns:
            A human-readable string summarising the plan.

        Raises:
            AgentGenerationError: If the LLM fails to produce valid output.
            AssertionError: If the run stream does not terminate with a
            ``FinalAnswerStep``.
        """
        # TODO: Use cost_budget to inform paraphrase content (e.g., estimated costs)
        # For now, we accept the parameter but do not use it
        _ = cost_budget
        
        # setup execution with phase-specific memory
        paraphrase_memory = self._setup_phase_execution(
            phase_type="paraphrase",
            prompt_template_key="paraphrase_prompt",
            template_vars={
                "code_opening_tag": self.code_block_tags[0],
                "code_closing_tag": self.code_block_tags[1],
                "plan_opening_tag": self.plan_tags[0],
                "plan_closing_tag": self.plan_tags[1],
                "logical_plan": str(logical_plan),
                "has_conversation": conversation is not None,
            },
            task_step_class=PlannerTaskStep,
            task_kwargs={"task": query, "datasets": datasets},
            conversation=conversation,
            plan_type_for_history="natural-language-plan",
            log_title="Plan Paraphrase"
        )

        # run and return with isolated memory
        steps = list(self._run_stream(phase="paraphrase", memory=paraphrase_memory))
        assert isinstance(steps[-1], FinalAnswerStep)
        return steps[-1].output

    # ------------------------------------------------------------------
    # Streaming variants — yield PlanningProgress events between steps
    # ------------------------------------------------------------------

    @staticmethod
    def _code_calls_data_discovery(code_action: str | None) -> bool:
        """Return ``True`` if *code_action* contains a ``data_discovery(`` call.

        Requires:
            - *code_action* may be ``None`` (returns ``False``).

        Returns:
            Whether the code text contains the pattern ``data_discovery(``.

        Raises:
            None.
        """
        if not code_action:
            return False
        return bool(re.search(r"\bdata_discovery\s*\(", code_action))

    def generate_logical_plan_stream(
        self,
        query: str,
        datasets: list[Dataset],
        conversation: Conversation | None = None,
        cost_budget: float | None = None,
    ) -> Generator[PlanningProgress | dict, None, None]:
        """Generate a logical plan, yielding progress events along the way.

        This is the streaming counterpart of :meth:`generate_logical_plan`.
        It performs exactly the same work but yields
        :class:`PlanningProgress` objects between steps so that callers
        can forward them to users.

        The **last value** yielded is always the logical-plan ``dict``
        (i.e. the same value that ``generate_logical_plan`` returns).

        Requires:
            - *query* is a non-empty string.
            - *datasets* is a non-empty list of ``Dataset`` objects.

        Returns:
            A generator that yields zero or more ``PlanningProgress``
            objects followed by exactly one ``dict`` (the logical plan).

        Raises:
            AgentGenerationError: If the LLM fails to produce valid output.
            AssertionError: If the underlying stream does not end with a
            ``FinalAnswerStep``.
        """
        _ = cost_budget  # reserved for future cost-aware planning

        # Update datasets if provided (also updates managed agent)
        if datasets:
            self._datasets = datasets
            if hasattr(self, "data_discovery_agent"):
                self.data_discovery_agent._datasets = datasets

        # Setup phase memory (identical to non-streaming variant)
        planning_memory = self._setup_phase_execution(
            phase_type="planning",
            prompt_template_key="system_prompt",
            template_vars={
                "logical_operators": {op: op.desc() for op in LOGICAL_OPERATORS},
                "code_opening_tag": self.code_block_tags[0],
                "code_closing_tag": self.code_block_tags[1],
                "plan_opening_tag": self.plan_tags[0],
                "plan_closing_tag": self.plan_tags[1],
                "has_conversation": conversation is not None,
                "managed_agents": self.managed_agents,
            },
            task_step_class=PlannerTaskStep,
            task_kwargs={"task": query, "datasets": datasets},
            conversation=conversation,
            plan_type_for_history="logical-plan",
        )

        yield PlanningProgress(
            phase="logical_plan",
            step=1,
            total_steps=self.max_steps,
            message="Starting to analyze query and plan execution…",
        )

        logical_plan = None
        for step in self._run_stream(phase="planning", memory=planning_memory):
            if isinstance(step, ActionStep):
                # Detect whether this step invoked the DataDiscoveryAgent
                if self._code_calls_data_discovery(step.code_action):
                    yield PlanningProgress(
                        phase="data_discovery",
                        step=step.step_number,
                        total_steps=self.max_steps,
                        message="Exploring datasets to understand their structure…",
                    )
                else:
                    yield PlanningProgress(
                        phase="logical_plan",
                        step=step.step_number,
                        total_steps=self.max_steps,
                        message=f"Building execution plan (step {step.step_number})…",
                    )

            elif isinstance(step, FinalAnswerStep):
                logical_plan = step.output

        assert logical_plan is not None, "Planner stream did not produce a FinalAnswerStep"
        # Yield the plan itself as the terminal value
        yield logical_plan

    def paraphrase_logical_plan_stream(
        self,
        query: str,
        logical_plan: dict,
        datasets: list[Dataset],
        conversation: Conversation | None = None,
        cost_budget: float | None = None,
    ) -> Generator[PlanningProgress | str, None, None]:
        """Translate a logical plan to natural language, yielding progress events.

        This is the streaming counterpart of
        :meth:`paraphrase_logical_plan`.  It yields
        :class:`PlanningProgress` events followed by exactly one ``str``
        (the natural-language plan).

        Requires:
            - *logical_plan* is a non-empty dict.
            - *datasets* lists the datasets referenced in the plan.

        Returns:
            A generator that yields zero or more ``PlanningProgress``
            objects followed by exactly one ``str``.

        Raises:
            AgentGenerationError: If the LLM fails to produce valid output.
            AssertionError: If the underlying stream does not end with a
            ``FinalAnswerStep``.
        """
        _ = cost_budget  # reserved for future cost-aware content

        paraphrase_memory = self._setup_phase_execution(
            phase_type="paraphrase",
            prompt_template_key="paraphrase_prompt",
            template_vars={
                "code_opening_tag": self.code_block_tags[0],
                "code_closing_tag": self.code_block_tags[1],
                "plan_opening_tag": self.plan_tags[0],
                "plan_closing_tag": self.plan_tags[1],
                "logical_plan": str(logical_plan),
                "has_conversation": conversation is not None,
            },
            task_step_class=PlannerTaskStep,
            task_kwargs={"task": query, "datasets": datasets},
            conversation=conversation,
            plan_type_for_history="natural-language-plan",
            log_title="Plan Paraphrase",
        )

        yield PlanningProgress(
            phase="paraphrase",
            step=1,
            total_steps=self.max_steps,
            message="Summarizing the execution plan in natural language…",
        )

        nl_plan = None
        for step in self._run_stream(phase="paraphrase", memory=paraphrase_memory):
            if isinstance(step, ActionStep):
                yield PlanningProgress(
                    phase="paraphrase",
                    step=step.step_number,
                    total_steps=self.max_steps,
                    message=f"Generating plan summary (step {step.step_number})…",
                )
            elif isinstance(step, FinalAnswerStep):
                nl_plan = step.output

        assert nl_plan is not None, "Paraphrase stream did not produce a FinalAnswerStep"
        # Yield the NL plan as the terminal value
        yield nl_plan
