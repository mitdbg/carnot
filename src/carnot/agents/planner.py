import queue
import re
import time
from collections.abc import Generator
from contextlib import contextmanager
from importlib import resources

import yaml
from rich.console import Group
from rich.text import Text

from carnot.agents.base import BaseAgent, populate_template
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
    ParaphraseTaskStep,
    PlannerTaskStep,
    ToolCall,
)
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
    2. ``paraphrase_plan()`` — translates the code-based plan
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
        max_steps: int = 5,
        **kwargs
    ):
        # Store datasets for use in planning
        self._datasets = datasets

        # Load prompt templates
        prompt_templates = yaml.safe_load(
            resources.files("carnot.agents.prompts").joinpath("planner_agent.yaml").read_text()
        )

        self.plan_tags = ["<plan>", "</plan>"]
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

        # phase-specific memories to prevent context bleeding between phases
        self.planning_memory = None
        self.paraphrase_memory = None

        # python executor for code execution during planning
        self.python_executor = None

        # maximum steps for planning and paraphrasing phases
        self.max_steps = max_steps

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()

    def cleanup(self):
        """Clean up resources used by the agent, such as the remote Python executor."""
        if self.python_executor and hasattr(self.python_executor, "cleanup"):
            self.python_executor.cleanup()

    def create_python_executor(self) -> PythonExecutor:
        return LocalPythonExecutor(self.additional_authorized_imports)

    @contextmanager
    def _progress_scope(self, progress_queue: queue.Queue | None):
        """Bind a progress queue to all managed agents for the duration of a call.

        Sets ``progress_queue`` on every managed agent that supports it,
        then unconditionally clears the reference in a ``finally`` block.
        This ensures the queue is scoped to a single call and no dangling
        references survive after the call completes.

        Requires:
            - *progress_queue* is ``None`` or a thread-safe ``queue.Queue``.

        Returns:
            A context manager that yields *progress_queue*.

        Raises:
            None.
        """
        if progress_queue is not None:
            for agent in self.managed_agents.values():
                if hasattr(agent, 'progress_queue'):
                    agent.progress_queue = progress_queue
        try:
            yield progress_queue
        finally:
            for agent in self.managed_agents.values():
                if hasattr(agent, 'progress_queue'):
                    agent.progress_queue = None

    def _update_progress(self, progress_queue: queue.Queue | None, **kwargs):
        """Helper to put a PlanningProgress update on the queue if it exists."""
        if progress_queue is not None:
            progress_event = PlanningProgress(**kwargs)
            progress_queue.put(progress_event.to_dict())

    def _get_step_cost_usd(self, step: ActionStep) -> float:
        """Helper to extract USD cost from an ActionStep's model output message."""
        # NOTE: I think we can safely assume model_output_message is always present
        msg = getattr(step, "model_output_message", None)
        if msg is not None and getattr(msg, "llm_call_stats", None) is not None:
            return msg.llm_call_stats.cost_usd
        return 0.0

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
        phase: str,
        memory: AgentMemory,
    ) -> Generator[ActionStep | FinalAnswerStep, None, None]:
        """Execute a planning phase, yielding intermediate steps. Each ActionStep
        represents a single complete trace of one call to the LLM, including the
        input messages, output, and any tool calls made (including code execution).
        The observation from executing the tool calls (e.g. code output) is included
        in the ActionStep, and the next step's input messages will include the updated
        memory with those observations.

        Requires:
            - *phase* is ``"planning"`` or ``"paraphrase"``.
            - *memory* is a properly initialized ``AgentMemory`` instance with
              a system prompt and task step for the current phase.

        Returns:
            A generator of step objects.  The last yielded value
            is always a ``FinalAnswerStep``.

        Raises:
            AgentGenerationError: If the LLM fails to produce valid output.
            ValueError: If *phase* is not recognised.
        """
        # Point the inherited self.memory at the phase-specific memory so
        # that write_memory_to_messages() (called by _generate_model_output)
        # reads from the correct context instead of the stale base-class default.
        self.memory = memory

        self.step_number = 1
        returned_final_answer = False
        while not returned_final_answer and self.step_number <= self.max_steps:
            try:
                # execute one action step for the given phase
                action_step = None
                if phase == "planning":
                    action_step = self._step_generating_logical_plan_stream(step_num=self.step_number)
                elif phase == "paraphrase":
                    action_step = self._step_paraphrase_stream(step_num=self.step_number)
                else:
                    raise ValueError(f"Unknown phase: {phase}")

                self.logger.log_rule(f"Step {self.step_number}", level=LogLevel.INFO)
                if action_step.is_final_answer:
                    returned_final_answer = True
                    final_answer = action_step.final_output
                    self.logger.log(
                        Text(f"Final answer: {final_answer}", style=f"bold {YELLOW_HEX}"),
                        level=LogLevel.INFO,
                    )

            # Agent generation errors are not caused by a Model error but an implementation error: so we should raise them and exit.
            except AgentGenerationError as e:
                raise e

            # Other AgentError types are caused by the Model, so we should log them and iterate.
            except AgentError as e:
                if action_step is None:
                    action_step = ActionStep(
                        step_number=self.step_number,
                        timing=Timing(start_time=time.time()),
                    )
                action_step.error = e

            finally:
                if action_step is None:
                    action_step = ActionStep(
                        step_number=self.step_number,
                        timing=Timing(start_time=time.time()),
                    )
                # set the end time for the step and add to memory
                action_step.timing.end_time = time.time()
                memory.steps.append(action_step)
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

        # Stop sequences prevent the model from simulating the full
        # conversation loop (generating fake Observations or tool calls).
        # NOTE: we intentionally do NOT include the code-block closing tag
        # ("\n```") here because it is a prefix of the opening sequence
        # "\n```python" — the stop would fire before the model can emit
        # "python", killing every code block.  Instead we rely on
        # _truncate_after_first_code_block() to trim extra output.
        default_stop_sequences = ["Observation:", "Calling tools:"]

        try:
            chat_message = self.model.generate(
                memory_messages,
                stop_sequences=stop_sequences or default_stop_sequences
            )
            memory_step.model_output_message = chat_message
            output_text = chat_message.content

            # Truncate after first code block for models that don't support stop sequences
            output_text = self._truncate_after_first_code_block(output_text)

            # If the model produced an incomplete code block (opened but
            # never closed), append the closing tag so downstream parsing
            # can still extract the code.
            if (
                output_text
                and self.code_block_tags[0] in output_text
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
            memory_step.code_action_output = final_answer
            return True, final_answer
        except Exception:
            return False, None

    def _parse_and_prepare_code(
        self,
        output_text: str,
        memory_step: ActionStep,
        num_memory_steps: int,
        error_context: str = "code blobs"
    ) -> str:
        """
        Parse code from output text and return it. This function also adds the corresponding tool call to the memory
        step.

        Args:
            output_text: Raw LLM output
            memory_step: Current action step
            error_context: Context string for error messages

        Returns:
            code_action: Parsed code
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
            id=f"call_{num_memory_steps}",
        )
        memory_step.tool_calls = [tool_call]

        return code_action

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
    ) -> ActionStep:
        """
        Finalize code execution step with observations and output.
        
        Args:
            code_output: Output from python executor
            observation: Accumulated observation string
            execution_outputs_console: Console outputs for logging
            memory_step: Current action step
            is_final_answer: Override for is_final_answer. If None, use code_output.is_final_answer
        
        Returns:
            ActionStep for yielding
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
        memory_step.code_action_output = code_output.output
        memory_step.final_output = code_output.output
        memory_step.is_final_answer = use_final_answer

        return memory_step

    def _step_with_code_and_final_answer(
        self,
        step_num: int,
        final_answer_tags: list[str] | None = None,
        parse_code_only: bool = False,
        add_closing_tag: bool = False
    ) -> ActionStep:
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
        memory_step = ActionStep(step_number=self.step_number, timing=Timing(start_time=time.time()))

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
                memory_step.final_output = final_answer
                memory_step.is_final_answer = True
                return memory_step

        # Parse as code
        error_context = "correct code blobs" if parse_code_only else "either code blobs or a final answer"
        code_action = self._parse_and_prepare_code(output_text, memory_step, step_num, error_context)

        # Execute code
        code_output, observation, execution_outputs_console = self._execute_code_and_collect_output(
            code_action,
            memory_step
        )

        # Finalize and return result
        # Use explicit False for planning/discovery, use code_output.is_final_answer for compilation
        is_final_override = None if parse_code_only else False
        memory_step = self._finalize_code_execution_step(
            code_output,
            observation,
            execution_outputs_console,
            memory_step,
            is_final_answer=is_final_override,
        )
        return memory_step

    def _step_generating_logical_plan_stream(self, step_num: int) -> ActionStep:
        """Handle logical plan generation with code execution and managed agent calls."""
        return self._step_with_code_and_final_answer(
            step_num=step_num,
            final_answer_tags=None,
            parse_code_only=True,
            add_closing_tag=True
        )

    def _step_paraphrase_stream(self, step_num: int) -> ActionStep:
        """Handle paraphrasing a logical plan to natural language."""
        return self._step_with_code_and_final_answer(
            step_num=step_num,
            final_answer_tags=self.plan_tags,
            parse_code_only=False,
            add_closing_tag=False
        )

    def _setup_phase_execution(
        self,
        phase: str,
        query: str,
        prompt_template_key: str,
        template_vars: dict,
        conversation: Conversation | None = None,
        plan: dict | None = None,
    ) -> tuple[AgentMemory, PythonExecutor]:
        """Initialise system prompt, memory, and executor for a phase.

        Creates a fresh ``AgentMemory`` instance, populates it with the
        system prompt and optional conversation history, and configures
        the Python executor with datasets and tools.

        Requires:
            - *phase* is either ``"planning"`` or ``"paraphrase"``.
            - *prompt_template_key* exists in ``self.prompt_templates``.
            - When *phase* is ``"paraphrase"``, *plan* must be a non-empty dict.

        Returns:
            A new ``AgentMemory`` instance ready for ``_run_stream`` and a configured ``PythonExecutor``.

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

        # add previous plan from conversation if applicable
        if conversation:
            plan_type_for_history = "logical-plan" if phase == "planning" else "natural-language-plan"
            latest_plan = conversation.get_latest_agent_plan(plan_type_for_history)
            if latest_plan:
                phase_memory.steps.append(latest_plan)

        # log task
        self.logger.log_task(
            content=query.strip(),
            subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
            level=LogLevel.INFO,
            title="Logical Plan Generation" if phase == "planning" else "Logical Plan Paraphrasing",
        )

        # add the appropriate task step to phase memory
        if phase == "paraphrase":
            task_step = ParaphraseTaskStep(task=query, plan=plan)
        else:
            task_step = PlannerTaskStep(task=query, datasets=self._datasets)
        phase_memory.steps.append(task_step)

        # setup Python executor state
        datasets_dict = {dataset.name: dataset for dataset in self._datasets}
        conversation_list = conversation.to_dict_list() if conversation else []
        state = {
            "datasets": datasets_dict,
            "conversation": conversation_list,
            **self.state
        }
        python_executor = self.create_python_executor()
        python_executor.send_variables(variables=state)
        python_executor.send_tools({**self.tools, **self.managed_agents})

        return phase_memory, python_executor

    # ------------------------------------------------------------------
    # Public interface — progress goes through the queue, results are
    # returned directly.
    # ------------------------------------------------------------------

    def generate_logical_plan(
        self,
        query: str,
        conversation: Conversation | None = None,
        progress_queue: queue.Queue | None = None,
    ) -> Dataset:
        """Generate a logical execution plan as code.

        Uses the managed ``DataDiscoveryAgent`` to explore datasets and then
        produces a code-based logical plan composed of semantic operators.

        When *progress_queue* is provided, ``PlanningProgress`` events are
        pushed to it in real time — both from the Planner's own steps and
        from managed agents (e.g. data discovery).  The queue is scoped to
        this call via :meth:`_progress_scope`.

        Requires:
            - *query* is a non-empty string.
            - *self._datasets* is a non-empty list of ``Dataset`` objects.
            - *progress_queue*, if provided, is a thread-safe ``queue.Queue``.

        Returns:
            A Dataset object which captures the logical plan of the query execution.

        Raises:
            AgentGenerationError: If the LLM fails to produce valid output.
            AssertionError: If no ``FinalAnswerStep`` is found in the
            underlying stream.
        """
        # TODO: execute basic data discovery and template into the prompt here

        # setup planning memory and python executor
        self.planning_memory, self.python_executor = self._setup_phase_execution(
            phase="planning",
            query=query,
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
            conversation=conversation,
        )

        # push initial progress update before starting the planning phase
        self._update_progress(
            progress_queue,
            phase="logical_plan",
            message="Starting to analyze query and plan execution...",
        )

        with self._progress_scope(progress_queue):
            logical_plan = None
            for step in self._run_stream(phase="planning", memory=self.planning_memory):
                if isinstance(step, FinalAnswerStep):
                    logical_plan = step.output
                elif isinstance(step, ActionStep):
                    self._update_progress(
                        progress_queue,
                        phase="logical_plan",
                        step=step.step_number,
                        total_steps=self.max_steps,
                        message=f"Building execution plan (step {step.step_number})...",
                        code_action=step.code_action,
                        observations=step.observations,
                        step_cost_usd=self._get_step_cost_usd(step),
                        error=str(step.error) if step.error else None,
                    )

            assert logical_plan is not None, "Planner stream did not produce a FinalAnswerStep"

        return logical_plan

    def paraphrase_plan(
        self,
        query: str,
        plan: dict,
        conversation: Conversation | None = None,
        progress_queue: queue.Queue | None = None,
    ) -> str:
        """Translate a logical or physical plan into a natural-language description.

        When *progress_queue* is provided, ``PlanningProgress`` events are
        pushed to it in real time.  The queue is scoped to this call via
        :meth:`_progress_scope`.

        Requires:
            - *plan* is a non-empty dict with a serialized logical or physical plan
            (produced by ``generate_logical_plan`` or Carnot's ``Optimizer``).
            - *self._datasets* lists the datasets referenced in the plan.
            - *progress_queue*, if provided, is a thread-safe ``queue.Queue``.

        Returns:
            A human-readable string summarising the plan.
6.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
        Raises:
            AgentGenerationError: If the LLM fails to produce valid output.
            AssertionError: If no ``FinalAnswerStep`` is found in the
            underlying stream.
        """
        # setup paraphrase memory and python executor
        self.paraphrase_memory, self.python_executor = self._setup_phase_execution(
            phase="paraphrase",
            query=query,
            prompt_template_key="paraphrase_prompt",
            template_vars={
                "code_opening_tag": self.code_block_tags[0],
                "code_closing_tag": self.code_block_tags[1],
                "plan_opening_tag": self.plan_tags[0],
                "plan_closing_tag": self.plan_tags[1],
                "has_conversation": conversation is not None,
            },
            conversation=conversation,
            plan=plan,
        )

        # push initial progress update before starting the paraphrasing phase
        self._update_progress(
            progress_queue,
            phase="paraphrase",
            message="Summarizing the execution plan in natural language...",
        )

        with self._progress_scope(progress_queue):
            nl_plan = None
            for step in self._run_stream(phase="paraphrase", memory=self.paraphrase_memory):
                if isinstance(step, FinalAnswerStep):
                    nl_plan = step.output
                elif isinstance(step, ActionStep):
                    self._update_progress(
                        progress_queue,
                        phase="paraphrase",
                        step=step.step_number,
                        total_steps=self.max_steps,
                        message=f"Generating plan summary (step {step.step_number})...",
                        code_action=step.code_action,
                        observations=step.observations,
                        step_cost_usd=self._get_step_cost_usd(step),
                        error=str(step.error) if step.error else None,
                    )

            assert nl_plan is not None, "Paraphrase stream did not produce a FinalAnswerStep"

        return nl_plan
