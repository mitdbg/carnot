"""Data Discovery Agent for Carnot.

This agent specializes in analyzing datasets to discover relevant data,
inspect schemas, and provide information about dataset structure.
It can be used as a managed agent by the Planner.

This agent inherits from ``CarnotBaseAgent``, a minimal base class that
contains only the machinery needed by the Planner and DataDiscoveryAgent.
"""

import queue
from collections.abc import Generator
from importlib import resources
from typing import Any

import yaml

from carnot.agents.base import ActionOutput, BaseAgent, populate_template
from carnot.agents.local_python_executor import (
    BASE_BUILTIN_MODULES,
    LocalPythonExecutor,
    PythonExecutor,
)
from carnot.agents.memory import ActionStep, FinalAnswerStep, PlanningStep, ToolCall
from carnot.agents.models import Model
from carnot.agents.monitoring import LogLevel
from carnot.agents.tools import Tool
from carnot.agents.utils import AgentParsingError
from carnot.data.dataset import Dataset
from carnot.index import INDEX_TYPES

MAX_PRINT_DATASETS = 20
MAX_STEPS_WARNING_THRESHOLD = 3


class DataDiscoveryAgent(BaseAgent):
    """
    An agent specialized in analyzing datasets and discovering relevant data.
    
    This agent inherits from BaseAgent and implements its own code execution logic
    for data exploration. It can be used standalone or as a managed agent by the Planner.
    
    Capabilities:
    - Identifying relevant datasets for a given query
    - Returning schema (field names and types) for dataset items
    - Classifying fields as structured vs unstructured
    - Checking if datasets have semantic indices
    - Sampling items from datasets
    
    Args:
        datasets: List of Dataset objects to make available for analysis.
        model: The LLM model to use for the agent.
        tools: Optional list of additional tools for the agent.
        max_steps: Maximum number of steps the agent can take.
        max_print_outputs_length: Maximum length of print outputs in code execution.
        **kwargs: Additional keyword arguments passed to BaseAgent.
    """

    def __init__(
        self,
        datasets: list[Dataset],
        model: Model,
        tools: list[Tool] | None = None,
        max_steps: int = 5,
        max_print_outputs_length: int | None = None,
        **kwargs
    ):
        # Store datasets for use in the Python executor
        self._datasets = datasets

        # Code execution configuration (previously from CodeAgent)
        self.additional_authorized_imports = [
            "carnot",
            "json",       # For JSON parsing/formatting
            "html",       # For HTML entity decoding
            "textwrap",   # For text formatting
            "copy",       # For deep copying data structures
            "functools",  # For functional programming utilities
            "operator",   # For operator functions
            "string",     # For string constants
            "csv",        # For CSV parsing (in-memory)
            "re",         # For regular expressions
        ]
        self.authorized_imports = sorted(
            set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports)
        )
        # Use code block tags where closing tag is NOT contained in opening tag
        # This allows us to use the closing tag as a stop sequence
        self.code_block_tags = ("```python", "\n```")
        self.max_print_outputs_length = max_print_outputs_length
        self._use_structured_outputs_internally = False

        # Load prompt templates
        prompt_templates = yaml.safe_load(
            resources.files("carnot.agents.prompts").joinpath("data_discovery_agent.yaml").read_text()
        )

        # Initialize BaseAgent
        super().__init__(
            tools=tools or [],
            model=model,
            prompt_templates=prompt_templates,
            name="data_discovery",
            description=self._build_description(),
            max_steps=max_steps,
            **kwargs
        )

        # optional progress queue for streaming updates to external consumers.
        # Set per-call by the Planner's _progress_scope context manager;
        # None when not being observed.
        self.progress_queue: queue.Queue | None = None

        # Create the Python executor for code execution
        self.python_executor = self.create_python_executor()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit - cleanup resources."""
        self.cleanup()

    def cleanup(self):
        """Clean up resources used by the agent, such as the Python executor."""
        if hasattr(self, "python_executor") and hasattr(self.python_executor, "cleanup"):
            self.python_executor.cleanup()

    def _get_step_cost_usd(self, step: ActionStep) -> float:
        """Helper to extract USD cost from an ActionStep's model output message."""
        # NOTE: I think we can safely assume model_output_message is always present
        msg = getattr(step, "model_output_message", None)
        if msg is not None and getattr(msg, "llm_call_stats", None) is not None:
            return msg.llm_call_stats.cost_usd
        return 0.0

    def _build_description(self) -> str:
        """
        Build the description that tells the Planner how to use this agent.
        
        This description should clearly explain the agent's capabilities
        and provide examples of how to invoke it.
        """
        dataset_names = [ds.name for ds in self._datasets]
        datasets_str = ", ".join(f'"{name}"' for name in dataset_names[:MAX_PRINT_DATASETS])
        if len(dataset_names) > MAX_PRINT_DATASETS:
            datasets_str += f", ... ({len(dataset_names)} total)"

        return f"""Data Discovery Agent: Analyzes datasets to find relevant data for queries.

**Capabilities:**
1. **Identify Relevant Datasets**: Given a query, intelligently determines which datasets may contain relevant information by inspecting their contents, not just their names.
   Example: "Which datasets contain research papers or academic publications?"

2. **Get Dataset Schema**: Returns the schema (field names and data types) for all *programmatically accessible* fields in a dataset's items. Also provides a qualitative assessment of which fields contain structured data (numbers, dates, categories) vs unstructured data (document contents, text, images).
   Example: "What is the schema of the Legal Contracts dataset? Classify each field as structured or unstructured."

3. **Check Index Availability**: Determines whether a dataset has a semantic index constructed, which enables efficient similarity search.
   Example: "Does the Movie Reviews dataset have an index?"

4. **Sample Items**: Retrieves sample items from a dataset to understand its contents.
   Example: "Show me 5 sample items from the FTC Data dataset."

**Available Datasets:** {datasets_str}

**Usage Notes:**
- When asking about relevance, be specific about what you're looking for (e.g., "datasets with financial data" vs "datasets with text documents")
- Schema queries return ONLY the keys from items in a Dataset, which are the fields that can be accessed programmatically
- If document contents contain additional information (titles, authors, etc.), this will be noted but not included in the schema
"""

    def create_python_executor(self) -> PythonExecutor:
        """Create a Python executor with datasets available in the local scope."""
        executor = LocalPythonExecutor(
            self.additional_authorized_imports,
            max_print_outputs_length=self.max_print_outputs_length,
        )
        return executor
    
    def _get_executor_state(self) -> dict[str, Any]:
        """
        Get the state dictionary to inject into the Python executor.
        
        Includes:
        - datasets: Dict mapping dataset names to Dataset objects
        - typename: Helper function to get type name without dunder access
        """
        def typename(obj: Any) -> str:
            """
            Get the type name of an object safely without accessing __name__.
            
            Use this instead of type(x).__name__ which is blocked by the executor.
            
            Examples:
                typename(42) -> 'int'
                typename("hello") -> 'str'
                typename([1,2,3]) -> 'list'
            """
            t = type(obj)
            # Extract type name from str(type(x)) which looks like "<class 'typename'>"
            type_str = str(t)
            # Parse out the name between quotes
            if "'" in type_str:
                return type_str.split("'")[1].split(".")[-1]
            return type_str
        
        return {
            "datasets": {ds.name: ds for ds in self._datasets},
            "typename": typename,
        }
    
    def initialize_system_prompt(self) -> str:
        """Initialize the system prompt with template variables."""
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={
                "tools": self.tools,
                "managed_agents": self.managed_agents,
                "authorized_imports": str(self.authorized_imports),
                "custom_instructions": self.instructions,
                "code_opening_tag": self.code_block_tags[0],
                "code_closing_tag": self.code_block_tags[1],
                "index_types": {cls.__name__: cls.description for cls in INDEX_TYPES},
            },
        )
        return system_prompt
    
    def run(
        self,
        task: str,
        reset: bool = True,
        additional_args: dict | None = None,
        max_steps: int | None = None,
    ):
        """Run the data discovery agent on a task.

        Injects datasets into the executor state before delegating to the
        parent ``CarnotBaseAgent.run``.

        Requires:
            - *task* is a non-empty string.
            - ``self._datasets`` is populated.

        Returns:
            The result of the data discovery task (a string report).

        Raises:
            Whatever the parent ``run`` raises.
        """
        # Inject datasets into the executor state
        self.python_executor.state.update(self._get_executor_state())

        # Merge any additional args
        if additional_args:
            self.python_executor.state.update(additional_args)

        # Run the parent class implementation
        return super().run(
            task=task,
            reset=reset,
            additional_args=additional_args,
            max_steps=max_steps,
        )
    
    def __call__(self, task: str, **kwargs):
        """
        Handle being called as a managed agent.
        
        This method is called when another agent (like the Planner) invokes
        this agent as a managed agent.
        """
        # Inject datasets before processing
        self.python_executor.state.update(self._get_executor_state())
        
        # Merge additional_args if provided
        additional_args = kwargs.get("additional_args", {})
        if additional_args:
            self.python_executor.state.update(additional_args)
        
        # Call parent implementation which handles managed agent prompting
        return super().__call__(task=task, **kwargs)

    def _step_stream(
        self, memory_step: ActionStep
    ) -> Generator[ToolCall | ActionOutput, None, None]:
        """
        Execute one step of the agent.
        
        Follows the Planner's simpler non-streaming pattern:
        1. Generate model output
        2. Parse code
        3. Pre-flight check for final_answer on last step
        4. Execute code
        
        When on the final step, if the code doesn't contain final_answer(),
        we raise an error forcing the agent to reconsider and actually
        return its findings.
        """
        # Generate model output (stores in memory_step.model_output)
        self._generate_model_output(memory_step)
        
        # Parse the code
        code_action, tool_call = self._parse_and_prepare_code(memory_step)
        
        # Calculate steps remaining
        steps_remaining = self.max_steps - self.step_number
        
        # Pre-flight check: on last step, code MUST contain final_answer
        if steps_remaining == 0 and "final_answer" not in code_action:
            # Force the agent to provide a final answer
            error_msg = (
                "⚠️ CRITICAL: This is your FINAL step and you have not called final_answer()!\n"
                "You MUST call final_answer(report) with your findings now.\n"
                "Compile what you've learned into a report string and return it immediately."
            )
            raise AgentParsingError(error_msg, self.logger)
        
        # Warn if approaching limit and no final_answer
        if 1 <= steps_remaining <= MAX_STEPS_WARNING_THRESHOLD - 1 and "final_answer" not in code_action:
            self.logger.log(
                f"[DataDiscoveryAgent] Warning: {steps_remaining} steps remaining but code doesn't contain final_answer()",
                level=LogLevel.INFO,
            )
        
        # Yield tool call
        yield tool_call
        
        # Execute and yield result
        yield from self._execute_code_action(memory_step, code_action)

    def _generate_model_output(
        self,
        memory_step: ActionStep,
        stop_sequences: list[str] | None = None
    ) -> str:
        """
        Generate model output and store in memory step.
        
        Follows the Planner's simpler non-streaming pattern.
        Uses stop sequences to prevent the model from continuing after a code block.
        For models that don't support stop sequences (e.g., gpt-5 series), we truncate
        the output after the first complete code block.
        """
        from carnot.agents.utils import AgentGenerationError
        
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

    def _parse_and_prepare_code(
        self,
        memory_step: ActionStep,
        error_context: str = "code blobs"
    ) -> tuple[str, ToolCall]:
        """
        Parse code from model output and create tool call.
        
        Follows the Planner's pattern for code parsing.
        
        Args:
            memory_step: Current action step
            error_context: Context string for error messages
            
        Returns:
            (code_action, tool_call): Parsed code and corresponding tool call
        """
        from carnot.agents.local_python_executor import fix_final_answer_code
        from carnot.agents.utils import parse_code_blobs
        
        output_text = memory_step.model_output
        
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

    def _execute_code_action(
        self, memory_step: ActionStep, code_action: str
    ) -> Generator[ActionOutput, None, None]:
        """Execute code and yield result."""
        from rich.console import Group
        from rich.text import Text

        from carnot.agents.utils import AgentExecutionError, truncate_content
        
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
                Text(f"Out: {truncated_output}"),
            ]
        self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
        memory_step.code_action_output = code_output.output
        yield ActionOutput(output=code_output.output, is_final_answer=code_output.is_final_answer)

    def _run_stream(
        self, task: str, max_steps: int, images: list | None = None
    ) -> Generator[ActionStep | PlanningStep | FinalAnswerStep, None, None]:
        """Override ``_run_stream`` to add countdown warnings and push
        progress events to the progress queue.

        When the agent has ``MAX_STEPS_WARNING_THRESHOLD`` steps remaining,
        injects a message into the conversation prompting the agent to
        return its final answer soon.

        If ``self.progress_queue`` is set (by the Planner's
        ``_progress_scope``), each completed ``ActionStep`` is converted
        to a ``PlanningProgress`` event and pushed to the queue.
        """
        from carnot.execution.progress import PlanningProgress

        for step in super()._run_stream(task, max_steps, images=images):
            yield step

            # After yielding an ActionStep, check if we should warn the agent
            if isinstance(step, ActionStep):
                # step_number is the step that just completed (before parent increments)
                # steps_remaining = max_steps - current_step
                steps_remaining = max_steps - self.step_number

                # Debug logging
                self.logger.log(
                    f"[DataDiscoveryAgent] Step {self.step_number} completed, {steps_remaining} steps remaining (max={max_steps})",
                    level=LogLevel.INFO,
                )

                # Countdown warning for the last few steps
                if 1 <= steps_remaining <= MAX_STEPS_WARNING_THRESHOLD:
                    step_word = "step" if steps_remaining == 1 else "steps"
                    warning_msg = (
                        f"\n\n⚠️ Warning: You have {steps_remaining} {step_word} remaining. "
                        f"Return your final answer with final_answer() soon."
                    )
                    # Modify the action step's observations
                    step.observations = (step.observations or "") + warning_msg
                    # Log that we're adding the warning
                    self.logger.log(
                        "[DataDiscoveryAgent] Added max_steps warning to observations",
                        level=LogLevel.INFO,
                    )

                # push progress to external consumer
                if self.progress_queue is not None:
                    event = PlanningProgress(
                        phase="data_discovery",
                        step=step.step_number,
                        message=f"Data discovery step {step.step_number}...",
                        code_action=step.code_action,
                        observations=step.observations,
                        step_cost_usd=self._get_step_cost_usd(step),
                        error=str(step.error) if step.error else None,
                    )
                    self.progress_queue.put(event.to_dict())
