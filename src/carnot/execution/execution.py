import os

from smolagents.tools import Tool

from carnot.agents.models import LiteLLMModel
from carnot.agents.planner import Planner
from carnot.conversation.conversation import Conversation
from carnot.data.dataset import Dataset
from carnot.index.index import CarnotIndex
from carnot.memory.memory import Memory

# from carnot.plan.logical import LogicalPlan
# from carnot.plan.physical import PhysicalPlan


class Execution:
    """
    Class for managing the execution of a query in Carnot.
    """
    def __init__(
            self,
            query: str,
            datasets: list[Dataset],
            tools: list[Tool] | None = None,
            conversation: Conversation | None = None,
            memory: Memory | None = None,
            indices: list[CarnotIndex] | None = None,
            llm_config: dict | None = None,
        ):
        self.query = query
        self.datasets = datasets
        self.tools = tools or []
        self.conversation = conversation
        self.memory = memory or Memory()
        self.indices = indices or []
        self.llm_config = llm_config or {}
        self.planner = Planner(tools=self.tools, model=LiteLLMModel(model_id="openai/gpt-4o-mini", api_key=llm_config.get("OPENAI_API_KEY")))

    def plan(self) -> tuple[str, dict]:
        """
        Generate a logical execution plan for the query.
        """
        # retrieve relevant context from memory
        memories = [] # self.memory.retrieve(self.query)

        # synthesize relevant context from conversation
        conversation_context = ""
        # if self.conversation:
        #     conversation_context = self.conversation.condense(self.query)

        # invoke the planner to create a logical plan in natural language
        nl_plan = self.planner.generate_logical_plan(self.query, self.datasets, self.indices, self.tools, memories, conversation_context)

        # convert the natural language plan to a LogicalPlan object
        plan = self.planner.compile_logical_plan(self.query, self.datasets, nl_plan)

        return nl_plan, plan

    # plan: LogicalPlan -> PhysicalPlan
    def compile(self, plan: dict):
        """
        Compile a logical plan into a physical execution plan.
        """
        return self.planner.compile_physical_plan(plan)

    # physical_plan: PhysicalPlan -> str
    def run(self, physical_plan) -> str:
        """
        Execute the physical plan and return the result.
        """
        # TODO: iterate in topological sort order over operators
        # TODO: handle data / code executor passing between operators
        # TODO: capture output from final operator and return as result
        for operator in physical_plan.operators:
            operator.execute()
