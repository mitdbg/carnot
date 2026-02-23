import logging

from smolagents.tools import Tool

from carnot.agents.models import LiteLLMModel
from carnot.agents.planner import Planner
from carnot.conversation.conversation import Conversation
from carnot.data.dataset import DataItem, Dataset
from carnot.index.index import CarnotIndex
from carnot.memory.memory import Memory

# from carnot.plan.logical import LogicalPlan
# from carnot.plan.physical import PhysicalPlan
from carnot.operators.code import CodeOperator
from carnot.operators.limit import LimitOperator
from carnot.operators.reasoning import ReasoningOperator
from carnot.operators.sem_agg import SemAggOperator
from carnot.operators.sem_filter import SemFilterOperator
from carnot.operators.sem_flat_map import SemFlatMapOperator
from carnot.operators.sem_groupby import SemGroupByOperator
from carnot.operators.sem_join import SemJoinOperator
from carnot.operators.sem_map import SemMapOperator
from carnot.operators.sem_topk import SemTopKOperator

Operator = CodeOperator | ReasoningOperator | SemAggOperator | SemFilterOperator | SemFlatMapOperator | SemGroupByOperator | SemJoinOperator | SemMapOperator | SemTopKOperator
logger = logging.getLogger('uvicorn.error')

class Execution:
    """
    Class for managing the execution of a query in Carnot.
    """
    def __init__(
            self,
            query: str,
            datasets: list[Dataset],
            plan: dict | None = None,
            tools: list[Tool] | None = None,
            conversation: Conversation | None = None,
            memory: Memory | None = None,
            indices: list[CarnotIndex] | None = None,
            llm_config: dict | None = None,
            progress_log_file: str | None = None,
        ):
        self.query = query
        self.datasets = datasets
        self._plan = plan or {}
        self.tools = tools or []
        self.conversation = conversation
        self.memory = memory or Memory()
        self.indices = indices or []
        self.llm_config = llm_config or {}
        self.progress_log_file = progress_log_file
        self.planner_model_id = "openai/gpt-5-2025-08-07"
        self.api_key_name = "OPENAI_API_KEY"
        if "OPENAI_API_KEY" not in self.llm_config and "ANTHROPIC_API_KEY" in self.llm_config:
            self.planner_model_id = "anthropic/claude-sonnet-4-5-20250929"
            self.api_key_name = "ANTHROPIC_API_KEY"
        elif "OPENAI_API_KEY" not in self.llm_config and "GEMINI_API_KEY" in self.llm_config:
            self.planner_model_id = "google/gemini-2.5-flash"
            self.api_key_name = "GEMINI_API_KEY"
        elif "OPENAI_API_KEY" not in self.llm_config and "GOOGLE_API_KEY" in self.llm_config:
            self.planner_model_id = "google/gemini-2.5-flash"
            self.api_key_name = "GOOGLE_API_KEY"
        self.planner = Planner(
            datasets=self.datasets,
            tools=self.tools, 
            model=LiteLLMModel(model_id=self.planner_model_id, api_key=llm_config.get(self.api_key_name))
        )

    def plan(self) -> tuple[str, dict]:
        """
        Generate a logical execution plan for the query.
        
        This method uses a two-phase approach:
        1. Generate a code-based logical plan (the Planner can call its managed 
           DataDiscoveryAgent to explore datasets during planning)
        2. Translate the logical plan into a natural language description for the user
        
        Returns:
            A tuple of (natural_language_plan, logical_plan_dict)
        """
        # Phase 1: Generate the code-based logical plan
        # The Planner can call its DataDiscoveryAgent as needed during planning
        logical_plan = self.planner.generate_logical_plan(
            self.query, 
            self.datasets, 
            conversation=self.conversation,
        )

        # Phase 2: Translate the logical plan to natural language for the user
        nl_plan = self.planner.paraphrase_logical_plan(
            self.query, 
            logical_plan, 
            self.datasets,
            conversation=self.conversation,
        )

        return nl_plan, logical_plan

    def _get_op_from_plan_dict(self, plan: dict) -> tuple[Operator | Dataset, list[str]]:
        """
        Return the physical operator associated with the given plan name.
        """
        # TODO: filter for model_id and max_workers from llm_config
        operator = None
        op_params = plan['params']
        op_name = op_params.get('operator', plan['name'])
        if op_name == "Code":
            operator = CodeOperator(task=op_params['task'], output_dataset_id=plan['output_dataset_id'], model_id="openai/gpt-5-mini", llm_config=self.llm_config)

        elif op_name == "Limit":
            operator = LimitOperator(n=op_params['n'], output_dataset_id=plan['output_dataset_id'])

        elif op_name == "SemanticAgg":
            operator = SemAggOperator(task=op_params['task'], agg_fields=op_params['agg_fields'], output_dataset_id=plan['output_dataset_id'], model_id="openai/gpt-5-mini", llm_config=self.llm_config, max_workers=4)

        elif op_name == "SemanticFilter":
            operator = SemFilterOperator(task=op_params['condition'], output_dataset_id=plan['output_dataset_id'], model_id="openai/gpt-5-mini", llm_config=self.llm_config, max_workers=4)

        elif op_name == "SemanticMap":
            output_fields = [{"name": op_params['field'], "type": op_params['type'], "description": op_params['field_desc']}]
            operator = SemMapOperator(
                task="Execute the map operation to compute the following output field.",
                output_fields=output_fields,
                output_dataset_id=plan['output_dataset_id'],
                model_id="openai/gpt-5-mini",
                llm_config=self.llm_config,
                max_workers=4,
            )

        elif op_name == "SemanticFlatMap":
            output_fields = [{"name": op_params['field'], "type": op_params['type'], "description": op_params['field_desc']}]
            operator = SemFlatMapOperator(
                task="Execute the flat map operation to compute the following output field.",
                output_fields=output_fields,
                output_dataset_id=plan['output_dataset_id'],
                model_id="openai/gpt-5-mini",
                llm_config=self.llm_config,
                max_workers=4,
            )

        elif op_name == "SemanticGroupBy":
            gby_field_names = [field['name'] for field in op_params['gby_fields']]
            agg_field_names = [field['name'] for field in op_params['agg_fields']]
            agg_funcs = [field['func'] for field in op_params['agg_fields']]
            task = f"Group by fields {gby_field_names} with aggregations on {agg_field_names} using {agg_funcs} for each aggregation field, respectively."
            operator = SemGroupByOperator(
                task=task,
                group_by_fields=op_params['gby_fields'],
                agg_fields=op_params['agg_fields'],
                output_dataset_id=plan['output_dataset_id'],
                model_id="openai/gpt-5-mini",
                llm_config=self.llm_config,
                max_workers=4,
            )

        elif op_name == "SemanticJoin":
            operator = SemJoinOperator(task=op_params['condition'], output_dataset_id=plan['output_dataset_id'], model_id="openai/gpt-5-mini", llm_config=self.llm_config, max_workers=4)

        elif op_name == "SemanticTopK":
            # Plan uses index_name; SemTopKOperator uses index_type (same semantic)
            index_type = op_params.get('index_type') or op_params.get('index_name') or 'chroma'
            operator = SemTopKOperator(task=op_params['search_str'], k=op_params['k'], output_dataset_id=plan['output_dataset_id'], model_id="openai/text-embedding-3-small", llm_config=self.llm_config, max_workers=4, index_type=index_type)

        else:
            for dataset in self.datasets:
                if dataset.name == op_name:
                    operator = dataset

        if operator is None:
            raise ValueError(f"Unknown operator or dataset name: {op_name}")

        return operator, [subplan['output_dataset_id'] for subplan in plan['parents']]

    def _get_ops_in_topological_order(self, plan: dict) -> list[tuple[Operator | Dataset, list[str]]]:
        """
        Get the operators in the physical plan in topological order. Returns a list of tuples of (operator, [parent_dataset_ids]).
        """
        # base case: this operator has no parents
        parents = plan.get('parents', [])
        if not parents:
            return [self._get_op_from_plan_dict(plan)]
        
        # recursive case: use DFS to get topological order, get parents first and then append this operator
        ops = []
        for parent in parents:
            ops.extend(self._get_ops_in_topological_order(parent))
        ops.append(self._get_op_from_plan_dict(plan))
        return ops

    def run(self) -> tuple[list[dict], str]: # physical_plan: PhysicalPlan -> str
        """
        Execute the physical plan and return the result.
        """
        # TODO: scope input_datasets based on children / parents in plan
        # TODO: capture output from final operator and return as result
        input_datasets = {}
        operators = self._get_ops_in_topological_order(self._plan)
        for operator, parent_ids in operators:
            if isinstance(operator, Dataset):
                # TODO: only materialize items as needed; should not have dictionaries here
                operator.items = [item.to_dict() if isinstance(item, DataItem) else item for item in operator.items]
                input_datasets = {operator.name: operator}
            elif isinstance(operator, CodeOperator):
                input_datasets = operator(input_datasets)
            elif isinstance(operator, SemJoinOperator):
                left_dataset_id = parent_ids[0]
                right_dataset_id = parent_ids[1]
                input_datasets = operator(left_dataset_id, right_dataset_id, input_datasets)
            else:
                dataset_id = parent_ids[0]
                input_datasets = operator(dataset_id, input_datasets)

        # TODO: use a (reasoning?) operator to return a final output dataset which has all of the items, (code_state,) and/or answer text
        # from the input datasets which is relevant to the user for interpreting the final answer
        # - list of items (dicts) can be written to a pd.DataFrame --> csv
        # - text can be displayed to the user
        # - for now, assume code state is debug only (exposed in the future)
        final_answer_operator = ReasoningOperator(task=self.query, output_dataset_id="final_dataset", model_id="openai/gpt-5-mini", llm_config=self.llm_config)
        output_datasets = final_answer_operator(input_datasets)
        final_dataset = output_datasets["final_dataset"]

        return final_dataset.items, final_dataset.code_state.get("final_answer_str", "")
