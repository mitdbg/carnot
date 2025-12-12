import logging
import os

from carnot.constants import Model
from carnot.core.data.context_manager import ContextManager
from carnot.core.lib.schemas import (
    AUDIO_FIELD_TYPES,
    AUDIO_LIST_FIELD_TYPES,
    IMAGE_FIELD_TYPES,
    IMAGE_LIST_FIELD_TYPES,
)
from carnot.operators.compute import SmolAgentsCompute
from carnot.operators.logical import (
    BaseScan,
    ComputeOperator,
    ContextScan,
    FilteredScan,
    SearchOperator,
)
from carnot.operators.physical import PhysicalOperator
from carnot.operators.scan import ContextScanOp, MarshalAndScanDataOp
from carnot.operators.search import (
    SmolAgentsSearch,  # SmolAgentsCustomManagedSearch,  # SmolAgentsManagedSearch
)
from carnot.optimizer.primitives import Expression, Group, LogicalExpression, PhysicalExpression
from carnot.prompts import CONTEXT_SEARCH_PROMPT

logger = logging.getLogger(__name__)


class Rule:
    """
    The abstract base class for transformation and implementation rules.
    """

    @classmethod
    def get_rule_id(cls):
        return cls.__name__

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
        raise NotImplementedError("Calling this method from an abstract base class.")

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **kwargs: dict) -> set[Expression]:
        raise NotImplementedError("Calling this method from an abstract base class.")


class TransformationRule(Rule):
    """
    Base class for transformation rules which convert a logical expression to another logical expression.
    The substitute method for a TransformationRule should return all new expressions and all new groups
    which are created during the substitution.
    """

    @classmethod
    def is_exploration_rule(cls) -> bool:
        """Returns True if this rule is an exploration rule and False otherwise. Default is False."""
        return False

    @classmethod
    def substitute(
        cls, logical_expression: LogicalExpression, groups: dict[int, Group], expressions: dict[int, Expression], **kwargs
    ) -> tuple[set[LogicalExpression], set[Group]]:
        """
        This function applies the transformation rule to the logical expression, which
        potentially creates new intermediate expressions and groups.

        The function returns a tuple containing:
        - the set of all new logical expressions created when applying the transformation
        - the set of all new groups created when applying the transformation
        - the next group id (after creating any new groups)
        """
        raise NotImplementedError("Calling this method from an abstract base class.")


class ImplementationRule(Rule):
    """
    Base class for implementation rules which convert a logical expression to a physical expression.
    """

    @classmethod
    def _get_image_fields(cls, logical_expression: LogicalExpression) -> set[str]:
        """Returns the set of fields which have an image (or list[image]) type."""
        return set([
            field_name.split(".")[-1]
            for field_name, field in logical_expression.input_fields.items()
            if field.annotation in IMAGE_FIELD_TYPES and field_name.split(".")[-1] in logical_expression.depends_on_field_names
        ])

    @classmethod
    def _get_list_image_fields(cls, logical_expression: LogicalExpression) -> set[str]:
        """Returns the set of fields which have a list[image] type."""
        return set([
            field_name.split(".")[-1]
            for field_name, field in logical_expression.input_fields.items()
            if field.annotation in IMAGE_LIST_FIELD_TYPES and field_name.split(".")[-1] in logical_expression.depends_on_field_names
        ])

    @classmethod
    def _get_audio_fields(cls, logical_expression: LogicalExpression) -> set[str]:
        """Returns the set of fields which have an audio (or list[audio]) type."""
        return set([
            field_name.split(".")[-1]
            for field_name, field in logical_expression.input_fields.items()
            if field.annotation in AUDIO_FIELD_TYPES and field_name.split(".")[-1] in logical_expression.depends_on_field_names
        ])

    @classmethod
    def _get_list_audio_fields(cls, logical_expression: LogicalExpression) -> set[str]:
        """Returns the set of fields which have a list[audio] type."""
        return set([
            field_name.split(".")[-1]
            for field_name, field in logical_expression.input_fields.items()
            if field.annotation in AUDIO_LIST_FIELD_TYPES and field_name.split(".")[-1] in logical_expression.depends_on_field_names
        ])

    @classmethod
    def _is_image_only_operation(cls, logical_expression: LogicalExpression) -> bool:
        """Returns True if the logical_expression processes only image input(s) and False otherwise."""
        return all([
            field.annotation in IMAGE_FIELD_TYPES
            for field_name, field in logical_expression.input_fields.items()
            if field_name.split(".")[-1] in logical_expression.depends_on_field_names
        ])

    @classmethod
    def _is_image_operation(cls, logical_expression: LogicalExpression) -> bool:
        """Returns True if the logical_expression processes image input(s) and False otherwise."""
        return any([
            field.annotation in IMAGE_FIELD_TYPES
            for field_name, field in logical_expression.input_fields.items()
            if field_name.split(".")[-1] in logical_expression.depends_on_field_names
        ])

    @classmethod
    def _is_audio_only_operation(cls, logical_expression: LogicalExpression) -> bool:
        """Returns True if the logical_expression processes only audio input(s) and False otherwise."""
        return all([
            field.annotation in AUDIO_FIELD_TYPES
            for field_name, field in logical_expression.input_fields.items()
            if field_name.split(".")[-1] in logical_expression.depends_on_field_names
        ])

    @classmethod
    def _is_audio_operation(cls, logical_expression: LogicalExpression) -> bool:
        """Returns True if the logical_expression processes audio input(s) and False otherwise."""
        return any([
            field.annotation in AUDIO_FIELD_TYPES
            for field_name, field in logical_expression.input_fields.items()
            if field_name.split(".")[-1] in logical_expression.depends_on_field_names
        ])

    @classmethod
    def _is_text_only_operation(cls, logical_expression: LogicalExpression) -> bool:
        """Returns True if the logical_expression processes only text input(s) and False otherwise."""
        return all([
            field.annotation not in IMAGE_FIELD_TYPES + AUDIO_FIELD_TYPES
            for field_name, field in logical_expression.input_fields.items()
            if field_name.split(".")[-1] in logical_expression.depends_on_field_names
        ])

    @classmethod
    def _is_text_operation(cls, logical_expression: LogicalExpression) -> bool:
        """Returns True if the logical_expression processes text input(s) and False otherwise."""
        return any([
            field.annotation not in IMAGE_FIELD_TYPES + AUDIO_FIELD_TYPES
            for field_name, field in logical_expression.input_fields.items()
            if field_name.split(".")[-1] in logical_expression.depends_on_field_names
        ])

    # TODO: support powerset of text + image + audio (+ video) multi-modal operations
    @classmethod
    def _is_text_image_multimodal_operation(cls, logical_expression: LogicalExpression) -> bool:
        """Returns True if the logical_expression processes text and image inputs and False otherwise."""
        return cls._is_image_operation(logical_expression) and cls._is_text_operation(logical_expression)

    @classmethod
    def _is_text_audio_multimodal_operation(cls, logical_expression: LogicalExpression) -> bool:
        """Returns True if the logical_expression processes text and audio inputs and False otherwise."""
        return cls._is_audio_operation(logical_expression) and cls._is_text_operation(logical_expression)

    @classmethod
    def _model_matches_input(cls, model: Model, logical_expression: LogicalExpression) -> bool:
        """Returns True if the model is capable of processing the input and False otherwise."""
        # compute how many image fields are in the input, and whether any fields are list[image] fields
        num_image_fields = len(cls._get_image_fields(logical_expression))
        has_list_image_field = len(cls._get_list_image_fields(logical_expression)) > 0
        num_audio_fields = len(cls._get_audio_fields(logical_expression))
        has_list_audio_field = len(cls._get_list_audio_fields(logical_expression)) > 0

        # corner-case: for now, all operators use text or vision models for processing inputs to __call__
        if model.is_embedding_model():
            return False

        # corner-case: Llama vision models cannot handle multiple image inputs (at least using Together)
        if model.is_llama_model() and model.is_vision_model() and (num_image_fields > 1 or has_list_image_field):
            return False

        # corner-case: Gemini models cannot handle multiple audio inputs
        if model.is_vertex_model() and model.is_audio_model() and (num_audio_fields > 1 or has_list_audio_field):
            return False

        # text-only input and text supporting model
        if cls._is_text_only_operation(logical_expression) and model.is_text_model():
            return True

        # image-only input and image supporting model
        if cls._is_image_only_operation(logical_expression) and model.is_vision_model():
            return True

        # audio-only input and audio supporting model
        if cls._is_audio_only_operation(logical_expression) and model.is_audio_model():
            return True

        # multi-modal input and multi-modal supporting model
        if cls._is_text_image_multimodal_operation(logical_expression) and model.is_text_image_multimodal_model():  # noqa: SIM103
            return True

        # multi-modal input and multi-modal supporting model
        if cls._is_text_audio_multimodal_operation(logical_expression) and model.is_text_audio_multimodal_model():  # noqa: SIM103
            return True

        return False

    @classmethod
    def _get_fixed_op_kwargs(cls, logical_expression: LogicalExpression, runtime_kwargs: dict) -> dict:
        """Get the fixed set of physical op kwargs provided by the logical expression and the runtime keyword arguments."""
        # get logical operator 
        logical_op = logical_expression.operator

        # set initial set of parameters for physical op
        op_kwargs = logical_op.get_logical_op_params()
        op_kwargs.update(
            {
                "verbose": runtime_kwargs["verbose"],
                "logical_op_id": logical_op.get_logical_op_id(),
                "unique_logical_op_id": logical_op.get_unique_logical_op_id(),
                "logical_op_name": logical_op.logical_op_name(),
                "api_base": runtime_kwargs["api_base"],
            }
        )

        return op_kwargs

    @classmethod
    def _perform_substitution(
        cls,
        logical_expression: LogicalExpression,
        physical_op_class: type[PhysicalOperator],
        runtime_kwargs: dict,
        variable_op_kwargs: list[dict] | dict | None = None,
    ) -> set[PhysicalExpression]:
        """
        This performs basic substitution logic which proceeds in four steps:

            1. The basic kwargs for the physical operator are computed using the logical operator
               and runtime kwargs.
            2. If variable kwargs are provided, then they are merged with the basic kwargs and one
               instance of the physical operator is created for each dictionary of variable kwargs.
            3. A physical expression is created for each physical operator instance.
            4. The unique set of physical expressions is returned.

        Args:
            logical_expression (LogicalExpression): The logical expression containing a logical operator.
            physical_op_class (type[PhysicalOperator]): The class of the physical operator we wish to construct.
            runtime_kwargs (dict): Keyword arguments which are provided at runtime.
            variable_op_kwargs (list[dict] | dict | None): A (list of) variable kwargs to customize each
                physical operator instance.

        Returns:
            set[PhysicalExpression]: The unique set of physical expressions produced by initializing the
                physical_op_class with the provided keyword arguments.
        """
        # get physical operator kwargs which are fixed for each instance of the physical operator
        fixed_op_kwargs = cls._get_fixed_op_kwargs(logical_expression, runtime_kwargs)

        # make variable_op_kwargs a list of dictionaries
        if variable_op_kwargs is None:
            variable_op_kwargs = [{}]
        elif isinstance(variable_op_kwargs, dict):
            variable_op_kwargs = [variable_op_kwargs]

        # construct physical operators for each set of kwargs
        physical_expressions = []
        for var_op_kwargs in variable_op_kwargs:
            # get kwargs for this physical operator instance
            op_kwargs = {**fixed_op_kwargs, **var_op_kwargs}

            # construct the physical operator
            op = physical_op_class(**op_kwargs)

            # construct physical expression and add to list of expressions
            expression = PhysicalExpression.from_op_and_logical_expr(op, logical_expression)
            physical_expressions.append(expression)

        return set(physical_expressions)


class AddContextsBeforeComputeRule(ImplementationRule):
    """
    Searches the ContextManager for additional contexts which may be useful for the given computation.

    TODO: track cost of generating search query
    """
    k = 1
    SEARCH_GENERATOR_PROMPT = CONTEXT_SEARCH_PROMPT

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
        is_match = isinstance(logical_expression.operator, ComputeOperator)
        logger.debug(f"AddContextsBeforeComputeRule matches_pattern: {is_match} for {logical_expression}")
        return is_match

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **runtime_kwargs) -> set[PhysicalExpression]:
        logger.debug(f"Substituting AddContextsBeforeComputeRule for {logical_expression}")

        # load an LLM to generate a short search query
        model = None
        if os.getenv("OPENAI_API_KEY"):
            model = "openai/gpt-4o-mini"
        elif os.getenv("ANTHROPIC_API_KEY"):
            model = "anthropic/claude-3-5-sonnet-20241022"
        elif os.getenv("GEMINI_API_KEY"):
            model = "vertex_ai/gemini-2.0-flash"
        elif os.getenv("TOGETHER_API_KEY"):
            model = "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"

        # importing litellm here because importing above causes deprecation warning
        import litellm

        # retrieve any additional context which may be useful
        cm = ContextManager()
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": cls.SEARCH_GENERATOR_PROMPT.format(instruction=logical_expression.operator.instruction)}]
        )
        query = response.choices[0].message.content
        variable_op_kwargs = {"additional_contexts": cm.search_context(query, k=cls.k, where={"materialized": True})}
        return cls._perform_substitution(logical_expression, SmolAgentsCompute, runtime_kwargs, variable_op_kwargs)


class BasicSubstitutionRule(ImplementationRule):
    """
    For logical operators with a single physical implementation, substitute the
    logical expression with its physical counterpart.
    """

    LOGICAL_OP_CLASS_TO_PHYSICAL_OP_CLASS_MAP = {
        BaseScan: MarshalAndScanDataOp,
        # ComputeOperator: SmolAgentsCompute,
        SearchOperator: SmolAgentsSearch, # SmolAgentsManagedSearch, # SmolAgentsCustomManagedSearch
        ContextScan: ContextScanOp,
        FilteredScan: SmolAgentsSearch,
    }

    @classmethod
    def matches_pattern(cls, logical_expression: LogicalExpression) -> bool:
        logical_op_class = logical_expression.operator.__class__
        is_match = logical_op_class in cls.LOGICAL_OP_CLASS_TO_PHYSICAL_OP_CLASS_MAP
        logger.debug(f"BasicSubstitutionRule matches_pattern: {is_match} for {logical_expression}")
        return is_match

    @classmethod
    def substitute(cls, logical_expression: LogicalExpression, **runtime_kwargs) -> set[PhysicalExpression]:
        logger.debug(f"Substituting BasicSubstitutionRule for {logical_expression}")
        physical_op_class = cls.LOGICAL_OP_CLASS_TO_PHYSICAL_OP_CLASS_MAP[logical_expression.operator.__class__]
        return cls._perform_substitution(logical_expression, physical_op_class, runtime_kwargs)
