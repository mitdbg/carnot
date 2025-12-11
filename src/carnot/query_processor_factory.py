import logging
import os
from enum import Enum

from carnot.config import QueryProcessorConfig
from carnot.core.data.dataset import Dataset
from carnot.core.elements.records import DataRecordCollection
from carnot.execution.execution_strategy import ExecutionStrategy
from carnot.execution.parallel_execution_strategy import ParallelExecutionStrategy
from carnot.optimizer.cost_model import SampleBasedCostModel
from carnot.optimizer.optimizer import Optimizer
from carnot.optimizer.optimizer_strategy_type import OptimizationStrategyType
from carnot.query_processor import QueryProcessor
from carnot.utils.model_helpers import get_models

logger = logging.getLogger(__name__)


class QueryProcessorFactory:

    @classmethod
    def _convert_to_enum(cls, enum_type: type[Enum], value: str) -> Enum:
        value = value.upper().replace('-', '_')
        try:
            return enum_type[value]
        except KeyError as e:
            raise ValueError(f"Unsupported {enum_type.__name__}: {value}") from e

    @classmethod
    def _normalize_strategies(cls, config: QueryProcessorConfig):
        """
        Convert the string representation of each strategy into its Enum equivalent and throw
        an exception if the conversion fails.
        """
        # Only handle optimizer strategy since execution strategy is always parallel
        strategy_str = getattr(config, "optimizer_strategy")
        if strategy_str is not None:
            try:
                strategy_enum = cls._convert_to_enum(OptimizationStrategyType, strategy_str)
                setattr(config, "optimizer_strategy", strategy_enum)
                logger.debug(f"Normalized optimizer_strategy: {strategy_enum}")
            except ValueError as e:
                raise ValueError(f"""Unsupported optimizer_strategy: {strategy_str}.
                                    The supported strategies are: {OptimizationStrategyType.__members__.keys()}""") from e

        return config

    @classmethod
    def _config_validation_and_normalization(cls, config: QueryProcessorConfig):
        if config.policy is None:
            raise ValueError("Policy is required for optimizer")
        
        # only one of progress or verbose can be set; we will default to progress=True
        if config.progress and config.verbose:
            print("WARNING: Both `progress` and `verbose` are set to True, but only one can be True at a time; defaulting to `progress=True`")
            config.verbose = False

        # convert the config values for processing, execution, and optimization strategies to enums
        config = cls._normalize_strategies(config)

        # get available models
        available_models = getattr(config, 'available_models', [])
        if available_models is None or len(available_models) == 0:
            available_models = get_models(use_vertex=config.use_vertex, gemini_credentials_path=config.gemini_credentials_path, api_base=config.api_base)

        # remove any models specified in the config
        remove_models = getattr(config, 'remove_models', [])
        if remove_models is not None and len(remove_models) > 0:
            available_models = [model for model in available_models if model not in remove_models]
            logger.info(f"Removed models from available models based on config: {remove_models}")

        # set the final set of available models in the config
        config.available_models = available_models

        return config

    @classmethod
    def _create_optimizer(cls, config: QueryProcessorConfig) -> Optimizer:
        return Optimizer(cost_model=SampleBasedCostModel(), **config.to_dict())

    @classmethod
    def _create_execution_strategy(cls, dataset: Dataset, config: QueryProcessorConfig) -> ExecutionStrategy:
        """
        Creates an execution strategy based on the configuration.
        Since we only have parallel execution, we always use ParallelExecutionStrategy.
        """
        # for parallel execution, set the batch size if there's a limit in the query
        limit = dataset.get_limit()
        if limit is not None:
            if config.batch_size is None:
                config.batch_size = limit
                logger.info(f"Setting batch size to query limit: {limit}")
            elif config.batch_size > limit:
                config.batch_size = limit
                logger.info(f"Setting batch size to query limit: {limit} since it was larger than the limit")

        # create the execution strategy (always parallel)
        return ParallelExecutionStrategy(**config.to_dict())

    @classmethod
    def create_processor(
        cls,
        dataset: Dataset,
        config: QueryProcessorConfig | None = None,
    ) -> QueryProcessor:
        """
        Creates a QueryProcessor with specified processing and execution strategies.

        Args:
            dataset: The dataset to process
            config: The user-provided QueryProcessorConfig; if it is None, the default config will be used
            kwargs: Additional keyword arguments to pass to the QueryProcessorConfig
        """
        if config is None:
            config = QueryProcessorConfig()

        # apply any additional keyword arguments to the config and validate its contents
        config = cls._config_validation_and_normalization(config)

        # if there are any API keys specified in the llm_config, set them in the config
        if len(config.llm_config.keys()) > 0:
            for key, value in config.llm_config.items():
                os.environ[key] = value

        # update the dataset's types if we're not enforcing types
        if not config.enforce_types:
            dataset.relax_types()

        # create the optimizer, execution strateg(ies), and processor
        optimizer = cls._create_optimizer(config)
        config.execution_strategy = cls._create_execution_strategy(dataset, config)
        processor = QueryProcessor(dataset, optimizer, **config.to_dict())

        return processor

    @classmethod
    def create_and_run_processor(
        cls,
        dataset: Dataset,
        config: QueryProcessorConfig | None = None,
    ) -> DataRecordCollection:
        logger.info(f"Creating processor for dataset: {dataset}")
        processor = cls.create_processor(dataset, config)
        logger.info(f"Created processor: {processor}")

        return processor.execute()
