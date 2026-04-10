from __future__ import annotations

import time

from carnot.core.models import LLMCallStats, OperatorStats
from carnot.data.dataset import Dataset
from carnot.index import FlatCarnotIndex, HierarchicalCarnotIndex
from carnot.index.index import ChromaIndex, FaissIndex
from carnot.operators.physical import PhysicalOperator
from carnot.optimizer.model_ids import get_api_key_for_model


class SemTopKOperator(PhysicalOperator):
    """Semantic top-k operator — retrieves the *k* most relevant items via index search.

    Unlike other semantic operators this does **not** use an LLM for
    generation.  Instead it relies on a pre-built (or on-the-fly)
    embedding index (Chroma, FAISS, Flat, or Hierarchical) to find the
    items closest to ``task``.

    When a new index is constructed on-the-fly, the **operator** is
    responsible for registering it with the ``IndexCatalog`` (if one
    was provided at construction time).

    Index naming:
        On-disk index names are formed as
        ``"ds{dataset_id}_{index_name}"`` to avoid collisions when
        multiple datasets use the same index kind (e.g. two datasets
        both requesting ``"chroma"``).  This mirrors the catalog's
        composite key ``(dataset_id, name)``.

    Representation invariant:
        - ``k >= 1``.
        - ``index_cls`` is one of the supported index classes.
        - ``index_name`` is a key in the supported ``index_map``.

    Abstraction function:
        An instance of this class is a callable that, given a dataset,
        returns a new dataset containing the *k* items most
        semantically similar to ``task``.
    """

    def __init__(
        self,
        task: str,
        k: int,
        dataset_id: str,
        max_workers: int,
        model_id: str = "openai/text-embedding-3-small",
        llm_config: dict | None = None,
        index_name: str = "chroma",
        catalog=None,
        logical_op_id: str | None = None,
        logical_op_class_name: str | None = None,
    ):
        super().__init__(logical_op_id=logical_op_id, logical_op_class_name=logical_op_class_name)
        self.task = task
        self.k = k
        self.dataset_id = dataset_id
        self.max_workers = max_workers
        self.model_id = model_id
        self.llm_config = llm_config or {}
        self.api_key = get_api_key_for_model(model_id, llm_config or {})
        self.index_name = index_name
        self.catalog = catalog
        index_map = {
            "chroma": ChromaIndex,
            "faiss": FaissIndex,
            "hierarchical": HierarchicalCarnotIndex,
            "flat": FlatCarnotIndex,
        }
        self.index_cls = index_map[index_name]

    def get_id_params(self):
        id_params = super().get_id_params()
        id_params = {
            "task": self.task,
            "k": self.k,
            "dataset_id": self.dataset_id,
            "model_id": self.model_id,
            "index_name": self.index_name,
            **id_params,
        }

        return id_params

    def get_op_params(self):
        op_params = super().get_op_params()
        op_params = {
            "task": self.task,
            "k": self.k,
            "dataset_id": self.dataset_id,
            "model_id": self.model_id,
            "llm_config": self.llm_config,
            "max_workers": self.max_workers,
            "index_name": self.index_name,
            "catalog": self.catalog,
            **op_params,
        }

        return op_params

    def __call__(self, dataset_id: str, input_datasets: dict[str, Dataset]) -> tuple[dict[str, Dataset], OperatorStats]:
        """Retrieve the top-k items from the input dataset via index search.

        If the dataset does not already have an index of the configured
        type, the operator checks the catalog for a previously-built
        index.  Only when both the ephemeral dataset cache and the
        catalog miss does it build a new index from scratch.

        Index names are scoped by ``dataset_id`` to prevent on-disk
        collisions between datasets, using the format
        ``"ds{dataset_id}_{index_name}"``.

        Requires:
            - *dataset_id* is a key in *input_datasets*.

        Returns:
            A tuple ``(output_datasets, stats)`` where *output_datasets*
            is a new ``dict[str, Dataset]`` with an additional entry
            keyed by ``self.dataset_id`` containing up to *k*
            items, and *stats* is an :class:`OperatorStats` with
            embedding call statistics collected from the index.

        Raises:
            KeyError: If *dataset_id* is not in *input_datasets*.
        """
        op_start = time.perf_counter()

        input_dataset = input_datasets[dataset_id]

        if self.index_name not in input_dataset.list_indices():
            # Try to load a previously-built index from the catalog
            loaded = self._load_from_catalog(input_dataset)

            if not loaded:
                # Build a new index from scratch
                disk_name = f"ds{input_dataset.dataset_id.replace(' ', '')}_{self.index_name}"

                # NOTE: I believe that input_dataset.items will be materialized here;
                #   in the future we may want to add support for lazy materialization within the index classes themselves
                index = self.index_cls(
                    name=disk_name,
                    items=input_dataset.items,
                    model=self.model_id,
                    api_key=self.api_key,
                )
                input_dataset.indices[self.index_name] = index

                # Register with the catalog (operator's responsibility)
                if self.catalog is not None and input_dataset.dataset_id is not None:
                    self.catalog.register_index(
                        dataset_id=input_dataset.dataset_id,
                        name=self.index_name,
                        index_type=index.__class__.__name__,
                        index_obj=index,
                    )

        results = input_dataset.indices[self.index_name].search(self.task, k=self.k)

        # Collect embedding stats from the index if available
        embed_stats: list[LLMCallStats] = []
        index_obj = input_dataset.indices[self.index_name]
        if hasattr(index_obj, "_llm_call_stats"):
            embed_stats = list(index_obj._llm_call_stats)

        output_dataset = Dataset(name=self.dataset_id, annotation=f"Sem Top-K operator output for task: {self.task}", items=results)
        output_datasets = {**input_datasets, output_dataset.name: output_dataset}

        op_stats = OperatorStats(
            operator_name="SemTopK",
            operator_id=self.dataset_id,
            wall_clock_secs=time.perf_counter() - op_start,
            llm_calls=embed_stats,
            items_in=len(input_dataset.items),
            items_out=len(results),
        )

        return output_datasets, op_stats

    def _load_from_catalog(self, dataset: Dataset) -> bool:
        """Attempt to load a pre-built index from the catalog.

        Checks the catalog for a non-stale index matching this
        dataset and index name.  If found, wraps it in the
        appropriate ``CarnotIndex`` subclass (with items for result
        mapping) and attaches it to the dataset.

        Requires:
            - ``self.catalog`` may be ``None``.
            - ``dataset`` is a ``Dataset`` with a ``dataset_id``.

        Returns:
            ``True`` if an index was successfully loaded and attached
            to the dataset; ``False`` otherwise.

        Raises:
            None.  Errors are logged and treated as a cache miss.
        """
        if self.catalog is None or dataset.dataset_id is None:
            return False

        try:
            meta = self.catalog.get_index_by_name(
                dataset.dataset_id, self.index_name
            )
            if meta is None or meta.is_stale:
                return False

            index_obj = self.catalog.load_index(meta.id)
            if index_obj is None:
                return False

            # Wrap the loaded inner index in the appropriate CarnotIndex
            # subclass, passing items for URI→item mapping.
            disk_name = f"ds{dataset.dataset_id.replace(' ', '')}_{self.index_name}"
            wrapped = self.index_cls(
                name=disk_name,
                items=dataset.items,
                model=self.model_id,
                api_key=self.api_key,
                index=index_obj,
            )
            dataset.indices[self.index_name] = wrapped
            return True
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(
                f"Failed to load index from catalog for dataset {dataset.dataset_id}: {e}"
            )
            return False
