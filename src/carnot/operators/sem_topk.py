from __future__ import annotations

from carnot.data.dataset import Dataset
from carnot.index import FlatCarnotIndex, HierarchicalCarnotIndex
from carnot.index.index import ChromaIndex, FaissIndex


class SemTopKOperator:
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
        output_dataset_id: str,
        max_workers: int,
        model_id: str = "openai/text-embedding-3-small",
        llm_config: dict | None = None,
        index_name: str = "chroma",
        catalog=None,
    ):
        self.task = task
        self.output_dataset_id = output_dataset_id
        self.k = k
        self.model_id = model_id
        self.api_key = llm_config.get("OPENAI_API_KEY")
        self.index_name = index_name
        self.catalog = catalog
        index_map = {
            "chroma": ChromaIndex,
            "faiss": FaissIndex,
            "hierarchical": HierarchicalCarnotIndex,
            "flat": FlatCarnotIndex,
        }
        self.index_cls = index_map[index_name]

    def __call__(self, dataset_id: str, input_datasets: dict[str, Dataset]) -> dict[str, Dataset]:
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
            A new ``dict[str, Dataset]`` that is a copy of
            *input_datasets* with an additional entry keyed by
            ``self.output_dataset_id`` containing up to *k* items.

        Raises:
            KeyError: If *dataset_id* is not in *input_datasets*.
        """
        input_dataset = input_datasets[dataset_id]

        if self.index_name not in input_dataset.list_indices():
            # Try to load a previously-built index from the catalog
            loaded = self._load_from_catalog(input_dataset)

            if not loaded:
                # Build a new index from scratch
                disk_name = f"ds{input_dataset.dataset_id}_{self.index_name}"

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

        output_dataset = Dataset(name=self.output_dataset_id, annotation=f"Sem Top-K operator output for task: {self.task}", items=results)
        output_datasets = {**input_datasets, output_dataset.name: output_dataset}

        return output_datasets

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
            disk_name = f"ds{dataset.dataset_id}_{self.index_name}"
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
