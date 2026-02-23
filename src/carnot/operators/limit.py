from carnot.data.dataset import Dataset


class LimitOperator:
    """
    Represents a limit operator.
    """
    def __init__(self, n: int, output_dataset_id: str):
        self.n = n
        self.output_dataset_id = output_dataset_id

    def __call__(self, dataset_id: str, input_datasets: dict[str, Dataset]) -> dict[str, Dataset]:
        """
        Apply a semantic top-k operator to the input dataset specified by the `dataset_id`.
        Semantic Top-K operator uses the input dataset's index() method to retrieve the top-k most semantically similar items.
        If the index() method is not implemented for the dataset, then the index is constructed on-the-fly.
        """
        # retrieve the input dataset
        input_dataset = input_datasets[dataset_id]
        
        # apply the limit operation to the dataset items
        results = input_dataset.items[:self.n]

        # create new dataset and return it with the input datasets
        output_dataset = Dataset(name=self.output_dataset_id, annotation=f"Limit operator output for n: {self.n}", items=results)
        output_datasets = {**input_datasets, output_dataset.name: output_dataset}

        return output_datasets
