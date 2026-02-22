from carnot.data.dataset import Dataset
from carnot.index import FlatCarnotIndex, HierarchicalCarnotIndex
from carnot.index.index import ChromaIndex, FaissIndex
from carnot.utils.hash_helpers import hash_for_id


class SemTopKOperator:
    """
    Represents a semantic top-k operator.
    """
    def __init__(self, task: str, k: int, output_dataset_id: str, max_workers: int, model_id: str = "openai/text-embedding-3-small", llm_config: dict | None = None, index_type: str = "chroma"):
        self.task = task
        self.output_dataset_id = output_dataset_id
        self.k = k
        self.model_id = model_id
        self.api_key = llm_config.get("OPENAI_API_KEY")
        # self.max_workers = max_workers
        self.index_type = index_type
        index_map = {
            "chroma": ChromaIndex,
            "faiss": FaissIndex,
            "hierarchical": HierarchicalCarnotIndex,
            "flat": FlatCarnotIndex,
        }  
        self.index_cls = index_map[index_type]
        # self.prompt_templates = yaml.safe_load(
        #     resources.files("carnot.agents.prompts").joinpath("sem_topk.yaml").read_text()
        # )

    def __call__(self, dataset_id: str, input_datasets: dict[str, Dataset]) -> dict[str, Dataset]:
        """
        Apply a semantic top-k operator to the input dataset specified by the `dataset_id`.
        Semantic Top-K operator uses the input dataset's index() method to retrieve the top-k most semantically similar items.
        If the index() method is not implemented for the dataset, then the index is constructed on-the-fly.
        """
        # retrieve the input dataset
        input_dataset = input_datasets[dataset_id]

        # check if the dataset has an index constructed and construct one on-the-fly if not
        if not input_dataset.has_index():
            name = f"{hash_for_id(input_dataset.name)}_{hash_for_id(self.task)}"
            index = self.index_cls(name=name, items=input_dataset.items, model=self.model_id, api_key=self.api_key)
            input_dataset._indices[self.index_type] = index

        # invoke the search() method on the index to retrieve items
        results = input_dataset._indices[self.index_type].search(self.task, k=self.k)

        # TODO: we could construct "instance-optimized" indices by using a map operator to extract the relevant info
        #       from the input items, and then building an index on that extracted info
        # # pre-populate system prompt
        # system_prompt = populate_template(
        #     self.prompt_templates["sem_topk_prompt"],
        #     variables={
        #         "output_opening_tag": self.output_tags[0],
        #         "output_closing_tag": self.output_tags[1],
        #     },
        # )

        # # construct futures
        # futures = []
        # with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
        #     for item in items:
        #         future = executor.submit(self._sem_map, item, system_prompt)
        #         futures.append(future)

        # # block until futures complete
        # done_futures, _ = wait(futures)
        # results = [fut.result() for fut in done_futures]

        # create new dataset and return it with the input datasets
        output_dataset = Dataset(name=self.output_dataset_id, annotation=f"Sem Top-K operator output for task: {self.task}", items=results)
        output_datasets = {**input_datasets, output_dataset.name: output_dataset}

        return output_datasets
