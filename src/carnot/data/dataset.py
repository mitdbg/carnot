import textwrap

from carnot.data.item import DataItem
from carnot.index.index import CarnotIndex


class Dataset:
    """
    A collection of files (DataItems) and associated metadata.
    """
    def __init__(self, name: str, annotation: str, items: list[DataItem] | None = None, index: CarnotIndex | None = None, code: str | None = None, code_state: dict | None = None):
        self.name = name
        self.annotation = annotation
        self.items = items or []
        self._index = index
        self.code = code
        self.code_state = code_state or {}

    def format_description(self, code_block_tags: list[str]) -> str:
        code_str = " None" if self.code is None else f"\n{code_block_tags[0]}\n{self.code}\n{code_block_tags[1]}"
        return textwrap.dedent(
            f"Dataset Name: {self.name}\n"
            f"Annotation: {self.annotation}\n"
            f"Number of Items: {len(self.items)}\n"
            f"Index: {'yes' if self.index else 'no'}\n"
            f"Code that Generated Code State: {code_str}\n"
            f"Available Code State Vars: {list(self.code_state.keys())}\n"
        )

    def has_index(self) -> bool:
        return self._index is not None

    def index(self, query: str, k: int = 5) -> list[DataItem]:
        if self._index is None:
            raise NotImplementedError("Dataset does not have an index constructed.")
        return self._index.search(query, k=k)
