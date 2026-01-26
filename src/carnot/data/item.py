


class DataItem:
    """
    Reference to a data item in Carnot. Each item is stored at the provided absolute path.
    Data items may also have embeddings associated with them, as well as dictionary metadata.
    """
    def __init__(self, path: str, embedding: list[float] | None = None, metadata: dict | None = None):
        self.path = path
        self.embedding = embedding
        self.metadata = metadata or {}
