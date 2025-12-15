import logging
import sys
from typing import Dict, List, Set, Union

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

logger = logging.getLogger(__name__)

# retrieve single query (used for decompose + retrieve)
def retrieve(collection, query: str, k: int, include_chunks: bool = False) -> Union[Set[str], Dict[str, str]]:
    """
    Retrieves k results for a single query string.
    """
    if k <= 0:
        return {} if include_chunks else set()

    include_fields = ["metadatas"]
    if include_chunks:
        include_fields.append("documents")

    results = collection.query(
        query_texts=[query], 
        n_results=k, 
        include=include_fields
    )

    return _process_single_result(results, 0, include_chunks)


# batch retrieval (used for retrieve)
def retrieve_batch(
    collection, 
    queries: List[str], 
    k: int, 
    include_chunks: bool = False
) -> List[Union[Set[str], Dict[str, str]]]:
    """
    Retrieves k results for a list of queries efficiently (vectorized).
    """
    if not queries or k <= 0:
        return []

    include_fields = ["metadatas"]
    if include_chunks:
        include_fields.append("documents")

    results = collection.query(
        query_texts=queries, 
        n_results=k, 
        include=include_fields
    )

    batch_output = []
    for i in range(len(queries)):
        processed = _process_single_result(results, i, include_chunks)
        batch_output.append(processed)
        
    return batch_output


# helper function to deduplicate the results
def _process_single_result(results, index, include_chunks):
    """
    Internal helper to extract and deduplicate logic for a specific query index.
    """
    if not results.get("metadatas") or not results["metadatas"][index]:
        return {} if include_chunks else set()

    metas = results["metadatas"][index]
    docs_text = results["documents"][index] if include_chunks else []

    if include_chunks:
        title_to_chunk = {}
        for i, meta in enumerate(metas):
            title = meta.get("title", "No Title")
            if title not in title_to_chunk:
                title_to_chunk[title] = docs_text[i]
        return title_to_chunk
    else:
        unique_titles = set()
        for meta in metas:
            if "title" in meta:
                unique_titles.add(meta["title"])
        return unique_titles