import os, shlex
from qatfd.env import load_env
load_env()
from hydra import compose, initialize_config_dir
from qatfd.config import benchmark_config_factory
from qatfd.registry import build_benchmark
from skunk.llm_client import LLMClient
from skunk.agents.search_agent.search_tools import (
    SearchCorpusTool, GrepCorpusTool, ReadDocumentTool,
)
from skunk.search_state.working_set import WorkingSet
from fastmcp import FastMCP

SERVER_NAME = "CorpusSearchServer"
SERVER_INSTRUCTIONS = "This server provides access to a corpus of documents for search, grep, and read operations. Use the provided tools to interact with the corpus."

SEARCH_CORPUS_DESC = """This tool performs a vector search over the corpus by embedding the input `query` and returning the `top_k` most relevant chunks, each labelled with its `chunk_id` and `doc_id`. You can optionally restrict the search to a subset of the corpus by passing a `metadata_filter`, which is a ChromaDB-style where clause over chunk metadata.

Supported `metadata_filter` syntax:
- Equality: `{"field": value}`
- Set membership: `{"field": {"$in": [v1, v2, ...]}}`
- Negation / not-in: `{"field": {"$nin": [...]}}`
- Compound: `{"$and": [clause1, clause2, ...]}` or `{"$or": [...]}`

```python
# find the 100 chunks most relevant to "topic X" anywhere in the corpus
search_corpus("topic X", top_k=100)
```

```python
# find the 50 chunks most relevant to "topic Y" within a filtered subset
search_corpus("topic Y", top_k=50, metadata_filter={"$and": [{"field_a": "value_a"}, {"field_b": {"$in": [1, 2, 3]}}]})
```

Args:
  query: the search query
  top_k: the number of chunks most relevant to the `query` to return
  metadata_filter: an optional metadata filter for hybrid vector search

Returns:
  A dictionary with the keys "chunks" and "error". "chunks" contains a list of dictionaries (one per chunk) with keys:
  - "chunk_id": the unique id of the chunk
  - "doc_id": the id of the document the chunk came from
  - "type": the type of the chunk (e.g. "text", "table", "header", etc.)
  - "est_num_tokens": the estimated number of tokens contained within the chunk
  - "distance": the distance of the chunk's embedding to the search query embedding (by some metric)
  - "text": the text of the chunk (includes the other field values in a prefixed header)
  
  "error" contains any error message (None on successful tool calls).
"""

GREP_CORPUS_DESC = """This tool performs a regex search over the corpus and returns the matching chunks grouped by their `doc_id`. Each hit includes its `chunk_id` so you can later refer to it or prune it. The output is unlimited by default (`limit=None` and `max_output_tokens=None`). This is useful for exhaustive queries -- "find every doc that mentions X" -- but you should pass `limit=N` or `max_output_tokens=M` for narrower exploratory searches. If the matches exceed the `max_output_tokens` cap, the result is truncated with a note telling you how many hits were omitted. You can optionally restrict the search to a subset of the corpus by passing a `metadata_filter`, which is a ChromaDB-style where clause over chunk metadata.

Supported `metadata_filter` syntax:
- Equality: `{"field": value}`
- Set membership: `{"field": {"$in": [v1, v2, ...]}}`
- Negation / not-in: `{"field": {"$nin": [...]}}`
- Compound: `{"$and": [clause1, clause2, ...]}` or `{"$or": [...]}`

Note that `max_output_tokens` will limit the number of tokens read, but not the number of tokens fetched. `limit` will limit the number of tokens read and fetched.

```python
# find every chunk that mentions "topic X" (case insensitive)
grep_corpus("(?i)topic X")
```

```python
# find chunks matching the literal phrase "topic Y" within a filtered subset, capped at 50 hits
grep_corpus("topic Y", metadata_filter={"field_a": {"$in": ["value_1", "value_2"]}}, limit=50)
```

```python
# broad sweep, but keep the observation small: cap the output at ~5k tokens
grep_corpus("(?i)topic Z", max_output_tokens=5000)
```

Args:
  pattern: the pattern for the grep search
  metadata_filter: an optional metadata filter for hybrid vector search
  limit: an optional limit on the number of chunks to return
  max_output_tokens: an optional limit on the number of output tokens in the returned chunks

Returns:
  A dictionary with the keys "chunks", "error", and "truncation_note". "chunks" contains a list of dictionaries (one per chunk) with keys:
  - "chunk_id": the unique id of the chunk
  - "doc_id": the id of the document the chunk came from
  - "text": the text of the chunk

  "error" contains any error message (None on successful tool calls). "truncation_note" provides info on how many chunks were omitted (None if all matching chunks were returned).
"""

READ_DOCUMENT_DESC = """This tool returns the cleaned text of one or more documents, given their `doc_id`(s). You can optionally specify `start_char_idx` and `end_char_idx` to fetch only a substring of each document. If you pass a list for `doc_id`, `start_char_idx` and `end_char_idx` must be lists of the same length or `None`, and each index pair will be applied to the corresponding document. If `end_char_idx` exceeds the document length, the document will be truncated to its actual length. If `start_char_idx` is negative, it counts from the end of the document. If `end_char_idx` is negative, it counts from the end of the document.

```python
# read two specific documents by id
read_document(["doc_id_1", "doc_id_2"])
```

```python
# read the first 1000 characters of a document
read_document("doc_id_3", start_char_idx=0, end_char_idx=1000)
```

```python
# read the last 1000 characters of a document and the full text of another document
read_document(["doc_id_4", "doc_id_5"], start_char_idx=[-1000, None])
```

Args:
  doc_id: the id(s) of one or more documents to read
  start_char_idx: the index (or indices) at which to start reading each document
  end_char_idx: the index (or indices) at which to finish reading each document

Returns:
  A dictionary with a single key "docs", this contains a list of dictionaries (one per document) with keys:
  - "doc_id": the id of the document
  - "text": the text of the document
"""

config_dir = os.environ.get("QATFD_CONFIG_DIR")
if config_dir is None:
    raise Exception("Need to set QATFD_CONFIG_DIR in env")
overrides = shlex.split(os.environ.get("QATFD_OVERRIDES", "benchmarks=officeqa"))
with initialize_config_dir(config_dir=config_dir, version_base=None):
    cfg = compose(config_name="config", overrides=overrides)
bench = build_benchmark(benchmark_config_factory(cfg))
res = bench.get_resources()
llm = LLMClient(cfg.inference, openrouter_api_key=os.environ["OPENROUTER_CODEX_API_KEY"])
ws = WorkingSet(collection=res.chroma_collection)

# construct the skunk tools to be placed in the MCP server
search = SearchCorpusTool(res.chroma_collection, llm, ws, working_set_collection_off=True, id_tracking_off=True)
grep = GrepCorpusTool(res.chroma_collection, ws, working_set_collection_off=True, id_tracking_off=True)
read = ReadDocumentTool(res.document_map, ws, id_tracking_off=True)

# construct the MCP server; wrap the tools to hide WorkingSet interface details
mcp = FastMCP(SERVER_NAME, instructions=SERVER_INSTRUCTIONS)

@mcp.tool(name="search_corpus", description=SEARCH_CORPUS_DESC, annotations={"readOnlyHint": True})
def search_corpus(query: str, top_k: int, metadata_filter: dict | None = None) -> dict:
    result_dict = search(query=query, top_k=top_k, metadata_filter=metadata_filter)
    return {"chunks": result_dict["read_chunks"], "error": result_dict.get("error", None)}

@mcp.tool(name="grep_corpus", description=GREP_CORPUS_DESC, annotations={"readOnlyHint": True})
def grep_corpus(pattern: str, metadata_filter: dict | None = None, limit: int | None = None, max_output_tokens: int | None = None) -> dict:
    result_dict = grep(pattern=pattern, metadata_filter=metadata_filter, limit=limit, max_output_tokens=max_output_tokens)
    final_chunks = []
    for group in result_dict["read_groups"]:
        for chunk in group["chunks"]:
            final_chunks.append({"chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"], "text": chunk["text"]})
    return {"chunks": final_chunks, "error": result_dict.get("error", None), "truncation_note": result_dict.get("truncation_note", None)}

@mcp.tool(name="read_document", description=READ_DOCUMENT_DESC, annotations={"readOnlyHint": True})
def read_document(doc_id: str | list[str], start_char_idx: int | None | list[int | None] = None, end_char_idx: int | None | list[int | None] = None) -> dict:
    result_dict = read(doc_id=doc_id, start_char_idx=start_char_idx, end_char_idx=end_char_idx)
    return {"docs": result_dict["docs"]}

if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=8765)
