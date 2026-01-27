from __future__ import annotations

import io
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
from cloudpathlib import S3Path
from pydantic import BaseModel
from pypdf import PdfReader
from smolagents import CodeAgent, LiteLLMModel

from carnot.core.data import context_manager
from carnot.core.data.dataset import Dataset
from carnot.core.lib.schemas import create_schema_from_fields, union_schemas
from carnot.operators.logical import ComputeOperator, ContextScan, LogicalOperator, SearchOperator
from carnot.utils.hash_helpers import hash_for_id

IS_LOCAL_ENV = os.getenv("LOCAL_ENV").lower() == "true"
FILESYSTEM = "file" if IS_LOCAL_ENV else "s3"
COMPANY_ENV = os.getenv("COMPANY_ENV", "dev")
BACKEND_ROOT = "/code/"
BASE_DIR = f"s3://carnot-research-{COMPANY_ENV}/"
DATA_DIR = f"s3://carnot-research-{COMPANY_ENV}/data/"
SHARED_DATA_DIR = f"s3://carnot-research-{COMPANY_ENV}/shared/"
SKIP_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".zip", ".exe", ".bin"}

def get_text_from_pdf(pdf_bytes):
    pdf = PdfReader(io.BytesIO(pdf_bytes))
    all_text = ""
    for page in pdf.pages:
        all_text += page.extract_text() + "\n"
    return all_text


PZ_INSTRUCTION = """\n\nYou are a CodeAgent who is a specialist at writing declarative AI programs with the Palimpzest (PZ) library.

Palimpzest is a programming framework which provides you with **semantic operators** (e.g. semantic maps, semantic filters, etc.)
which are like their traditional counterparts, except they can execute instructions provided in natural language.

For example, if you wanted to write a program to extract the title and abstract from a directory of papers,
you could write the following in PZ:
```
import palimpzest as pz
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define columns for semantic map (sem_map) operation; each column is specified
# with a dictionary containing the following keys:
# - "name": the name of the field to compute
# - "type": the type of the field to compute
# - "description": the natural language description of the field
paper_cols = [
    {"name": "title", "type": str, "description": "the title of the paper"},
    {"name": "abstract", "type": str, "description": "the paper's abstract"},
]

# construct the data processing pipeline with PZ
ds = pz.TextFileDataset(id="papers", path="path/to/papers")
ds = ds.sem_map(cols)

# optimize and execute the PZ program
validator = pz.Validator()
config = pz.QueryProcessorConfig(
    policy=pz.MaxQuality(),
    execution_strategy="parallel",
    available_models=[pz.Model.GPT_5, pz.Model.GPT_5_MINI],
    max_workers=20,
    progress=True,
)
output = ds.run(config=config, validator=validator)

# write the execution stats to json
output.execution_stats.to_json("pz_program_stats.json")

# write the output to a CSV and print the output CSV filepath so the user knows where to find it
output_filepath = "pz_program_output.csv"
output.to_df().to_csv(output_filepath, index=False)
print(f"Results at: {output_filepath}")
```

To initialize a dataset in PZ, simply provide the path to a directory to `pz.TextFileDirectory()`
(if your data contains text-based files). For example:
```
import palimpzest as pz
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

ds = pz.TextFileDataset(id="files", path="path/to/files")
```

Palimpzest has two primary **semantic operators** which you can use to construct data processing pipelines:
- sem_filter(predicate: str): executes a semantic filter specified by the natural language predicate on a given PZ dataset
- sem_map(cols: list[dict]): executes a semantic map to compute the `cols` on a given PZ dataset

As a second example, consider the following PZ program which filters for papers about batteries that are from MIT
and computes a summary for each one:
```
import palimpzest as pz
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# construct the PZ program
ds = pz.TextFileDataset(id="papers", path="path/to/research-papers")
ds = ds.sem_filter("The paper is about batteries")
ds = ds.sem_filter("The paper is from MIT")
ds = ds.sem_map([{"name": "summary", "type": str, "description": "A summary of the paper"}])

# optimize and execute the PZ program
validator = pz.Validator()
config = pz.QueryProcessorConfig(
    policy=pz.MaxQuality(),
    execution_strategy="parallel",
    available_models=[pz.Model.GPT_5, pz.Model.GPT_5_MINI],
    max_workers=20,
    progress=True,
)
output = ds.run(config=config, validator=validator)

# write the execution stats to json
output.execution_stats.to_json("pz_program_stats.json")

# write the output to a CSV and print the output CSV filepath so the user knows where to find it
output_filepath = "pz_program_output.csv"
output.to_df().to_csv(output_filepath, index=False)
print(f"Results at: {output_filepath}")
```

Be sure to always:
- execute your program using the `.run()` format shown above
- call `output.execution_stats.to_json("pz_program_stats.json")` to write execution statistics to disk
- write your output to CSV and print where you wrote it!
"""

class BaseFileService(ABC):
    """Abstract base class for file services"""

    @abstractmethod
    def exists(self, path: str) -> bool:
        pass

    @abstractmethod
    def is_dir(self, path: str) -> bool:
        pass

    @abstractmethod
    def list_all_subfiles(self, path: str) -> list[str]:
        pass

    @abstractmethod
    def read_file(self, path: str, bytes: bool = False) -> str:
        pass

class LocalFileService(BaseFileService):
    """File service for local filesystem"""
    def exists(self, path: str) -> bool:
        return os.path.exists(path)

    def is_dir(self, path: str) -> bool:
        return os.path.isdir(path)

    def list_all_subfiles(self, path: str) -> list[str]:
        """
        NOTE: this method assumes path is an absolute path; without this assumption file_paths
        will not have correct absolute paths.
        """
        file_paths = []
        for root, _, files in os.walk(path):
            for file in files:
                file_paths.append(os.path.join(root, file))
        return file_paths

    def read_file(self, path: str, bytes: bool = False) -> str:
        read_kwargs = {"mode": "rb"} if bytes else {"encoding": "utf-8"}
        with open(path, **read_kwargs) as file:
            content = file.read()
        return content

class S3FileService(BaseFileService):
    """File service for AWS S3"""
    def __init__(self):
        self.s3_bucket = DATA_DIR.replace("s3://", "").split("/")[0]

    def _get_s3_key_from_path(self, path: str) -> str:
        return "/".join(path.replace("s3://", "").split("/")[1:])

    def exists(self, path: str) -> bool:
        """Check if a file or directory exists"""
        s3 = boto3.client('s3')
        s3_prefix = self._get_s3_key_from_path(path)
        response = s3.list_objects_v2(Bucket=self.s3_bucket, Prefix=s3_prefix, MaxKeys=1)
        return 'Contents' in response

    def is_dir(self, path: str) -> bool:
        """Check if a path is a directory; S3 prefixes are always treated as directories"""
        return True

    def list_all_subfiles(self, path: str) -> list[str]:
        """List all files under the given s3 prefix"""
        s3 = boto3.client('s3')

        file_paths = []
        prefix = self._get_s3_key_from_path(path)
        paginator = s3.get_paginator('list_objects_v2')
        result_iterator = paginator.paginate(Bucket=self.s3_bucket, Prefix=prefix)

        for page in result_iterator:
            for obj in page.get('Contents', []):
                file_paths.append(f"s3://{self.s3_bucket}/{obj['Key']}")

        return file_paths

    def read_file(self, path: str, bytes: bool = False) -> str:
        """Read the contents of a file from s3"""
        s3 = boto3.client('s3')
        s3_key = self._get_s3_key_from_path(path)
        response = s3.get_object(Bucket=self.s3_bucket, Key=s3_key)
        content = response['Body'].read()
        return content if bytes else content.decode('utf-8')


class Context(Dataset, ABC):
    """
    The `Context` class is an abstract base class for root `Datasets` whose data is accessed
    via user-defined methods. Classes which inherit from this class must implement two methods:

    - `list_filepaths()`: which lists the files that the `Context` has access to.
    - `read_filepath(path: str)`: which reads the file corresponding to the given `path`.

    A `Context` is a special type of `Dataset` that represents a view over an underlying `Dataset`.
    Each `Context` has a `name` which uniquely identifies it, as well as a natural language `description`
    of the data / computation that the `Context` represents. Similar to `Dataset`s, `Context`s can be
    lazily transformed using functions such as `sem_filter`, `sem_map`, `sem_join`, etc., and they may
    be materialized or unmaterialized.
    """

    def __init__(
            self,
            id: str,
            description: str,
            operator: LogicalOperator,
            schema: type[BaseModel] | None = None,
            sources: list[Context] | Context | None = None,
            materialized: bool = False,
            llm_config: dict | None = None,
        ) -> None:
        """
        Constructor for the `Context` class.

        Args:
            id (`str`): a string identifier for the `Context`
            description (`str`): the description of the data contained within the `Context`
            operator (`LogicalOperator`): The `LogicalOperator` used to compute this `Context`.
            schema: (`type[BaseModel] | None`): The schema of this `Context`.
            sources (`list[Context] | Context | None`): The (list of) `Context(s)` which are input(s) to
                the operator used to compute this `Context`.
            materialized (`bool`): True if the `Context` has been computed, False otherwise
        """
        # if there are any API keys specified in the llm_config, set them in the config
        if llm_config is not None and len(llm_config.keys()) > 0:
            for key, value in llm_config.items():
                os.environ[key] = value

        # set the description
        self._description = description

        # set the materialization status
        self._materialized = materialized

        # compute schema and call parent constructor
        if schema is None:
            schema = create_schema_from_fields([{"name": "context", "description": "The context", "type": str}])
        super().__init__(sources=sources, operator=operator, schema=schema, id=id)

        # set the tools associated with this Context
        self._tools = [getattr(self, attr) for attr in dir(self) if attr.startswith("tool_")]

        # add Context to ContextManager
        cm = context_manager.ContextManager()
        cm.add_context(self)

    @property
    def description(self) -> str:
        """The string containing all of the information computed for this `Context`"""
        return self._description

    @property
    def materialized(self) -> bool:
        """The boolean which specifies whether the `Context` has been computed or not"""
        return self._materialized

    @property
    def tools(self) -> list[Callable]:
        """The list of tools associated with this `Context`"""
        return self._tools

    def __str__(self) -> str:
        return f"Context(id={self.id}, description={self.description:20s}, materialized={self.materialized})"

    def set_description(self, description: str) -> None:
        """
        Update the context's description.
        """
        self._description = description

    def set_materialized(self, materialized: str) -> None:
        """
        Update the context's materialization status.
        """
        self._materialized = materialized

    def compute(self, instruction: str) -> Context:
        # construct new description and output schema
        new_id = hash_for_id(instruction)
        new_description = f"Parent Context ID: {self.id}\n\nThis Context is the result of computing the following instruction on the parent context.\n\nINSTRUCTION: {instruction}\n\n"
        inter_schema = create_schema_from_fields([{"name": f"result-{new_id}", "desc": "The result from computing the instruction on the input Context",  "type": str | Any}])
        new_output_schema = union_schemas([self.schema, inter_schema])

        # construct logical operator
        operator = ComputeOperator(
            input_schema=self.schema,
            output_schema=new_output_schema,
            context_id=new_id,
            instruction=instruction,
        )        

        return Context(id=new_id, description=new_description, operator=operator, sources=[self], materialized=False)

    def search(self, search_query: str) -> Context:
        # construct new description and output schema
        new_id = hash_for_id(search_query)
        new_description = f"Parent Context ID: {self.id}\n\nThis Context is the result of searching the parent context for information related to the following query.\n\nSEARCH QUERY: {search_query}\n\n"

        # construct logical operator
        operator = SearchOperator(
            input_schema=self.schema,
            output_schema=self.schema,
            context_id=new_id,
            search_query=search_query,
        )

        return Context(id=new_id, description=new_description, operator=operator, sources=[self], materialized=False)

class TextFileContext(Context):
    def __init__(self, paths: str | list[str], id: str, description: str, **kwargs) -> None:
        """
        Constructor for the `TextFileContext` class.

        Args:
            paths (str | list[str]): (list of) path(s) to text files / directories to include in the `Context`
            id (str): a string identifier for the `Context`
            description (str): the description of the data contained within the `Context`
            kwargs (dict): keyword arguments containing the `Context's` id and description.
        """
        if isinstance(paths, str):
            paths = [paths]

        self.file_service = S3FileService() if FILESYSTEM == "s3" else LocalFileService()
        self.filepaths = []
        for path in paths:
            if self.file_service.is_dir(path):
                self.filepaths.extend([
                    fp
                    for fp in self.file_service.list_all_subfiles(path)
                    if not any(fp.lower().endswith(suffix) for suffix in SKIP_SUFFIXES)
                ])
            else:
                self.filepaths.append(path)

        # call parent constructor to set id, operator, and schema
        schema = create_schema_from_fields([{"name": "context", "desc": "The context", "type": str | Any}])
        super().__init__(
            id=id,
            description=description,
            operator=ContextScan(context=self, output_schema=schema),
            schema=schema,
            materialized=True,
            **kwargs,
        )

    def _check_filter_answer_text(self, answer_text: str) -> dict | None:
        """
        Return {"passed_operator": True} if and only if "true" is in the answer text.
        Return {"passed_operator": False} if and only if "false" is in the answer text.
        Otherwise, return None.
        """
        # NOTE: we may be able to eliminate this condition by specifying this JSON output in the prompt;
        # however, that would also need to coincide with a change to allow the parse_answer_fn to set "passed_operator"
        if "true" in answer_text.lower():
            return {"passed_operator": True}
        elif "false" in answer_text.lower():
            return {"passed_operator": False}
        elif "yes" in answer_text.lower():
            return {"passed_operator": True}

        return None

    def _parse_filter_answer(self, completion_text: str) -> dict[str, list]:
        """Extract the answer from the completion object for filter operations."""
        # if the model followed the default instructions, the completion text will place
        # its answer between "ANSWER:" and "---"
        regex = re.compile("answer:(.*?)---", re.IGNORECASE | re.DOTALL)
        matches = regex.findall(completion_text)
        if len(matches) > 0:
            answer_text = matches[0].strip()
            field_answers = self._check_filter_answer_text(answer_text)
            if field_answers is not None:
                return field_answers

        # if the first regex didn't find an answer, try taking all the text after "ANSWER:"
        regex = re.compile("answer:(.*)", re.IGNORECASE | re.DOTALL)
        matches = regex.findall(completion_text)
        if len(matches) > 0:
            answer_text = matches[0].strip()
            field_answers = self._check_filter_answer_text(answer_text)
            if field_answers is not None:
                return field_answers

        # finally, try taking all of the text; throw an exception if this doesn't work
        field_answers = self._check_filter_answer_text(completion_text)
        if field_answers is None:
            raise Exception(f"Could not parse answer from completion text: {completion_text}")

        return field_answers

    # def tool_list_filepaths(self) -> list[str]:
    #     """
    #     This tool returns the list of all of the filepaths which the `Context` has access to.

    #     Args:
    #         None
        
    #     Returns:
    #         list[str]: A list of file paths for all files in the `Context`.
    #     """
    #     return self.filepaths

    # def tool_read_filepath(self, path: str) -> str:
    #     """
    #     This tool takes a filepath (`path`) as input and returns the content of the file as a string.
    #     It handles both CSV files and html / regular text files. It does not handle images.

    #     Args:
    #         path (str): The path to the file to read.

    #     Returns:
    #         str: The content of the file as a string.
    #     """
    #     if path.endswith(".csv"):
    #         return pd.read_csv(path, encoding="ISO-8859-1").to_string(index=False)

    #     with open(path, encoding='utf-8') as file:
    #         content = file.read()

    #     return content

    def tool_execute_semantic_operators(self, instruction: str) -> str:
        """
        This tool takes an `instruction` as input and invokes an expert to write a semantic data processing pipeline
        to execute the instruction. The tool returns the path to a CSV file which contains the output of the pipeline.

        For example, the tool could be invoked as follows to extract the title and abstract from a dataset of research papers:
        ```
        instruction = "Write a program to extract the title and abstract from each research paper"
        result_csv_filepath = tool_execute_semantic_operators(instruction)
        ```

        Args:
            instruction: The instruction specifying the semantic data processing pipeline that you need to execute.

        Returns:
            str: the filepath to the CSV containing the output from running the data processing pipeline.
        """
        from smolagents import tool
        @tool
        def tool_list_filepaths() -> list[str]:
            """
            This tool returns the list of all of the filepaths which the `Context` has access to.

            NOTE: You may want to execute this before writing your PZ program to determine where the data lives.

            Args:
                None
            
            Returns:
                list[str]: A list of file paths for all files in the `Context`.
            """
            return self.filepaths

        @tool
        def tool_read_filepath(path: str) -> str:
            """
            This tool takes a filepath (`path`) as input and returns the content of the file as a string.
            It handles both CSV files and html / regular text files as well as files in S3. It does not
            handle images.

            Args:
                path (str): The path to the file to read.

            Returns:
                str: The content of the file as a string.
            """
            if path.endswith(".csv"):
                return pd.read_csv(path, encoding="ISO-8859-1").to_string(index=False)
            
            file_service = S3FileService() if FILESYSTEM == "s3" else LocalFileService()

            if path.endswith(".pdf"):
                pdf_bytes = file_service.read_file(path, bytes=True)
                return get_text_from_pdf(pdf_bytes)

            return file_service.read_file(path)

        agent = CodeAgent(
            model=LiteLLMModel(model_id="openai/o1", api_key=os.getenv("OPENAI_API_KEY")),
            tools=[tool_list_filepaths, tool_read_filepath],
            max_steps=20,
            planning_interval=4,
            add_base_tools=False,
            return_full_result=True,
            additional_authorized_imports=["dotenv", "json", "palimpzest", "io", "os", "re", "pandas"],
            instructions=PZ_INSTRUCTION,
        )
        result = agent.run(instruction)
        response = result.output

        return response
