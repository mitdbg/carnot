"""System #3: Codex Agent

Can be run with tools-only (via MCP server) or with direct context interaction (via files in domain).
"""
import asyncio
import json
import os
import shutil
import time

from collections import deque
from jinja2 import Environment, StrictUndefined
from pathlib import Path
from threading import Lock

from chromadb.api import ClientAPI

from skunk.common import ExecutionContext
from skunk.config import AgentConfig, InferenceConfig

from qatfd.agents.bootstrap_agent import BootstrapAgent
from qatfd.agents.enrich_agent import EnrichAgent
from qatfd.benchmarks.base import BenchmarkResources
from qatfd.config import CodexConfig
from qatfd.constants import METADATA_LIST_DELIMITER
from qatfd.paths import resolve_under_benchmarks
from qatfd.prompts import load_qatfd_prompts
from qatfd.systems.base import System
from qatfd.tools import ListCollectionsTool
from qatfd.types import AnswerOutput, Question

_ENV = Environment(
    autoescape=False, keep_trailing_newline=True, undefined=StrictUndefined
)
_BS_PROMPTS = load_qatfd_prompts("bootstrap_agent")
_EN_PROMPTS = load_qatfd_prompts("enrich_agent")
_SA_PROMPTS = load_qatfd_prompts("search_agent")
_WS_PROMPTS = load_qatfd_prompts("working_set")

# chromadb imposes a limit of 100 collections per-call to list_collections()
_LIST_LIMIT = 100

# example answer
EXAMPLE_ANSWER = """The final answer will need to be a JSON object with the following syntax and semantics:
{
  "answer": "The answer to the provided question.",
  "doc_ids": ["The list of doc_id(s) of the documents which were relevant to answering the question."]
}"""
COMPUTE_OBJECTIVE = """### End-to-End Evaluation
Your final answer quality will be judged as follows: {compute_objective}

You will be judged on final answer quality, execution cost (in dollars), and execution latency (in seconds). The primary goal is to generate high quality answers. The secondary goal is to do so while keeping cost and latency low."""
# SEQUENTIAL_NOTE = """### Sequential Query Execution
# In addition to answering this question, you will be given more questions grounded in this corpus in subsequent requests. Keeping your evaluation metrics in mind, please generate any notes, memories, or intermediate state that will help you to answer future questions."""
SEQUENTIAL_NOTE_WITH_CODEX_SHELL = """\n\n### Sequential Query Execution
In addition to answering this question, you will be given more questions grounded in this corpus in subsequent requests. Keeping your evaluation metrics in mind, please generate any notes or memories that will help you to answer future questions more effectively and efficiently. Your working directory persists across all of these questions, so you may keep notes in files there; anything you write to AGENTS.md is loaded automatically at the start of every future question."""
SEQUENTIAL_NOTE = """\n\n### Sequential Query Execution
In addition to answering this question, you will be given more questions grounded in this corpus in subsequent requests. Keeping your evaluation metrics in mind, please generate any notes or memories that will help you to answer future questions more effectively and efficiently."""

# the only parent-process env vars codex inherits (plus CODEX_HOME / HOME / the codex key / proxy vars set
# explicitly): codex and any shell it runs never see the runner's other secrets (OPENROUTER_API_KEY, the
# OpenRouter management key, AWS credentials / IRSA token paths, chroma / benchmark paths)
_CODEX_ENV_PASSTHROUGH = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TZ", "TMPDIR", "SSL_CERT_FILE", "SSL_CERT_DIR", "RUST_LOG")


def _chown_tree(root: Path, uid: int) -> None:
    """chown `root` and everything under it to uid:uid (the codex sandbox user owns its home and work dirs)."""
    os.chown(root, uid, uid)
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            os.chown(os.path.join(dirpath, name), uid, uid, follow_symlinks=False)


def _event_message(evt: dict) -> str | None:
    err = evt.get("error")
    return evt.get("message") or (err.get("message") if isinstance(err, dict) else err)


def codex_log_path(trace_path: Path) -> Path:
    """`traces/<qid>.codex.jsonl`: codex's raw event stream for the question whose trace is `trace_path`."""
    return trace_path.with_suffix(".codex.jsonl")


class CodexSystem(System):
    """Codex Agent which answers question(s) given tools via MCP-server."""
    name = "codex"

    def __init__(self, codex_config: AgentConfig, inference_cfg: InferenceConfig, run_dir: Path) -> None:
        assert isinstance(codex_config, CodexConfig)
        self.codex_config = codex_config
        self.inference_cfg = inference_cfg

        # lock and state tracking for sequential execution
        self._question_lock = Lock()
        self._question_num = None
        self._question_history: deque[str] = deque(maxlen=max(0, self.codex_config.enrich_config.max_previous_queries))

        # create codex home and scratch directories if they don't exist
        self.codex_home_dir = Path(self.codex_config.codex_home_dir) if self.codex_config.codex_home_dir else run_dir / "codex_home"
        Path(self.codex_home_dir).mkdir(parents=True, exist_ok=True)
        self.codex_scratch_dir = Path(self.codex_config.codex_scratch_dir) if self.codex_config.codex_scratch_dir else run_dir / "codex_scratch"
        Path(self.codex_scratch_dir).mkdir(parents=True, exist_ok=True)
        self.thread_id_path = run_dir / "codex_thread_id"

        # store the thread id associated with this Codex agent once it executes a query;
        # if we are resuming a previous session, read the thread id from disk
        self._thread_id = None
        if self.codex_config.run_mode == "sequential" and self.codex_config.session_resume and self.thread_id_path.exists():
            with open(self.thread_id_path) as f:
                self._thread_id = f.read().strip()

        # copy config.toml into codex home directory
        config_filepath = resolve_under_benchmarks(self.codex_config.codex_config_toml)
        new_config_filepath = self.codex_home_dir / os.path.basename(config_filepath)
        if not new_config_filepath.exists():
            shutil.copy(config_filepath, new_config_filepath)

        # copy AGENTS.md into codex scratch directory
        agents_md_filepath = resolve_under_benchmarks(self.codex_config.codex_agents_md)
        new_agents_md_filepath = self.codex_scratch_dir / os.path.basename(agents_md_filepath)
        if not new_agents_md_filepath.exists():
            shutil.copy(agents_md_filepath, new_agents_md_filepath)

        # codex reads the answer schema itself (--output-schema): give it a copy in its own home, so it never needs
        # read access to the benchmarks dir (which holds the gold answers)
        schema_src = resolve_under_benchmarks(self.codex_config.codex_answer_schema_file)
        self.schema_file = self.codex_home_dir / f"answer_schema{Path(schema_src).suffix or '.json'}"
        if not self.schema_file.exists():
            shutil.copy(schema_src, self.schema_file)

        # sandbox user: it owns its home and work dirs, and nothing else the runner writes
        if self.codex_config.sandbox_uid is not None:
            _chown_tree(Path(self.codex_home_dir), self.codex_config.sandbox_uid)
            _chown_tree(Path(self.codex_scratch_dir), self.codex_config.sandbox_uid)

    def _codex_env(self) -> dict[str, str]:
        """The codex subprocess's whole environment (see _CODEX_ENV_PASSTHROUGH)."""
        env = {k: os.environ[k] for k in _CODEX_ENV_PASSTHROUGH if k in os.environ}
        env["CODEX_HOME"] = str(self.codex_home_dir)
        # the sandbox user has no home of its own; locally keep the invoking user's HOME (codex's previous behavior)
        env["HOME"] = str(self.codex_home_dir) if self.codex_config.sandbox_uid is not None else os.environ.get("HOME", str(self.codex_home_dir))
        # config.toml's provider auth reads this one (`echo $OPENROUTER_CODEX_API_KEY`)
        env["OPENROUTER_CODEX_API_KEY"] = os.environ["OPENROUTER_CODEX_API_KEY"]
        if self.codex_config.egress_proxy:
            for k in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
                env[k] = self.codex_config.egress_proxy
            # the MCP server is on loopback
            env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
        return env

    @property
    def system_usage_key(self) -> str:
        return self.codex_config.agent_id

    @property
    def bootstrap_usage_key(self) -> str:
        return self.codex_config.bootstrap_config.agent_id

    @property
    def enrich_usage_key(self) -> str:
        return self.codex_config.enrich_config.agent_id

    def _create_working_set_summaries(self, chroma_client: ClientAPI, base_collection_name: str):
        """Return a string with summaries of the names, size, metadata fields, and actions for all (non-base) collections."""
        working_set_summaries = ""
        ws_summary_template = _WS_PROMPTS["working_set_summary"]
        offset = 0
        collections = chroma_client.list_collections(limit=_LIST_LIMIT, offset=offset)
        while len(collections) > 0:
            for collection in collections:
                if collection.name == base_collection_name or not collection.metadata.get("is_working_set"):
                    continue

                # get information about this collection
                total_num_chunks = collection.count()
                description = collection.metadata.get("description", "")
                metadata_fields = collection.metadata.get("fields")
                actions = collection.metadata["actions"].split(METADATA_LIST_DELIMITER)

                # summarize this collection
                ws_summary = _ENV.from_string(ws_summary_template).render(
                    name=collection.name,
                    description=description,
                    total_num_chunks=total_num_chunks,
                    metadata_fields=metadata_fields,
                    actions="\n".join(actions),
                )

                # add summary
                working_set_summaries += f"{ws_summary}\n\n"

            # stop once we get an incomplete page
            if len(collections) < _LIST_LIMIT:
                break

            # otherwise, fetch next page
            offset += _LIST_LIMIT
            collections = chroma_client.list_collections(limit=_LIST_LIMIT, offset=offset)

        return working_set_summaries

    @staticmethod
    def _base_collection_summary(ctx: ExecutionContext, resources: BenchmarkResources) -> str:
        """Name / size / metadata fields of the base collection (the Bootstrap agent's whole input)."""
        base_collection_name = ctx.config.storage.collection_name
        c = resources.chroma_client.get_collection(base_collection_name)
        return _ENV.from_string(_BS_PROMPTS["base_collection_summary"]).render(
            base_collection_name=base_collection_name,
            total_num_chunks=c.count(),
            metadata_fields=(c.metadata or {}).get("fields"),
        )

    @staticmethod
    def _collection_summaries(ctx: ExecutionContext, resources: BenchmarkResources) -> str:
        """One `working_set_summary` per agent-created collection (description, size, fields, actions)."""
        base_collection_name = ctx.config.storage.collection_name
        listing = ListCollectionsTool(resources.chroma_client, base_collection_name)()["collections"]
        template = _ENV.from_string(_WS_PROMPTS["working_set_summary"])
        summaries = [
            template.render(
                name=s["name"],
                description=s["description"],
                total_num_chunks=s["num_chunks"],
                metadata_fields=s["fields"],
                actions="\n".join(s["actions"]),
            )
            for s in listing
            if not s["is_base"]
        ]
        return "\n\n".join(summaries)

    def _add_collections_message(self, chroma_client: ClientAPI, base_collection_name: str) -> str:
        """Add a message to the agent's context informing it of the collection(s) at its disposal."""
        # get total chunks and metadata fields for the base collection
        c = chroma_client.get_collection(base_collection_name)
        total_num_chunks = c.count()
        metadata_fields = c.metadata.get("fields")

        collections_summary = ""
        if self.codex_config.enrich_working_sets is not None:
            collections_summary_template = _SA_PROMPTS["collections_summary_with_working_sets"]
            collections_summary = _ENV.from_string(collections_summary_template).render(
                base_collection_name=base_collection_name,
                total_num_chunks=total_num_chunks,
                metadata_fields=metadata_fields,
                working_set_summaries=self._create_working_set_summaries(chroma_client, base_collection_name),
            )
        else:
            collections_summary_template = _SA_PROMPTS["collections_summary_without_working_sets"]
            collections_summary = _ENV.from_string(collections_summary_template).render(
                base_collection_name=base_collection_name,
                total_num_chunks=total_num_chunks,
                metadata_fields=metadata_fields,
            )

        # prepend the collections summary to the input question
        return collections_summary

    async def _run_collection_agent(self, which: str, ctx: ExecutionContext, coro) -> None:
        """Run a Bootstrap / Enrich agent for its side effects on the collections. Its failure (typically
        `StepFailed`: out of steps without a final answer) must not fail the question it happens to run
        before / after: that question's own retrieval and answer are independent of it. Log it, leave a
        trace event, and carry on with whatever collections the agent managed to curate."""
        try:
            await coro
        except Exception as e:  # noqa: BLE001 — any collection-agent failure is non-fatal for the question
            msg = f"{type(e).__name__}: {e}"
            ctx.tracer.emit(id=f"{which}_agent_failed", kind="lifecycle", level="warning", data={"error": msg})

    def _build_bootstrap_agent(self, ctx: ExecutionContext, resources: BenchmarkResources) -> BootstrapAgent:
        self.codex_config: CodexConfig
        additional_notes = None
        if resources.corpus_details:
            additional_notes_template = _BS_PROMPTS["additional_notes"]
            additional_notes = _ENV.from_string(additional_notes_template).render(
                corpus_details=resources.corpus_details,
            )
        return BootstrapAgent(
            config=self.codex_config.bootstrap_config,
            document_map=resources.document_map,
            chroma_client=resources.chroma_client,
            llm_client=ctx.llm_client,
            storage_config=ctx.config.storage,
            additional_notes=additional_notes,
            page_locator=resources.page_locator,
        )

    def _build_enrich_agent(self, ctx: ExecutionContext, resources: BenchmarkResources) -> EnrichAgent:
        self.codex_config: CodexConfig
        additional_notes = None
        if resources.corpus_details:
            additional_notes_template = _EN_PROMPTS["additional_notes"]
            additional_notes = _ENV.from_string(additional_notes_template).render(
                corpus_details=resources.corpus_details,
            )
        return EnrichAgent(
            config=self.codex_config.enrich_config,
            document_map=resources.document_map,
            chroma_client=resources.chroma_client,
            llm_client=ctx.llm_client,
            storage_config=ctx.config.storage,
            additional_notes=additional_notes,
            page_locator=resources.page_locator,
        )

    async def _precompute_working_sets(self, ctx: ExecutionContext, resources: BenchmarkResources):
        # construct the bootstrap agent
        agent = self._build_bootstrap_agent(ctx, resources)

        # have the agent create an initial set of working sets from a description of the base collection
        _ = await agent.call(ctx, self._base_collection_summary(ctx, resources))

    async def _enrich(self, ctx: ExecutionContext, resources: BenchmarkResources, history: list[str]) -> None:
        """Have the EnrichAgent curate the collections given the base collection, the existing
        (agent-created) collections, and the most recent questions handled by this system."""
        # construct the enrich agent
        agent = self._build_enrich_agent(ctx, resources)

        # describe the base collection, the existing collections, and the recent query workload
        enrich_summary = _ENV.from_string(_EN_PROMPTS["enrich_summary"]).render(
            base_collection_summary=self._base_collection_summary(ctx, resources),
            collection_summaries=self._collection_summaries(ctx, resources),
            previous_queries=history,
        )

        # have the agent curate the collections
        _ = await agent.call(ctx, enrich_summary)

    async def run_codex(self, q: Question, resources: BenchmarkResources, ctx: ExecutionContext, analytics_id: str) -> AnswerOutput:
        # set working directory, result file, and schema file
        work_dir = Path(self.codex_scratch_dir) / q.qid if self.codex_config.isolation else Path(self.codex_scratch_dir)
        result_file = str(work_dir / "answer.json") if self.codex_config.isolation else str(work_dir / f"answer-{q.qid}.json")
        schema_file = str(self.schema_file)

        # ensure the working directory is created before processing
        work_dir.mkdir(parents=True, exist_ok=True)
        if self.codex_config.sandbox_uid is not None:
            os.chown(work_dir, self.codex_config.sandbox_uid, self.codex_config.sandbox_uid)

        # prepare the query
        t0 = time.monotonic()
        question_text = f"## {q.text}" if q.text.startswith("Question: ") else f"## Question: {q.text}"
        prompt = f"{question_text}\n\n### Answer Formatting and Hints\n{EXAMPLE_ANSWER}"
        if resources.answer_format_hint:
            prompt += f"\n\n{resources.answer_format_hint}"
        if resources.compute_objective:
            compute_objective = COMPUTE_OBJECTIVE.format(compute_objective=resources.compute_objective)
            prompt += f"\n\n{compute_objective}"
        if resources.corpus_details:
            prompt += f"\n\n### Corpus Details\n{resources.corpus_details}"
        if not self.codex_config.isolation:
            if self.codex_config.codex_shell:
                prompt += SEQUENTIAL_NOTE_WITH_CODEX_SHELL
            else:
                prompt += SEQUENTIAL_NOTE
        prompt += f"\n\n{self._add_collections_message(resources.chroma_client, ctx.config.storage.collection_name)}\n\n"

        # construct the codex command based on the run mode
        resume = not self.codex_config.isolation and self.codex_config.session_resume and self._thread_id is not None
        cmd = ["codex", "exec", "resume", self._thread_id] if resume else ["codex", "exec"]
        if self.codex_config.isolation:
            cmd.append("--ephemeral")

        work_dir_args = ["-C", str(work_dir)] if not resume else []
        auto_compaction_args = (
            ["-c", f"model_auto_compact_token_limit={self.codex_config.auto_compact_token_limit}"]
            if self.codex_config.auto_compact_token_limit is not None
            else []
        )
        shell_access_args = (
            ["--enable", "shell_tool", "-c", f"sandbox_mode={json.dumps(self.codex_config.shell_sandbox_mode)}"]
            if self.codex_config.codex_shell
            else []
        )
        cmd += [
            "--json", "--skip-git-repo-check",
            # NOTE: currently, memories can only be generated after a session is idle for 1 hour
            # thus, we disable memories and enable note taking via shell tooling
            "--disable", "memories",
            *work_dir_args,
            *auto_compaction_args,
            *shell_access_args,
            "-c", f"mcp_servers.corpus.url={json.dumps(self.codex_config.mcp_url)}",
            "-c", f"mcp_servers.corpus.enabled_tools={json.dumps(self.codex_config.enabled_tools)}",
            "-c", f'model_providers.openrouter.http_headers={{"x-session-id"="{analytics_id}"}}',
            "-c", f'mcp_servers.corpus.http_headers={{"x-session-id"="{analytics_id}"}}',
            "--output-schema", schema_file, "-o", result_file, prompt,
        ]

        # run the codex agent
        uid = self.codex_config.sandbox_uid
        user_kwargs = {"user": uid, "group": uid, "extra_groups": []} if uid is not None else {}
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._codex_env(),
            # `codex exec resume` takes no -C: without an explicit cwd every resumed question ran in the runner's cwd
            # (the qatfd checkout, a relative path away from the gold answers) instead of its workspace
            cwd=str(work_dir),
            **user_kwargs,
        )
        stdout, stderr = await process.communicate()
        wall_s = time.monotonic() - t0

        # gather all events from stdout
        events = []
        for line in stdout.splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        # set thread_id for subsequent queries if this is the first sequential question
        if self.codex_config.run_mode == "sequential" and self._thread_id is None:
            self._thread_id = next((evt["thread_id"] for evt in events if evt.get("type") == "thread.started"), None)

        # persist the thread_id if it's set
        if self._thread_id and not self.thread_id_path.exists():
            with open(self.thread_id_path, "w") as f:
                f.write(self._thread_id)

        # dump codex's raw `--json` stdout next to the question's trace, NOT into it: the Tracer holds
        # `log_path` open and keeps writing (e.g. the nugget judge's event after this returns), so
        # rewriting that file here would leave the two interleaved at stale offsets.
        codex_log_path(ctx.tracer.log_path).write_bytes(stdout)

        error = None
        if process.returncode != 0 or not Path(result_file).exists():
            detail = next(
                (_event_message(evt) for evt in reversed(events) if evt.get("type") in ("error", "turn.failed")),
                None,
            ) or stderr.decode(errors="replace").strip()
            error = f"codex exited {process.returncode}: {detail}"

        # read answer from result_file
        answer = {"answer": "", "doc_ids": []}
        if error is None:
            with open(result_file) as f:
                answer = json.load(f)

        # parse answer and return
        return AnswerOutput(
            answer=answer["answer"],
            retrieved_doc_ids=answer["doc_ids"],
            terminate_state="finished" if error is None else "error",
            compute_wall_s=wall_s,
            error=error,
        )

    async def answer(self, q: Question, resources: BenchmarkResources, ctx: ExecutionContext, analytics_id: str) -> AnswerOutput:
        t0 = time.monotonic()
        mode = self.codex_config.enrich_working_sets
        with self._question_lock:
            first_question = self._question_num is None
            if first_question:
                self._question_num = 0
            if mode in ("before", "both") and first_question:
                await self._run_collection_agent("bootstrap", ctx, self._precompute_working_sets(ctx, resources))
        t1 = time.monotonic()

        try:
            answer_output = await self.run_codex(q, resources, ctx, analytics_id)
        except Exception as e:
            answer_output = AnswerOutput("", error=str(e))

        t2 = time.monotonic()
        with self._question_lock:
            assert isinstance(self._question_num, int)
            self._question_num += 1
            self._question_history.append(q.text)

            if mode in ("after", "both"):
                assert self.codex_config.enrich_query_batch_size is not None
                if self._question_num % self.codex_config.enrich_query_batch_size == 0:
                    await self._run_collection_agent("enrich", ctx, self._enrich(ctx, resources, list(self._question_history)))
        t3 = time.monotonic()

        # update answer_output with timing info
        answer_output.precompute_wall_s = t1 - t0 if mode in ("before", "both") else 0.0
        answer_output.enrich_wall_s = t3 - t2 if mode in ("after", "both") else 0.0

        return answer_output