"""System #3: Codex Agent

Can be run with tools-only (via MCP server) or with direct context interaction (via files in domain).
"""
import asyncio
import json
import os
import shutil
import time

from pathlib import Path

from skunk.common import ExecutionContext
from skunk.config import AgentConfig, InferenceConfig

from qatfd.benchmarks.base import BenchmarkResources
from qatfd.config import CodexConfig
from qatfd.paths import resolve_under_benchmarks
from qatfd.systems.base import System
from qatfd.types import AnswerOutput, Question

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
SEQUENTIAL_NOTE_WITH_CODEX_SHELL = """### Sequential Query Execution
In addition to answering this question, you will be given more questions grounded in this corpus in subsequent requests. Keeping your evaluation metrics in mind, please generate any notes or memories that will help you to answer future questions more effectively and efficiently. Your working directory persists across all of these questions, so you may keep notes in files there; anything you write to AGENTS.md is loaded automatically at the start of every future question."""
SEQUENTIAL_NOTE = """### Sequential Query Execution
In addition to answering this question, you will be given more questions grounded in this corpus in subsequent requests. Keeping your evaluation metrics in mind, please generate any notes or memories that will help you to answer future questions more effectively and efficiently."""

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

        # create codex home and scratch directories if they don't exist
        self.codex_home_dir = run_dir / "codex_home"
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

    @property
    def system_usage_key(self) -> str:
        return self.codex_config.agent_id

    async def answer(self, q: Question, resources: BenchmarkResources, ctx: ExecutionContext, analytics_id: str) -> AnswerOutput:
        # set working directory, result file, and schema file
        work_dir = Path(self.codex_scratch_dir) / q.qid if self.codex_config.run_mode == "parallel" else Path(self.codex_scratch_dir)
        result_file = str(work_dir / "answer.json") if self.codex_config.run_mode == "parallel" else str(work_dir / f"answer-{q.qid}.json")
        schema_file = str(resolve_under_benchmarks(self.codex_config.codex_answer_schema_file))

        # ensure the working directory is created before processing
        work_dir.mkdir(parents=True, exist_ok=True)

        # prepare the query
        t0 = time.monotonic()
        prompt = f"## Question: {q.text}\n\n### Answer Formatting and Hints\n{EXAMPLE_ANSWER}"
        if resources.answer_format_hint:
            prompt += f"\n\n{resources.answer_format_hint}"
        if resources.compute_objective:
            compute_objective = COMPUTE_OBJECTIVE.format(compute_objective=resources.compute_objective)
            prompt += f"\n\n{compute_objective}"
        if resources.corpus_details:
            prompt += f"\n\n### Corpus Details\n{resources.corpus_details}"
        if self.codex_config.run_mode == "sequential":
            if self.codex_config.codex_shell:
                prompt += SEQUENTIAL_NOTE_WITH_CODEX_SHELL
            else:
                prompt += SEQUENTIAL_NOTE

        # construct the codex command based on the run mode
        resume = self.codex_config.run_mode == "sequential" and self.codex_config.session_resume and self._thread_id is not None
        cmd = ["codex", "exec", "resume", self._thread_id] if resume else ["codex", "exec"]
        if self.codex_config.run_mode == "parallel":
            cmd.append("--ephemeral")

        work_dir_args = ["-C", str(work_dir)] if not resume else []
        auto_compaction_args = (
            ["-c", f"model_auto_compact_token_limit={self.codex_config.auto_compact_token_limit}"]
            if self.codex_config.auto_compact_token_limit is not None
            else []
        )
        shell_access_args = (
            ["--enable", "shell_tool", "-c", 'sandbox_mode="workspace-write"']
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
        env = {**os.environ, "CODEX_HOME": str(self.codex_home_dir)}
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
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
