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
# from skunk.llm_client import LLMResponse

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

class CodexSystem(System):
    """Codex Agent which answers question(s) given tools via MCP-server."""
    name = "codex"

    def __init__(self, codex_config: AgentConfig, inference_cfg: InferenceConfig) -> None:
        assert isinstance(codex_config, CodexConfig)
        self.codex_config = codex_config
        self.inference_cfg = inference_cfg

        # create codex scratch directory if it doesn't exist
        self.scratch_dir = resolve_under_benchmarks(self.codex_config.codex_scratch_dir)
        Path(self.scratch_dir).mkdir(parents=True, exist_ok=True)

    @property
    def system_usage_key(self) -> str:
        return self.codex_config.agent_id

    async def answer(self, q: Question, resources: BenchmarkResources, ctx: ExecutionContext) -> AnswerOutput:
        # clear and create subdirectory for this question
        work_dir = Path(self.scratch_dir) / q.qid
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(exist_ok=True)
        result_file = f"{work_dir}/answer.json"

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

        # NOTE: for now this is only geared towards isolation experiments
        # run the codex agent
        env = {**os.environ, "CODEX_HOME": str(resolve_under_benchmarks(self.codex_config.codex_home))}
        schema_file = str(resolve_under_benchmarks(self.codex_config.codex_answer_schema_file))
        process = await asyncio.create_subprocess_exec(
            "codex", "exec", "--json", "--ephemeral", "--skip-git-repo-check",
            "-C", str(work_dir),
            "-c", f"mcp_servers.corpus.url={json.dumps(self.codex_config.mcp_url)}",
            "-c", f"mcp_servers.corpus.enabled_tools={json.dumps(self.codex_config.enabled_tools)}",
            "--output-schema", schema_file, "-o", result_file, prompt,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env,
        )
        stdout, _ = await process.communicate()
        wall_s = time.monotonic() - t0
        terminate_state = "finished" if process.returncode == 0 else "error"
        (work_dir / f"{q.qid}.codex.jsonl").write_bytes(stdout)

        # the codex thread id is forwarded to OpenRouter as `session_id` on every request (parent and
        # spawned subagents alike), so the runner can meter this question's spend exactly
        session_id: str | None = None
        for line in stdout.decode(errors="replace").splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "thread.started":
                session_id = ev.get("thread_id")
                break
        # for line in stdout.decode().splitlines():
        #     ev = json.loads(line)
        #     if ev.get("type") == "turn.completed":
        #         u = ev["usage"]
        #         ctx.llm_client.usage.add(
        #             LLMResponse(
        #                 text="", latency_s=0.0, input_tokens=u["input_tokens"],
        #                 output_tokens=u["output_tokens"], cache_input_tokens=u["cached_input_tokens"],
        #             ),
        #             model="openai/gpt-5.6-luna",
        #             key=self.system_usage_key,
        #         )

        # read answer from result_file
        with open(result_file) as f:
            answer = json.load(f)

        # parse answer and return
        return AnswerOutput(
            answer=answer["answer"],
            retrieved_doc_ids=answer["doc_ids"],
            terminate_state=terminate_state,
            compute_wall_s=wall_s,
            session_id=session_id,
        )
