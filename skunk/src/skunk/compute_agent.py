"""ComputeAgentOp — the compute operator as a multi-turn, code-executing agent.

The single-shot `ComputeOp` (`compute.py`) emits ONE Python block over the gathered
`input_values` and takes the result. `ComputeAgentOp` instead runs a `MultiTurnAgent` loop:
it writes and executes Python step-by-step over the SAME pre-loaded `input_values`, observing
intermediate results across steps, AND can call the corpus tools the SearchAgent uses
(`search_corpus` / `grep_corpus` / `read_document` / `view_figure`). Those tools are for
CONTEXTUALIZING the inputs it was handed — resolving a unit, footnote, print-flag, scope
qualifier, or reading a chart behind a value — NOT for retrieving the primary data from
scratch (that is the upstream agents' job; genuinely-missing data still yields `NeedsMore` and
the orchestrator replans).

Selected behind `SKUNK_COMPUTE_AGENT=1` (see `make_compute_op` / `SkunkConfig.compute_agent`).
It keeps the same `run(...)` signature and best-of-N voting as `ComputeOp`, so it drops into the
orchestrator's call sites unchanged. It REQUIRES the search-agent corpus artifacts (chromadb +
clean_page_map), like `SKUNK_RETRIEVER=search_agent`.

Final answer (a single ```json``` block, parsed as data, not executed):
  - success:      {"result": "<answer string>"}
  - missing data: {"missing": [...], "description": "...", "keep_inputs": [<indices>],
                   "keep_computed": {"<name>": <scalar | [scalars] | {label: scalar}>}}
Since JSON can't carry live exec-env objects, `keep` references inputs by INDEX and computed
values as literals, then maps back to `AnnotatedValue`s (provenance preserved for inputs)."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from skunk.common import (
    AnnotatedValue,
    B64Image,
    ExecutionContext,
    Final,
    NeedsMore,
)
from skunk.compute import (
    MissingDataSignal,
    _NONFINITE_RE,
    annotated_value_from_computed,
    build_compute_user_message,
    vote_compute_outcomes,
)
from skunk.config import SkunkConfig
from skunk.local_python_executor import CodeOutput, LocalPythonExecutor
from skunk.multi_turn_agent import Block, ImageBlock, MultiTurnAgent, TextBlock
from skunk.pyexec import _AUTHORIZED_IMPORTS, _PRELOADED_GLOBALS
from skunk.question_explainer import ConceptExplanation
from skunk.search_agent.search_tools import (
    EMPTY_RESULT_MESSAGE,
    GREP_RESULT_TAG,
    READ_DOCUMENT_RESULT_TAG,
    SEARCH_RESULT_TAG,
    VIEW_FIGURE_RESULT_TAG,
)


def outcome_from_payload(
    payload: Any, input_values: list[AnnotatedValue]
) -> Final | NeedsMore:
    """Map the agent's validated JSON final answer to a `Final | NeedsMore`. Raises
    `ValueError` (with a fix-it detail) on any malformed / non-finite shape — `validate_final_answer`
    wraps this so a bad final answer becomes loop feedback rather than a crash."""
    if not isinstance(payload, dict):
        raise ValueError(
            'final answer must be a JSON object with "result" (success) or "missing" '
            "(missing data), not a bare value/list"
        )
    if "result" in payload:
        result = str(payload["result"])
        if _NONFINITE_RE.search(result):
            raise ValueError(
                f"`result` is non-finite ({result!r}). A NaN/inf answer means the computation "
                "is undefined — typically NaN cells in the inputs ('n/a') or division by an "
                "empty/zero quantity. Drop/skip the NaN cells before aggregating, or — if the "
                "inputs genuinely cannot support the answer — emit the missing-data form instead."
            )
        return Final(result)
    if "missing" in payload:
        try:
            signal = MissingDataSignal.model_validate(
                {
                    "missing": payload.get("missing", []),
                    "description": payload.get("description", ""),
                }
            )
        except Exception as e:
            raise ValueError(
                "`missing` must be a list of short id strings, alongside a string "
                f'"description" — {e}'
            )
        keep: list[AnnotatedValue] = []
        keep_inputs = payload.get("keep_inputs", []) or []
        if not isinstance(keep_inputs, list) or not all(
            isinstance(i, int) and not isinstance(i, bool) for i in keep_inputs
        ):
            raise ValueError(
                "`keep_inputs` must be a list of integer indices into input_values"
            )
        for i in keep_inputs:
            if not (0 <= i < len(input_values)):
                raise ValueError(
                    f"keep_inputs index {i} out of range (0..{len(input_values) - 1})"
                )
            keep.append(input_values[i])
        keep_computed = payload.get("keep_computed", {}) or {}
        if not isinstance(keep_computed, dict) or not all(
            isinstance(k, str) for k in keep_computed
        ):
            raise ValueError(
                "`keep_computed` must be a dict keyed by str names, each value a scalar, a "
                "list of scalars, or a flat {label: scalar} dict"
            )
        for name, v in keep_computed.items():
            keep.append(annotated_value_from_computed(name, v))
        return NeedsMore(
            keep=keep, missing_reason=signal.description, missing=signal.missing
        )
    raise ValueError(
        'final answer must contain "result" (success) or "missing" (missing data)'
    )


_BRIEFING = """\
You are the COMPUTE step of a deep-research QA pipeline. Upstream agents have already gathered \
the values relevant to the question into `input_values`; your job is to compute the question's \
final answer from them — and ONLY from them, except to contextualize them as described below.

## Your environment
`input_values: list[AnnotatedValue]` is already loaded in your Python sandbox, along with \
`numpy` (np), `pandas` (pd), `statsmodels.api` (sm), `math`, and `statistics`. The sandbox is \
PERSISTENT across your steps: variables you assign survive, so you can compute incrementally — \
inspect a frame, `print(...)` an intermediate, then build on it next step.

## AnnotatedValue API
  .description   natural-language label
  .notes         prose page context (footnotes, headnotes, caveats, print-flag p/r meanings);
                 shared across a vector/table's cells, so it is context, not a per-cell label
  .frame         pd.DataFrame view of the payload (uniform across kinds)
  .unit          natural-language unit, e.g. "millions of dollars", "percent"
  .kind          "scalar" | "vector" | "table"  (prefer .frame)
  .index_name    (vector)   .row_name / .col_name (table)
  .value         raw payload — only if you specifically need the dict/list form
  .source        publisher/origin of an external-lookup value (empty for corpus extracts)
  .bulletin      source issue "YYYY-MM" the value was printed in
  .pages         source PDF page number(s)
  .requested_period / .retrieve_key   the period/concept this datum was retrieved for

## Payload access (always read data through `e.frame`)
  scalar  →  1x1 DataFrame; `e.frame.iat[0, 0]` for the raw cell (a lookup scalar may be N x 1).
  vector  →  N x 1 DataFrame; index is `e.index_name`. `e.frame.iloc[:, 0]` for the Series.
  table   →  R x C DataFrame; index.name == e.row_name, columns.name == e.col_name.
Read the actual cells to spot NaN / "n/a" values and handle them before aggregating. Apply unit
conversions once over the whole frame, never cell-by-cell.

## Selecting inputs
- Reference entries by index (`input_values[7].frame`).
- Entries may repeat. Pick the one whose description (and notes) match the question's wording,
  including qualifier words ("subject to limitation", "accepted", "issued"). Do not combine
  multiple entries for max/min/avg/sum.
- Extra scope words mark a different, broader series, not a looser label: "public debt" is not
  answered by "public debt and guaranteed obligations"; "savings bonds" is not "savings bonds
  and savings notes". Match the exact series even when a broader entry is easier to read.

## Tools — use them ONLY to contextualize the inputs you were given
Call these from a ```python``` block, exactly like any function. They search the SAME source
corpus the inputs came from. Use them sparingly and only to make sense of an input you already
have — e.g. resolve an ambiguous unit, read a footnote/headnote or print-flag, confirm a scope
qualifier, or view a chart/figure that a value was read from. Do NOT use them to gather the
primary quantities from scratch: if a value the question needs is simply absent from
`input_values`, that is missing data — emit the missing-data final answer (below) so the
pipeline can replan and retrieve it. Each input carries `.bulletin` / `.pages`, so you know
which source pages to look at when you need context.

## Discipline
- Carry full precision through intermediates; round/format only in the final answer string, to
  the decimal places the question states.
- Never produce a non-finite answer (NaN / inf). If a computation is undefined, fix the inputs
  (drop NaN cells) or emit missing data.
- Never fabricate. Real-world reference data (exchange rates, deflators, CPI, GDP, population,
  market prices) is data, not knowledge: if no input carries it and a tool can't find it on the
  inputs' own source pages, list it under missing data rather than supplying it from memory."""

_FINAL_ANSWER_DOC = """\
A single ```json``` block, exactly ONE of these two shapes:

(a) Success — you computed the answer:
```json
{"result": "<the answer string ONLY, in the exact format/units/precision the question asks>"}
```
The string contains only the requested answer — no prose, no "Answer:", no question restatement;
for a multi-part question, only the ultimate quantity asked for.

(b) Missing data — `input_values` cannot support the answer even after contextualizing:
```json
{"missing": ["<short id>", "..."], "description": "<one-line reason>",
 "keep_inputs": [<indices into input_values to carry to the next round>],
 "keep_computed": {"<name_with_unit>": <scalar | [scalars] | {label: scalar}>}}
```
Do as much as you can first, then signal. `keep_inputs` lists the indices of the input entries
worth reusing (carried with their provenance); `keep_computed` holds any partial results you
derived (recorded provenance-free). Both are optional — omit or use [] / {} to carry nothing.
Everything not kept is dropped for the next round."""


class _ComputeAgent(MultiTurnAgent):
    """One compute-agent trial: a `MultiTurnAgent` whose sandbox is pre-seeded with this call's
    `input_values` (+ numpy/pandas/...) and the corpus tools. Built fresh per trial."""

    name = "compute.agent"
    default_effort = (
        "high"  # codegen/reasoning over the extracted values, like compute.codegen
    )
    briefing = _BRIEFING
    final_answer_doc = _FINAL_ANSWER_DOC
    # Generous: a contextualizing read_document over dense Treasury tables is large, and the
    # trajectory accumulates compute steps. Char budget (~1.5 chars/token on these numerics).
    context_budget_chars = 1_000_000
    warn_steps_remaining = 2
    # Python steps need the same imports the single-shot codegen sandbox authorizes.
    authorized_imports = _AUTHORIZED_IMPORTS

    def __init__(
        self,
        config: SkunkConfig,
        input_values: list[AnnotatedValue],
        tools: list,
    ) -> None:
        self._input_values = input_values
        super().__init__(
            tools,
            max_steps=config.compute_agent_max_steps,
            max_misfires=config.agent_max_misfires,
        )

    def _build_executor(self) -> LocalPythonExecutor:
        """The base binds the tools; we additionally seed the persistent sandbox with the
        compute preload set (np/pd/sm/math/...) and this call's `input_values`, so the agent's
        python steps can read `input_values[i].frame` and compute over them across steps."""
        executor = super()._build_executor()
        executor.send_variables(
            {**_PRELOADED_GLOBALS, "input_values": self._input_values}
        )
        return executor

    def validate_final_answer(
        self, payload: object, observations: list[str]
    ) -> str | None:
        """Reject a malformed / non-finite final answer with feedback so the agent self-corrects
        within its step budget; accept (None) otherwise. The accept path is exactly what
        `outcome_from_payload` parses, so a validated payload converts without raising."""
        try:
            outcome_from_payload(payload, self._input_values)
        except ValueError as e:
            return str(e)
        return None

    def _blocks_from_output(self, out: CodeOutput) -> list[Block]:
        """Render the corpus tools' tagged-dict returns as observation blocks. No `prune` tool
        here, so chunks need no redaction — they render as plain `TextBlock`s; `view_figure`
        yields an `ImageBlock`. Everything else falls back to the base [stdout]/[result]."""
        blocks: list[Block] = []
        stdout_s = (out.logs or "").strip()
        if stdout_s:
            blocks.append(TextBlock(f"[stdout]\n{stdout_s}"))
        output = out.output

        if isinstance(output, dict) and output.get(SEARCH_RESULT_TAG):
            if output.get("error"):
                blocks.append(TextBlock(f"[error]\n{output['error']}"))
            elif not output["chunks"]:
                blocks.append(TextBlock(EMPTY_RESULT_MESSAGE))
            else:
                blocks.extend(TextBlock(c["text"]) for c in output["chunks"])
            return blocks

        if isinstance(output, dict) and output.get(GREP_RESULT_TAG):
            if output.get("error"):
                blocks.append(TextBlock(f"[error]\n{output['error']}"))
            elif not output["groups"]:
                blocks.append(TextBlock(EMPTY_RESULT_MESSAGE))
            else:
                for group in output["groups"]:
                    blocks.append(TextBlock(group["header"]))
                    blocks.extend(TextBlock(c["text"]) for c in group["chunks"])
            if output.get("truncation_note"):
                blocks.append(TextBlock(output["truncation_note"]))
            return blocks

        if isinstance(output, dict) and output.get(READ_DOCUMENT_RESULT_TAG):
            blocks.extend(TextBlock(d["text"]) for d in output["docs"])
            return blocks

        if isinstance(output, dict) and output.get(VIEW_FIGURE_RESULT_TAG):
            if output.get("error"):
                blocks.append(TextBlock(f"[error]\n{output['error']}"))
            else:
                caption = (
                    f"[full-page image of doc_id={output['doc_id']} "
                    f"(contains <figure id={output['figure_id']}>)]"
                )
                blocks.append(
                    ImageBlock(
                        doc_id=output["doc_id"],
                        figure_id=output["figure_id"],
                        image=B64Image(mime=output["mime"], data=output["data"]),
                        text=caption,
                    )
                )
            return blocks

        # Non-tool output (e.g. a bare print/compute step): default [result] rendering.
        result_s = "" if output is None else str(output).strip()
        if result_s and result_s != stdout_s and result_s not in stdout_s:
            blocks.append(TextBlock(f"[result]\n{result_s}"))
        if not blocks:
            blocks.append(TextBlock("[no output]"))
        return blocks


class ComputeAgentOp:
    """The multi-turn compute operator. Same public `run()` as `ComputeOp`."""

    async def run(
        self,
        input_values: list[AnnotatedValue],
        ctx: ExecutionContext,
        concept_explanations: Sequence[ConceptExplanation] = (),
        *,
        round_idx: int = 0,
    ) -> Final | NeedsMore:
        """Best-of-N multi-turn compute: run `compute_best_of_n` independent agent trials in
        parallel and vote on the outcome (see `vote_compute_outcomes`). Each trial is one
        `_ComputeAgent` loop returning `Final | NeedsMore` (or raising `StepFailed`). N≤1 runs a
        single trial. Raises `StepFailed` only if every trial does."""
        # Same final-stage provenance event the single-shot ComputeOp emits (eval recall report).
        src_pages = sorted(
            {f"{e.bulletin}:{p}" for e in input_values if e.bulletin for p in e.pages}
        )
        ctx.emit(
            f"compute_inputs n_values={len(input_values)} n_pages={len(src_pages)}",
            data={"pages": src_pages},
        )

        tools = self._build_tools(ctx.config)
        user_msg = build_compute_user_message(ctx, input_values, concept_explanations)

        n = ctx.config.compute_best_of_n
        if n <= 1:
            return await self._run_trial(input_values, tools, ctx, user_msg)

        results = await asyncio.gather(
            *(self._run_trial(input_values, tools, ctx, user_msg) for _ in range(n)),
            return_exceptions=True,
        )
        return vote_compute_outcomes(results, ctx)

    @staticmethod
    async def _run_trial(
        input_values: list[AnnotatedValue],
        tools: list,
        ctx: ExecutionContext,
        user_msg: str,
    ) -> Final | NeedsMore:
        """One agent trial. Tools are shared across trials (read-only for compute — no prune);
        each trial gets its own agent (fresh sandbox + trajectory). `agent.call` returns the
        validated final-answer payload or raises `StepFailed` when the step budget is spent."""
        agent = _ComputeAgent(ctx.config, input_values, tools)
        payload = await agent.call(ctx, user_msg)
        return outcome_from_payload(payload, input_values)

    @staticmethod
    def _build_tools(config: SkunkConfig) -> list:
        """Build the corpus tools over the shared, process-wide-cached corpus resources. No
        `prune` tool (irrelevant to compute), so the search/grep tools read empty prune sets."""
        # Local imports: keep ComputeAgentOp importable without pulling chromadb / the search
        # stack until the agent path is actually selected (and to avoid an import cycle through
        # `retrieve`, which the orchestrator also imports).
        from skunk.retrieve import load_corpus_resources
        from skunk.search_agent.search_agent import _make_embedding_client
        from skunk.search_agent.search_tools import (
            GrepCorpusTool,
            ReadDocumentTool,
            SearchCorpusTool,
            ViewFigureTool,
        )

        collection, document_map = load_corpus_resources(config)
        emb_client, emb_model_id = _make_embedding_client(config.emb_model_id)
        pruned_chunk_ids: set[str] = set()
        pruned_doc_ids: set[str] = set()
        return [
            SearchCorpusTool(
                collection,
                emb_model_id,
                emb_client,
                pruned_chunk_ids,
                pruned_doc_ids,
            ),
            GrepCorpusTool(
                collection,
                pruned_chunk_ids,
                pruned_doc_ids,
                config.grep_max_output_tokens,
            ),
            ReadDocumentTool(
                document_map,
                config.agent_max_pages_per_tool_call,
                config.read_document_max_output_chars,
            ),
            ViewFigureTool(document_map, config.pdf_dir),
        ]
