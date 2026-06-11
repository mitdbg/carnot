#!/usr/bin/env python3
"""Run one Skunk question with a terminal trace and terminal HITL prompts."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


SKUNK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SKUNK_ROOT.parent
SKUNK_SRC = SKUNK_ROOT / "src"
DEFAULT_QUESTIONS_DIR = SKUNK_ROOT / "harness_ui" / "questions"
DEFAULT_TRACE_DIR = SKUNK_ROOT / "interactive_traces"
DEFAULT_UID = "UID0018"

if str(SKUNK_SRC) not in sys.path:
    sys.path.insert(0, str(SKUNK_SRC))


def _set_default_env() -> None:
    os.environ.setdefault("SKUNK_PAGE_INDEX_DIR", str(SKUNK_ROOT / "cache/build_v3"))
    os.environ.setdefault(
        "OFFICEQA_PARSED_JSON_DIR",
        str(REPO_ROOT / "data/officeqa/treasury_bulletins_parsed/jsons"),
    )
    os.environ.setdefault(
        "OFFICEQA_PDF_DIR",
        str(REPO_ROOT / "data/officeqa/treasury_bulletin_pdfs"),
    )


def _load_dotenv() -> None:
    env_path = SKUNK_ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _question_files(paths: list[str] | None) -> list[Path]:
    if paths:
        return [Path(p).expanduser().resolve() for p in paths]
    return sorted(DEFAULT_QUESTIONS_DIR.glob("*.json"))


def _load_question(uid: str, files: list[Path]) -> tuple[str, dict[str, Any]]:
    wanted = uid.upper() if uid.lower().startswith("uid") else uid
    seen: list[str] = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        for round_payload in data.get("rounds", []):
            for question in round_payload.get("questions", []):
                qid = str(question.get("question_id", ""))
                seen.append(qid)
                if qid.upper() == wanted.upper():
                    metadata = {
                        "uid": qid,
                        "questions_file": str(path),
                        "round_num": round_payload.get("round_num"),
                        "canonical_answer": question.get("canonical_answer"),
                    }
                    return str(question["prompt"]), metadata
    sample = ", ".join(sorted({item for item in seen if item})[:12])
    raise KeyError(f"UID {uid!r} not found in question files. Sample IDs: {sample}")


def _trace_path(uid: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_uid = re.sub(r"[^A-Za-z0-9_.-]+", "_", uid)
    return DEFAULT_TRACE_DIR / f"{safe_uid}_{stamp}.json"


def _short(value: Any, limit: int = 140) -> str:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "..."


def _fmt_branch(branch: dict[str, Any]) -> str:
    branch_id = branch.get("branch_id", "?")
    kind = branch.get("kind", "branch")
    if kind == "retrieve":
        parts = [f"key={branch.get('key')!r}"]
        if branch.get("period"):
            parts.append(f"period={branch.get('period')!r}")
        if branch.get("as_of"):
            parts.append(f"as_of={branch.get('as_of')!r}")
        return f"  [{branch_id}] retrieve " + " ".join(parts)
    return (
        f"  [{branch_id}] lookup_external target={branch.get('target')!r} "
        f"src={branch.get('src')!r}"
    )


class TerminalTrace:
    def __init__(self) -> None:
        self._branches_by_key: dict[str, list[int]] = {}
        self._branch_labels: dict[int, str] = {}

    @staticmethod
    def _pages(data: dict[str, Any], total_key: str | None = None) -> str:
        pages = data.get("sample_pages")
        if not isinstance(pages, list) or not pages:
            return "(none)"
        refs = []
        for page in pages[:5]:
            if isinstance(page, dict):
                refs.append(f"{page.get('bulletin')} p{page.get('page')}")
        total = data.get(total_key) if total_key else None
        if isinstance(total, int) and total > len(refs):
            refs.append("...")
        return ", ".join(refs)

    def _branch_id_for_key(self, key: Any) -> str:
        if not isinstance(key, str):
            return "?"
        ids = self._branches_by_key.get(key, [])
        if len(ids) == 1:
            return str(ids[0])
        if ids:
            return ",".join(str(item) for item in ids)
        return "?"

    def _branch_line(self, branch_id: int | str) -> str:
        if isinstance(branch_id, int):
            return self._branch_labels.get(branch_id, f"branch {branch_id}")
        return f"branch {branch_id}"

    def render(self, event: dict[str, Any]) -> None:
        kind = event.get("kind")
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        message = str(event.get("message", ""))
        t = event.get("t")
        prefix = f"[{t:>7.3f}s]" if isinstance(t, (int, float)) else "[       ]"

        if kind == "plan" and data:
            label = data.get("label", "plan")
            suffix = ""
            if data.get("recovery_round") is not None:
                suffix += f" round={data.get('recovery_round')}"
            if data.get("reason"):
                suffix += f" reason={_short(data.get('reason'), 90)}"
            print(f"\n{prefix} plan: {label}{suffix}")
            self._branches_by_key = {}
            for branch in data.get("branches", []):
                if isinstance(branch, dict):
                    branch_id = branch.get("branch_id")
                    if isinstance(branch_id, int):
                        if branch.get("kind") == "retrieve":
                            key = str(branch.get("key"))
                            self._branches_by_key.setdefault(key, []).append(branch_id)
                            label_text = f"branch {branch_id} retrieve key={key!r}"
                        else:
                            label_text = (
                                f"branch {branch_id} lookup "
                                f"target={branch.get('target')!r}"
                            )
                        self._branch_labels[branch_id] = label_text
                        print(f"  |-- {label_text}")
            return

        if kind == "step" and data:
            op = event.get("op")
            branch_id = data.get("branch_id")
            if data.get("error"):
                branch = self._branch_line(branch_id) if isinstance(branch_id, int) else str(op)
                print(f"{prefix} |-- {branch}")
                print(f"      `-- {op} failed: {_short(data['error'])}")
                return
            summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
            if op == "retrieve":
                print(f"{prefix} retrieve sweep complete")
            elif op == "extract":
                blocks = summary.get("blocks", [])
                values = summary.get("values", [])
                branch = self._branch_line(branch_id) if isinstance(branch_id, int) else "branch ?"
                print(
                    f"{prefix} |-- {branch}\n"
                    f"      `-- extract: blocks={len(blocks)} values={len(values)}"
                )
                for value in values[:4]:
                    print(
                        "          |-- value "
                        f"{_short(value.get('description'), 70)} = "
                        f"{_short(value.get('value'), 70)}"
                    )
            elif op == "compute":
                print(f"{prefix} compute: {_short(summary, 100)}")
            return

        if message.startswith("human_document_scope"):
            branch = self._branch_id_for_key(data.get("key"))
            print(f"{prefix} |-- {self._branch_line(branch)}")
            print(
                "      |-- source docs: "
                f"{data.get('bulletins')} pages={data.get('pages')} -> "
                f"{self._pages(data, 'pages')}"
            )
        elif message.startswith("pick_chapters") and data.get("sample_pages"):
            print(
                f"{prefix} |-- retrieve sweep\n"
                "      |-- toc candidates: "
                f"{data.get('pages')} pages -> {self._pages(data, 'pages')}"
            )
        elif message.startswith("year_filter"):
            branch = self._branch_id_for_key(data.get("key"))
            print(f"{prefix} |-- {self._branch_line(branch)}")
            print(
                "      |-- year/date filter: "
                f"{data.get('kept_pages')}/{data.get('input_pages')} pages -> "
                f"{self._pages(data, 'kept_pages')}"
            )
        elif message.startswith("semantic_filter"):
            print(
                f"{prefix} |-- retrieve sweep\n"
                "      |-- semantic filter: "
                f"{data.get('kept_pages')}/{data.get('input_pages')} pages, "
                f"{data.get('kept_blocks')}/{data.get('input_blocks')} blocks -> "
                f"{self._pages(data, 'kept_pages')}"
            )
        elif message.startswith("page_index_retrieve"):
            branch = self._branch_id_for_key(data.get("key"))
            print(f"{prefix} |-- {self._branch_line(branch)}")
            print(
                "      |-- page survivors: "
                f"{data.get('anchor_count')} pages, {data.get('block_count')} blocks -> "
                f"{self._pages(data, 'anchor_count')}"
            )
        elif message.startswith("block_select"):
            branch = self._branch_id_for_key(data.get("key"))
            print(f"{prefix} |-- {self._branch_line(branch)}")
            print(
                "      `-- block select: "
                f"{data.get('selected_blocks')}/{data.get('candidates')} blocks -> "
                f"{self._pages(data, 'selected_blocks')}"
            )
        elif message.startswith("human_directed_retrieval"):
            print(f"{prefix} human directed retrieval: {_short(data, 100)}")
        elif message.startswith("human_annotation_retry_incomplete"):
            print(f"{prefix} human annotation retry incomplete: {_short(data, 100)}")
        elif message.startswith("mandatory_human_intervention_resolved"):
            print(f"{prefix} human resolved: {_short(data, 100)}")
        elif message.startswith("codegen_missing_data"):
            print(f"{prefix} compute missing data: {message}")


def _documents_from_text(raw: str) -> list[str]:
    docs: list[str] = []
    for item in re.split(r"[,;\n]+", raw):
        token = item.strip()
        if not token:
            continue
        match = re.search(r"(\d{4})[-_/](0[1-9]|1[0-2])", token)
        if match:
            ref = f"Treasury Bulletin {match.group(1)}-{match.group(2)} PDF"
        elif token.startswith("Treasury Bulletin "):
            ref = token
        else:
            print(f"  ignoring unrecognized document ref: {token!r}")
            continue
        if ref not in docs:
            docs.append(ref)
    return docs


def _retrieve_branches(guidance: dict[str, Any]) -> list[dict[str, Any]]:
    failed = [
        item
        for item in guidance.get("failed_branches", [])
        if isinstance(item, dict)
        and isinstance(item.get("branch_id"), int)
        and isinstance(item.get("branch"), dict)
        and item["branch"].get("kind") == "retrieve"
    ]
    if failed:
        return [
            {"branch_id": item["branch_id"], **item["branch"]}
            for item in failed
        ]
    return [
        branch
        for branch in guidance.get("partial_plan", [])
        if isinstance(branch, dict)
        and isinstance(branch.get("branch_id"), int)
        and branch.get("kind") == "retrieve"
    ]


def _print_intervention_context(
    kind: str,
    instructions: str,
    context: str | None,
    source_docs: list[str],
    guidance: dict[str, Any],
) -> None:
    print("\n" + "=" * 80)
    print(f"HUMAN INTERVENTION: {kind}")
    print(instructions)
    if context:
        print(context)
    if guidance.get("reason"):
        print(f"Reason: {guidance['reason']}")
    if guidance.get("missing"):
        print("Missing:")
        for item in guidance["missing"]:
            print(f"  - {item}")
    if source_docs:
        print("Likely pages:")
        for doc in source_docs[:12]:
            print(f"  - {doc}")
    previous = guidance.get("previous_round_plan", [])
    if previous:
        print("Previous branch outcomes:")
        for branch in previous:
            if not isinstance(branch, dict):
                continue
            status = branch.get("outcome_status") or branch.get("status")
            print(f"{_fmt_branch(branch)} status={status}")
            considered = branch.get("considered_pages") or []
            if considered:
                pages = ", ".join(
                    f"{p.get('bulletin')} p{p.get('page')}"
                    for p in considered[:12]
                    if isinstance(p, dict)
                )
                print(f"    considered: {pages}")
            blocks = branch.get("blocks") or []
            if blocks:
                for block in blocks[:6]:
                    if not isinstance(block, dict):
                        continue
                    title = block.get("title") or block.get("summary") or "(untitled)"
                    print(
                        f"    block {block.get('bulletin')} p{block.get('page')} "
                        f"#{block.get('block_index')}: {_short(title, 100)}"
                    )


async def terminal_human_handler(
    kind: str,
    instructions: str,
    context: str | None,
    source_docs: list[str],
    guidance: dict[str, Any] | None,
) -> dict[str, Any]:
    guidance = guidance or {}
    _print_intervention_context(kind, instructions, context, source_docs, guidance)
    directives: list[dict[str, Any]] = []
    for branch in _retrieve_branches(guidance):
        branch_id = branch["branch_id"]
        print(f"\nSource documents for branch {branch_id}: {branch.get('key')!r}")
        raw = input("  Enter bulletin months, e.g. 1986-06,1985-12 (blank for none): ")
        docs = _documents_from_text(raw)
        if docs:
            directives.append({"branch_id": branch_id, "documents": docs})

    response_lines: list[str] = []
    missing = [str(item) for item in guidance.get("missing", []) if str(item).strip()]
    if missing:
        print("\nManual values are optional. Leave blank to only send source docs.")
        for item in missing:
            value = input(f"  Value for {item!r}: ").strip()
            if value:
                response_lines.append(f"{item}: {value}")
    else:
        response = input("\nManual response (blank if none): ").strip()
        if response:
            response_lines.append(response)

    source_docs_response = sorted(
        {doc for directive in directives for doc in directive["documents"]}
    )
    print("=" * 80 + "\n")
    return {
        "response": "\n".join(response_lines),
        "source_docs": source_docs_response,
        "retrieval_directives": directives,
    }


def _source_docs_from_events(events: list[dict[str, Any]]) -> list[str]:
    docs: list[str] = []
    seen: set[str] = set()
    for event in events:
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        summary = data.get("summary")
        if not isinstance(summary, dict):
            continue
        for value in summary.get("values", []):
            if not isinstance(value, dict):
                continue
            month = value.get("bulletin")
            pages = value.get("pages")
            if isinstance(month, str) and isinstance(pages, list):
                for page in pages:
                    if isinstance(page, int):
                        ref = f"Treasury Bulletin {month} PDF page {page}"
                        if ref not in seen:
                            seen.add(ref)
                            docs.append(ref)
    return docs


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    from skunk import MissingData, Orchestrator, SkunkConfig, StepFailed, load_prompt_overrides

    _load_dotenv()
    _set_default_env()

    uid = args.uid.upper() if args.uid.lower().startswith("uid") else args.uid
    metadata: dict[str, Any] = {"uid": uid}
    if args.question:
        question = args.question
        metadata["questions_file"] = None
        metadata["canonical_answer"] = None
    else:
        question, metadata = _load_question(uid, _question_files(args.questions_file))
        uid = str(metadata["uid"])

    trace_out = Path(args.trace_out).expanduser() if args.trace_out else _trace_path(uid)
    if not trace_out.is_absolute():
        trace_out = SKUNK_ROOT / trace_out
    trace_out.parent.mkdir(parents=True, exist_ok=True)

    config = SkunkConfig.from_env()
    if not Path(config.prompt_overrides_path).is_absolute():
        config.prompt_overrides_path = str(SKUNK_ROOT / config.prompt_overrides_path)
    overrides_path = Path(config.prompt_overrides_path)
    prompt_overrides = load_prompt_overrides(overrides_path) if overrides_path.exists() else ()

    print(f"UID: {uid}")
    print(f"Question: {question}")
    if metadata.get("canonical_answer") is not None:
        print(f"Canonical answer: {metadata['canonical_answer']}")
    print(f"Trace JSON: {trace_out}")
    print()

    terminal = TerminalTrace()
    started = time.perf_counter()
    orch = Orchestrator(
        question,
        uid=uid,
        config=config,
        prompt_overrides=prompt_overrides,
        verbose=False,
        human_intervention_handler=terminal_human_handler,
    )
    original_emit = orch.ctx.emit

    def emit_and_render(
        message: str,
        level: str | None = None,
        *,
        kind: str | None = None,
        data: dict | None = None,
    ) -> None:
        original_emit(message, level, kind=kind, data=data)
        terminal.render(orch.ctx.events[-1])

    orch.ctx.emit = emit_and_render  # type: ignore[method-assign]

    answer: str | None = None
    failure: str | None = None
    try:
        try:
            answer = await orch.execute()
        except (MissingData, StepFailed, RuntimeError) as exc:
            failure = str(exc)
            print(f"\nFAILED: {failure}", file=sys.stderr)
        elapsed = time.perf_counter() - started
        source_docs = _source_docs_from_events(orch.ctx.events)
        payload = {
            "uid": uid,
            "question": question,
            "canonical_answer": metadata.get("canonical_answer"),
            "questions_file": metadata.get("questions_file"),
            "round_num": metadata.get("round_num"),
            "answer": answer,
            "failure": failure,
            "elapsed_s": round(elapsed, 3),
            "source_docs": source_docs,
            "events": orch.ctx.events,
        }
        trace_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        print("\n" + "=" * 80)
        if answer is not None:
            print(f"ANSWER: {answer}")
        if failure is not None:
            print(f"FAILURE: {failure}")
        print(f"Elapsed: {elapsed:.2f}s")
        print(f"Trace JSON written: {trace_out}")
        return payload
    finally:
        orch.ctx.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one Skunk question with live terminal retrieve-flow tracing."
    )
    parser.add_argument(
        "--uid",
        default=DEFAULT_UID,
        help=f"Question ID to load from bundled question files. Default: {DEFAULT_UID}.",
    )
    parser.add_argument(
        "--question",
        help="Run this literal question instead of loading --uid from question files.",
    )
    parser.add_argument(
        "--questions-file",
        action="append",
        help="Question JSON file to search. May be repeated. Defaults to all harness_ui/questions/*.json files.",
    )
    parser.add_argument(
        "--trace-out",
        help="Output JSON trace path. Defaults to interactive_traces/<uid>_<timestamp>.json.",
    )
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
