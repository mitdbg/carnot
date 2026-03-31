import asyncio
import concurrent.futures
import json
import logging
import os
import queue
import re
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import fsspec
import pandas as pd
from cloudpathlib import S3Path
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.background import BackgroundTask

import carnot
from app.auth import get_current_user
from app.database import (
    AsyncSessionLocal,
    Conversation,
    Dataset,
    DatasetFile,
    Message,
    Notebook,
    QueryEvent,
    QueryStats,
    Workspace,
    get_db,
)
from app.database import (
    File as FileRecord,
)
from app.env import BASE_DIR, FILESYSTEM, IS_LOCAL_ENV
from app.services.file_service import LocalFileService, S3FileService
from app.services.llm import get_user_llm_config
from app.services.serializer import jsonb_serializer
from carnot.data.dataset import Dataset as CarnotDataset
from carnot.data.item import DataItem
from carnot.plan import PhysicalPlan
from carnot.storage.backend import LocalStorageBackend, S3StorageBackend
from carnot.storage.tiered import TieredStorageManager

router = APIRouter()
# logger = logging.getLogger(__name__)
logger = logging.getLogger('uvicorn.error')
file_service = LocalFileService() if IS_LOCAL_ENV else S3FileService()


def _build_storage() -> TieredStorageManager:
    """Create a TieredStorageManager appropriate for the current environment.

    Requires:
        - ``IS_LOCAL_ENV`` is set correctly.
        - For S3: the ``S3_BUCKET`` / ``S3_PREFIX`` env vars are present.

    Returns:
        A ready-to-use :class:`TieredStorageManager` wrapping either a
        :class:`LocalStorageBackend` or :class:`S3StorageBackend`.

    Raises:
        None.
    """
    if IS_LOCAL_ENV:
        backend = LocalStorageBackend(base_dir=BASE_DIR)
    else:
        bucket = os.getenv("S3_BUCKET", "")
        prefix = os.getenv("S3_PREFIX", "")
        backend = S3StorageBackend(bucket=bucket, prefix=prefix)
    return TieredStorageManager(backend=backend)


async def _load_carnot_datasets(
    dataset_ids: list[int],
    db_session: AsyncSession | None = None,
) -> list[CarnotDataset]:
    """Load Carnot ``Dataset`` objects for a list of database IDs.

    Opens a fresh ``AsyncSessionLocal`` when *db_session* is ``None``,
    otherwise reuses the caller's session.

    Requires:
        - Every ID in *dataset_ids* exists in the ``datasets`` table.

    Returns:
        A list of :class:`CarnotDataset` instances (same order as
        *dataset_ids*) with their items populated.

    Raises:
        HTTPException 404: if any dataset ID is not found.
    """

    async def _inner(db: AsyncSession) -> list[CarnotDataset]:
        datasets: list[CarnotDataset] = []
        for dataset_id in dataset_ids:
            dataset_result = await db.execute(
                select(Dataset).where(Dataset.id == dataset_id)
            )
            dataset = dataset_result.scalar_one_or_none()
            if not dataset:
                raise HTTPException(
                    status_code=404,
                    detail=f"Dataset {dataset_id} not found",
                )
            files_result = await db.execute(
                select(FileRecord)
                .join(DatasetFile, FileRecord.id == DatasetFile.file_id)
                .where(DatasetFile.dataset_id == dataset_id)
            )
            files = files_result.scalars().all()
            carnot_dataset = CarnotDataset(
                name=dataset.name,
                annotation=dataset.annotation,
                items=[DataItem(path=file.file_path) for file in files],
                storage=_build_storage(),
            )
            datasets.append(carnot_dataset)
        return datasets

    if db_session is not None:
        return await _inner(db_session)

    async with AsyncSessionLocal() as db:
        return await _inner(db)


# heartbeat and session timeout settings; the heartbeat ensures that the connection
# to the frontend is kept alive during long-running queries
HEARTBEAT_INTERVAL = 30
SESSION_TIMEOUT = timedelta(minutes=30)

class OutputCapture:
    """
    Captures output and writes it to a file (local or S3) while also echoing it to the original stream.
    """
    def __init__(self, filepath: str, original_stream):
        self.filepath = filepath
        self.original_stream = original_stream
        self.temp_filepath = Path(tempfile.gettempdir()) / f"output_temp_{uuid4()}.txt"
        self.file_handle = None
        self.file_stream = None
        self._open_stream()

    def _open_stream(self):
        """Open the stream using fsspec."""
        fs = fsspec.filesystem("file")
        fp = self.filepath if IS_LOCAL_ENV else str(self.temp_filepath)
        self.file_handle = fs.open(fp, mode='a', encoding='utf-8')
        self.file_stream = self.file_handle.__enter__()

    def write(self, data: str):
        """Write data to both the file stream (S3/local) and the original stream (console/terminal)."""
        self.file_stream.write(data)
        self.original_stream.write(data)

    def flush(self):
        """Flush data to both the file stream (S3/local) and the original stream (console/terminal)."""
        self.file_stream.flush()
        self.original_stream.flush()

    def close(self):
        """Close the file stream safely."""
        self.file_handle.__exit__(None, None, None)
        self.file_handle = None
        self.file_stream = None

active_sessions: dict[str, dict] = {}


@dataclass
class NotebookState:
    """In-memory state for an interactive notebook session.

    Representation invariant:
        - ``notebook_id`` is a non-empty UUID string.
        - ``physical_plan`` is a ``PhysicalPlan`` with at least one node.
        - ``datasets_store`` keys are a superset of all
          ``output_dataset_id`` values for executed nodes.
        - ``cell_statuses`` maps ``node_id`` → ``"pending"`` |
          ``"running"`` | ``"success"`` | ``"error"``.  This is
          application-level state, not part of the plan itself.
        - ``cell_outputs`` maps ``node_id`` → output-preview dict for
          every successfully executed cell.  Cleared on invalidation.

    Abstraction function:
        Represents a running notebook backed by a physical query plan.
        ``datasets_store`` accumulates the operator outputs as the user
        executes nodes one-by-one, acting as the notebook's "kernel
        state".  ``cell_outputs`` mirrors the preview dicts sent to the
        frontend so they can be persisted and restored.
    """

    notebook_id: str
    query: str
    physical_plan: PhysicalPlan
    datasets_store: dict = field(default_factory=dict)
    cell_statuses: dict[str, str] = field(default_factory=dict)
    cell_outputs: dict[str, dict] = field(default_factory=dict)
    llm_config: dict = field(default_factory=dict)
    user_id: str = ""
    session_id: str = ""
    dataset_ids: list[int] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_access: datetime = field(default_factory=datetime.now)
    cost_budget: float | None = None
    storage: TieredStorageManager | None = None
    execution: carnot.Execution | None = None

    def get_cells(self) -> list[dict]:
        """Serialise the plan to frontend cell descriptors.

        Each cell dict contains the plan node's fields plus
        application-level ``cell_id`` (alias for ``node_id``),
        ``status`` from ``cell_statuses``, and ``output`` from
        ``cell_outputs`` (if the cell has been executed).

        Requires:
            None.

        Returns:
            A list of cell-descriptor dicts in topological order,
            suitable for the notebook frontend.

        Raises:
            None.
        """
        cells = self.physical_plan.to_node_dicts()
        for cell in cells:
            node_id = cell["node_id"]
            # Add frontend-compatible aliases
            cell["cell_id"] = node_id
            cell["cell_type"] = cell["node_type"]
            cell["status"] = self.cell_statuses.get(node_id, "pending")
            # Include persisted output so cells survive page navigation
            output = self.cell_outputs.get(node_id)
            if output is not None:
                cell["output"] = output
        return cells


active_notebooks: dict[str, NotebookState] = {}
NOTEBOOK_TIMEOUT = timedelta(minutes=60)


async def _rehydrate_notebook(
    notebook_uuid: str,
    user_id: str,
    db: AsyncSession,
) -> NotebookState | None:
    """Re-create an in-memory ``NotebookState`` from its DB row.

    When a notebook's kernel has been evicted (timeout or server
    restart), the ``active_notebooks`` dict no longer contains it.
    This function rebuilds the full execution context from the
    persisted ``plan_json`` and ``cells_json`` so that the notebook
    becomes interactive again.

    Requires:
        - ``notebook_uuid`` is a valid UUID string.
        - ``user_id`` is the authenticated user.

    Returns:
        A fully initialised ``NotebookState`` (already inserted into
        ``active_notebooks``), or ``None`` if the DB row is missing
        or rehydration fails.

    Raises:
        None.  Errors are logged and ``None`` is returned.
    """
    try:
        result = await db.execute(
            select(Notebook).where(Notebook.notebook_uuid == notebook_uuid)
        )
        nb_row = result.scalar_one_or_none()
        if nb_row is None:
            return None

        plan_json = nb_row.plan_json
        if not plan_json:
            logger.warning("Cannot rehydrate notebook %s — no plan_json", notebook_uuid)
            return None

        # Get dataset_ids from the parent workspace
        ws_result = await db.execute(
            select(Workspace).where(Workspace.id == nb_row.workspace_id)
        )
        ws_row = ws_result.scalar_one_or_none()
        if ws_row is None:
            logger.warning("Cannot rehydrate notebook %s — workspace %s not found",
                           notebook_uuid, nb_row.workspace_id)
            return None

        dataset_ids: list[int] = []
        if ws_row.dataset_ids:
            dataset_ids = [int(x) for x in ws_row.dataset_ids.split(",") if x.strip()]

        if not dataset_ids:
            logger.warning("Cannot rehydrate notebook %s — no dataset_ids on workspace %s",
                           notebook_uuid, nb_row.workspace_id)
            return None

        user_config = await get_user_llm_config(db, user_id)
        if not user_config:
            logger.warning("Cannot rehydrate notebook %s — no LLM config for user", notebook_uuid)
            return None

        datasets = await _load_carnot_datasets(dataset_ids, db_session=db)
        storage = _build_storage()

        # plan_json may be either the original logical plan (recursive tree
        # with "parents"/"params") or an updated physical plan (flat dict
        # with "nodes" list from PhysicalPlan.to_dict(), saved after
        # add/delete/move).  Detect the format and reconstruct accordingly.
        if "nodes" in plan_json:
            # Physical plan format — reconstruct directly
            physical_plan = PhysicalPlan.from_dict(plan_json)
        else:
            # Logical plan format — need to convert via Execution
            physical_plan = PhysicalPlan.from_plan_dict(
                plan_json, datasets, query=nb_row.query,
            )

        exec_instance = carnot.Execution(
            query=nb_row.query,
            datasets=datasets,
            plan=physical_plan,
            tools=[],
            memory=None,
            indices=[],
            llm_config=user_config,
            cost_budget=None,
            storage=storage,
        )

        # Rebuild cell_outputs from persisted cells_json so the frontend
        # can display previous results.  However, all statuses are reset to
        # "pending" because the in-memory datasets_store (operator outputs)
        # is gone — cells must be re-executed from the top to rebuild it.
        cell_statuses: dict[str, str] = {}
        cell_outputs: dict[str, dict] = {}
        for cell in (nb_row.cells_json or []):
            nid = cell.get("cell_id") or cell.get("node_id")
            if nid:
                cell_statuses[nid] = "pending"
                if cell.get("output") is not None:
                    cell_outputs[nid] = cell["output"]

        # Seed datasets_store with the user-supplied datasets keyed by name
        initial_store: dict[str, carnot.Dataset] = {
            ds.name: ds for ds in datasets
        }

        nb = NotebookState(
            notebook_id=notebook_uuid,
            query=nb_row.query,
            physical_plan=physical_plan,
            datasets_store=initial_store,
            cell_statuses=cell_statuses,
            cell_outputs=cell_outputs,
            llm_config=user_config,
            user_id=user_id,
            session_id=ws_row.session_id,
            dataset_ids=dataset_ids,
            cost_budget=None,
            storage=storage,
            execution=exec_instance,
        )
        active_notebooks[notebook_uuid] = nb
        logger.info("Rehydrated notebook %s from DB", notebook_uuid)
        return nb

    except Exception:
        logger.warning("Failed to rehydrate notebook %s", notebook_uuid, exc_info=True)
        return None


class QueryExecutionStreamer:
    """
    Encapsulates the query execution and concurrent heartbeat task for SSE streaming.
    Uses an asyncio.Queue to safely merge output from the query task and the heartbeat task.
    """
    def __init__(self, query: str, plan: dict, dataset_ids: list[int], user_id: str, session_id: str, user_config: dict, cost_budget: float | None = None):
        self.query = query
        self.plan = plan
        self.dataset_ids = dataset_ids
        self.user_id = user_id
        self.session_id = session_id
        self.user_config = user_config
        self.cost_budget = cost_budget
        self.queue = asyncio.Queue()
        self.heartbeat_task = None
        self.query_task = None

    async def _heartbeat_sender(self):
        """Sends an SSE comment line to the queue periodically to keep the connection alive."""
        try:
            while True:
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                # Put the SSE comment directly into the queue
                # The browser's EventSource client ignores lines starting with ':'
                await self.queue.put(":keep-alive\n\n")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Heartbeat sender failed: {e}")

    async def _run_query_logic(self):
        """Runs the main query execution logic and puts SSE data into the queue."""
        conversation_id = None

        try:
            cleanup_old_sessions()

            session_id = self.session_id or str(uuid4())
            conversation_id = await get_or_create_conversation(
                self.user_id, session_id, self.query, self.dataset_ids
            )
            # NOTE: removing b/c I believe this adds a duplicate user query message to the conversation
            # await save_message(conversation_id, "user", self.query)

            # Mark the conversation as having an active query so the
            # poll-based catch-up (Phase 5) reports it as in-flight.
            await _set_query_active(conversation_id, active=True)

            await self.queue.put(f"data: {json.dumps({'type': 'step_detail', 'source': 'execution', 'message': 'Starting query execution...', 'session_id': session_id})}\n\n")
            await self.queue.put(f"data: {json.dumps({'type': 'step_detail', 'source': 'execution', 'message': 'Loading datasets...'})}\n\n")

            datasets = await _load_carnot_datasets(self.dataset_ids)
            for dataset in datasets:
                logger.info(f"Dataset: {dataset.name}, Annotation: {dataset.annotation}, Items: {[item.path for item in dataset.items]}")

            await self.queue.put(f"data: {json.dumps({'type': 'step_detail', 'source': 'execution', 'message': f'Loaded {len(datasets)} dataset(s)'})}\n\n")

            all_files = [
                S3Path(item.path) if item.path.startswith("s3://") else Path(item.path)
                for dataset in datasets
                for item in dataset
            ]
            if not all_files:
                await self.queue.put(f"data: {json.dumps({'type': 'error', 'message': 'No files found in selected datasets'})}\n\n")
                return

            await self.queue.put(f"data: {json.dumps({'type': 'step_detail', 'source': 'execution', 'message': f'Processing {len(all_files)} files...'})}\n\n")

            session_exists = session_id in active_sessions
            if session_exists and set(active_sessions[session_id].get("dataset_ids", [])) != set(self.dataset_ids):
                session_exists = False

            session_dir = Path(BASE_DIR, ".sessions", session_id) if IS_LOCAL_ENV else S3Path(BASE_DIR, ".sessions", session_id)
            file_service.create_dir(str(session_dir))

            await self.queue.put(f"data: {json.dumps({'type': 'step_detail', 'source': 'execution', 'message': 'Preparing data context...'})}\n\n")

            # TODO: remove this and place file check into Carnot context
            if not session_exists:
                text_file_count = 0
                for file_path in all_files:
                    if file_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".zip", ".exe", ".bin"}:
                        continue
                    text_file_count += 1

                if text_file_count == 0:
                    await self.queue.put(f"data: {json.dumps({'type': 'error', 'message': 'No text files found in selected datasets'})}\n\n")
                    return

                await self.queue.put(f"data: {json.dumps({'type': 'step_detail', 'source': 'execution', 'message': f'Processing {text_file_count} text files...'})}\n\n")
            else:
                await self.queue.put(f"data: {json.dumps({'type': 'step_detail', 'source': 'execution', 'message': 'Continuing conversation...'})}\n\n")

            await self.queue.put(f"data: {json.dumps({'type': 'step_detail', 'source': 'execution', 'message': f'Executing query: {self.query}'})}\n\n")

            # setup progress logging and clear old progress log if it exists
            fs = fsspec.filesystem(FILESYSTEM)
            progress_log = str(Path(session_dir, "progress.jsonl") if IS_LOCAL_ENV else str(S3Path(session_dir, "progress.jsonl")))
            if fs.exists(progress_log):
                fs.rm(progress_log, recursive=False)

            logger.info(f"Query: {self.query}")
            logger.info(f"Plan: {json.dumps(self.plan, indent=2)}")
            logger.info(f"Datasets: {[dataset.name for dataset in datasets]}")
            logger.info(f"Cost budget: ${self.cost_budget}" if self.cost_budget else "Cost budget: None")
            
            # Load existing conversation history from database
            conversation = await load_conversation_from_db(self.user_id, self.session_id)

            # Load planning-phase stats so the execution-phase Execution
            # instance can include them in the final ExecutionStats.
            # Prefer the in-memory cache (set by PlanningStreamer); fall
            # back to the DB in case the server restarted between plan
            # and execute.
            cached_planning_stats = None
            session_data = active_sessions.get(session_id)
            if session_data:
                cached_planning_stats = session_data.get("planning_stats")
            if cached_planning_stats is None:
                cached_planning_stats = await _load_planning_stats_from_db(session_id)

            # create execution and execute plan
            exec_instance = carnot.Execution(
                query=self.query,
                datasets=datasets,
                plan=self.plan,
                planning_stats=cached_planning_stats,
                conversation=conversation,
                tools=[],
                memory=None,
                indices=[],
                llm_config=self.user_config,
                progress_log_file=progress_log,
                cost_budget=self.cost_budget,
                storage=_build_storage(),
            )

            await self.queue.put(f"data: {json.dumps({'type': 'step_detail', 'source': 'execution', 'message': 'Running Carnot query processor...'})}\n\n")

            output_log = str(Path(session_dir, "output.txt") if IS_LOCAL_ENV else str(S3Path(session_dir, "output.txt")))
            if fs.exists(output_log):
                fs.rm(output_log, recursive=False)

            original_stdout = sys.stdout
            original_stderr = sys.stderr

            stdout_capture = OutputCapture(output_log, original_stdout)
            stderr_capture = OutputCapture(output_log, original_stderr)

            current_local_output_path = (
                output_log
                if IS_LOCAL_ENV
                else str(stdout_capture.temp_filepath)
            )

            active_sessions[session_id] = {
                "context": None,
                "last_access": datetime.now(),
                "dataset_ids": self.dataset_ids,
                "session_dir": str(session_dir),
                "local_output_path": current_local_output_path,
            }

            # Use run() with a progress queue to get operator-level progress events.
            # The generator is synchronous; we drive it in a background
            # thread and ferry ExecutionProgress events through a
            # thread-safe queue so we can push them to the SSE stream.
            exec_progress_queue: queue.Queue = queue.Queue()

            def run_query_with_capture():
                local_output_path = None
                try:
                    sys.stdout = stdout_capture
                    sys.stderr = stderr_capture
                    return exec_instance.run(progress_queue=exec_progress_queue)

                finally:
                    sys.stdout = original_stdout
                    sys.stderr = original_stderr
                    stdout_capture.close()
                    stderr_capture.close()

                    # Note: We only need to upload one file since both stdout and stderr write to the same file in OutputCapture
                    local_output_path = stdout_capture.temp_filepath
                    if not IS_LOCAL_ENV and local_output_path and Path(local_output_path).exists():
                        s3_fs = fsspec.filesystem("s3")
                        try:
                            s3_fs.put(str(local_output_path), output_log)
                        except Exception as e:
                            original_stderr.write(f"Error uploading output.txt to S3: {e}\n")
                        finally:
                            # clean up the local temp file
                            Path(local_output_path).unlink(missing_ok=True)

            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(run_query_with_capture)

            # Accumulate execution step dicts for persistence (§ 7 of design doc)
            accumulated_exec_steps = []

            # Poll for execution progress events while the thread is working
            while not future.done():
                try:
                    progress_dict = exec_progress_queue.get(timeout=0.25)
                    step_dict = {'type': 'step_detail', 'source': 'execution', **progress_dict}
                    accumulated_exec_steps.append(step_dict)
                    await self.queue.put(f"data: {json.dumps(step_dict)}\n\n")
                    await persist_query_event(
                        conversation_id, session_id,
                        event_type="step_detail", source="execution",
                        payload=step_dict,
                        step_cost_usd=progress_dict.get("step_cost_usd"),
                    )
                except queue.Empty:
                    await asyncio.sleep(0.1)

            # Drain any remaining progress events
            while not exec_progress_queue.empty():
                try:
                    progress_dict = exec_progress_queue.get_nowait()
                    step_dict = {'type': 'step_detail', 'source': 'execution', **progress_dict}
                    accumulated_exec_steps.append(step_dict)
                    await self.queue.put(f"data: {json.dumps(step_dict)}\n\n")
                    await persist_query_event(
                        conversation_id, session_id,
                        event_type="step_detail", source="execution",
                        payload=step_dict,
                        step_cost_usd=progress_dict.get("step_cost_usd"),
                    )
                except queue.Empty:
                    break

            items, answer, execution_stats = future.result()
            executor.shutdown(wait=False)

            # --- persist execution step_group to DB ---------------------------
            if accumulated_exec_steps:
                try:
                    await save_message(
                        conversation_id, "agent",
                        json.dumps(accumulated_exec_steps),
                        message_type="step_group",
                    )
                except Exception:
                    logger.exception("Failed to save execution step_group")

            await self.queue.put(f"data: {json.dumps({'type': 'step_detail', 'source': 'execution', 'message': 'Processing results...'})}\n\n")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"query_results_{timestamp}.csv"
            results_dir = Path(BASE_DIR, ".results") if IS_LOCAL_ENV else S3Path(BASE_DIR, ".results")
            file_service.create_dir(str(results_dir))
            csv_path = results_dir / csv_filename
            fs = fsspec.filesystem(FILESYSTEM)

            result_message_id = None
            try:
                # First, save to our timestamp-based file
                df = pd.DataFrame(items)
                with fs.open(str(csv_path), 'w', encoding='utf-8') as f:
                    df.to_csv(f, index=False)

                if df.empty and (not answer or answer.strip() == ""):
                    message_text = "No results found for your query."
                    result_message_id = await save_message(conversation_id, "agent", message_text, "result")
                    await self.queue.put(f"data: {json.dumps({'type': 'result', 'message': message_text, 'session_id': session_id})}\n\n")
                elif df.empty:
                    message_text = (
                        "Query completed successfully!\n\n"
                        f"Answer Text: {answer}\n\n"
                        "No tabular results found."
                    )
                    result_message_id = await save_message(conversation_id, "agent", message_text, "result")
                    await self.queue.put(f"data: {json.dumps({'type': 'result', 'message': message_text, 'session_id': session_id})}\n\n")
                elif not answer or answer.strip() == "":
                    body = str(df.head())
                    message_text = (
                        "Query completed successfully!\n\n"
                        f"Found {len(df)} result(s):\n\n{body}\n..."
                    )
                    result_message_id = await save_message(conversation_id, "agent", message_text, "result", csv_filename, len(df))
                    await self.queue.put(f"data: {json.dumps({'type': 'result', 'message': message_text, 'csv_file': csv_filename, 'row_count': len(df), 'session_id': session_id})}\n\n")
                else:
                    body = str(df.head())
                    message_text = (
                        "Query completed successfully!\n\n"
                        f"Answer Text: {answer}\n\n"
                        f"Found {len(df)} result(s):\n\n{body}\n..."
                    )
                    result_message_id = await save_message(conversation_id, "agent", message_text, "result", csv_filename, len(df))
                    await self.queue.put(f"data: {json.dumps({'type': 'result', 'message': message_text, 'csv_file': csv_filename, 'row_count': len(df), 'session_id': session_id})}\n\n")

            except Exception as exc:
                logger.exception("Error processing query results")
                error_msg = f"Error processing results: {exc}"
                result_message_id = await save_message(conversation_id, "agent", error_msg, "error")
                await self.queue.put(f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n")

            # Emit execution stats before the final done event
            if execution_stats is not None:
                try:
                    stats_payload = execution_stats.to_summary_dict()
                    await self.queue.put(
                        f"data: {json.dumps({'type': 'execution_stats', **stats_payload})}\n\n"
                    )
                    await persist_query_event(
                        conversation_id, session_id,
                        event_type="execution_stats", source="execution",
                        payload={'type': 'execution_stats', **stats_payload},
                        step_cost_usd=stats_payload.get("total_cost_usd"),
                    )
                except Exception:
                    logger.exception("Failed to serialize execution stats")

                # Persist execution stats to DB as a new row
                try:
                    await save_step_stats(
                        conversation_id=conversation_id,
                        session_id=session_id,
                        step_type="execute",
                        stats_dict=stats_payload,
                        query=self.query,
                        message_id=result_message_id,
                    )
                except Exception:
                    logger.exception("Failed to save execution stats to DB")

            done_payload = {'type': 'done', 'message': 'Query execution complete'}
            await self.queue.put(f"data: {json.dumps(done_payload)}\n\n")
            if conversation_id is not None:
                await persist_query_event(
                    conversation_id, session_id,
                    event_type="done", source="execution",
                    payload=done_payload,
                )

        except Exception as exc:
            logger.exception("Query execution failed")
            error_msg = f"Error executing query: {exc}"
            if conversation_id is not None:
                try:
                    await save_message(conversation_id, "agent", error_msg, "error")
                except Exception:
                    logger.exception("Failed to save error message")
            error_payload = {'type': 'error', 'message': error_msg}
            await self.queue.put(f"data: {json.dumps(error_payload)}\n\n")
            if conversation_id is not None:
                await persist_query_event(
                    conversation_id, session_id,
                    event_type="error", source="execution",
                    payload=error_payload,
                )
        finally:
            # Clear the active flag so poll-based catch-up sees the
            # query as complete regardless of success or failure.
            if conversation_id is not None:
                await _set_query_active(conversation_id, active=False)
            # Signal the consumer in stream_response_iterator to stop iterating
            await self.queue.put(None)

    async def stream_response_iterator(self):
        """The async iterator that FastAPI's StreamingResponse will consume."""
        # Start both the heartbeat and the query logic tasks
        self.heartbeat_task = asyncio.create_task(self._heartbeat_sender())
        self.query_task = asyncio.create_task(self._run_query_logic())

        try:
            # Yield items from the queue until the query task signals completion (sends None)
            while True:
                item = await self.queue.get()
                if item is None:
                    break
                yield item
        finally:
            # Ensure both tasks are cleaned up when the stream is complete or breaks
            self.heartbeat_task.cancel()
            self.query_task.cancel()
            # Wait for tasks to finish cancelling to avoid resource leakage
            await asyncio.gather(self.heartbeat_task, self.query_task, return_exceptions=True)


class PlanningStreamer:
    """Encapsulates plan generation with SSE streaming and a heartbeat.

    Uses the same queue + heartbeat pattern as :class:`QueryExecutionStreamer`
    so the connection stays alive during long-running LLM calls.  Progress
    events emitted by :meth:`Execution.plan_stream` are forwarded to the
    SSE stream so the frontend can keep the user informed.
    """

    def __init__(
        self,
        query: str,
        dataset_ids: list[int],
        user_id: str,
        session_id: str,
        user_config: dict,
        cost_budget: float | None = None,
    ):
        self.query = query
        self.dataset_ids = dataset_ids
        self.user_id = user_id
        self.session_id = session_id
        self.user_config = user_config
        self.cost_budget = cost_budget
        self.queue: asyncio.Queue = asyncio.Queue()
        self.heartbeat_task = None
        self.plan_task = None

    async def _heartbeat_sender(self):
        """Send periodic keep-alive comments on the SSE stream."""
        try:
            while True:
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                await self.queue.put(":keep-alive\n\n")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Planning heartbeat sender failed: {e}")

    async def _run_plan_logic(self):
        """Run plan generation in a thread and push SSE events to the queue."""
        conversation_id = None
        try:
            # --- load datasets -------------------------------------------------
            async with AsyncSessionLocal() as db:
                # Validate user LLM config
                user_config = await get_user_llm_config(db, self.user_id)
                if not user_config:
                    await self.queue.put(
                        f"data: {json.dumps({'type': 'error', 'message': 'No LLM API keys found for this user. Please go to the Settings page to configure your keys.'})}\n\n"
                    )
                    return

            try:
                datasets = await _load_carnot_datasets(self.dataset_ids)
            except HTTPException as exc:
                await self.queue.put(
                    f"data: {json.dumps({'type': 'error', 'message': exc.detail})}\n\n"
                )
                return

            # --- conversation bookkeeping --------------------------------------
            conversation_id = await get_or_create_conversation(
                self.user_id, self.session_id, self.query, self.dataset_ids
            )

            # Mark conversation as having an active query so the poll-based
            # catch-up (Phase 5) knows the stream is still running.
            await _set_query_active(conversation_id, active=True)

            conversation = await load_conversation_from_db(self.user_id, self.session_id)
            await save_message(conversation_id, "user", self.query, cost_budget=self.cost_budget)

            # Propagate query-derived title to conversation + workspace
            title = f"{self.query[:50]}..." if len(self.query) > 50 else self.query
            await update_conversation_and_workspace_title(conversation_id, title)

            await self.queue.put(
                f"data: {json.dumps({'type': 'step_detail', 'source': 'planning', 'message': 'Starting plan generation…', 'phase': 'logical_plan', 'session_id': self.session_id})}\n\n"
            )

            # --- build Execution and run plan_stream in a thread ---------------
            exec_instance = carnot.Execution(
                query=self.query,
                datasets=datasets,
                conversation=conversation,
                tools=[],
                memory=None,
                indices=[],
                llm_config=self.user_config,
                cost_budget=self.cost_budget,
                storage=_build_storage(),
            )

            # plan_stream() is a synchronous generator; we drive it from a
            # background thread so the event loop stays free for heartbeats.
            # We use a thread-safe queue so progress events can be pushed to
            # the SSE stream in real time while the generator runs.
            plan_progress_queue: queue.Queue = queue.Queue()

            def _run_plan_with_progress():
                """Run plan, which pushes all progress to the sync queue."""
                nl_plan, logical_plan = exec_instance.plan(
                    progress_queue=plan_progress_queue,
                )
                # Retrieve planning stats set by plan_stream()
                planning_stats = getattr(exec_instance, "_planning_stats", None)
                return nl_plan, logical_plan, planning_stats

            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(_run_plan_with_progress)

            # accumulate step dicts for persistence
            accumulated_steps = []

            # Poll for progress events while the thread is working
            while not future.done():
                try:
                    progress_dict = plan_progress_queue.get(timeout=0.25)
                    step_dict = {'type': 'step_detail', 'source': 'planning', **progress_dict}
                    accumulated_steps.append(step_dict)
                    await self.queue.put(
                        f"data: {json.dumps(step_dict)}\n\n"
                    )
                    await persist_query_event(
                        conversation_id, self.session_id,
                        event_type="step_detail", source="planning",
                        payload=step_dict,
                        step_cost_usd=progress_dict.get("step_cost_usd"),
                    )
                except queue.Empty:
                    await asyncio.sleep(0.1)

            # Drain any remaining progress events
            while not plan_progress_queue.empty():
                try:
                    progress_dict = plan_progress_queue.get_nowait()
                    step_dict = {'type': 'step_detail', 'source': 'planning', **progress_dict}
                    accumulated_steps.append(step_dict)
                    await self.queue.put(
                        f"data: {json.dumps(step_dict)}\n\n"
                    )
                    await persist_query_event(
                        conversation_id, self.session_id,
                        event_type="step_detail", source="planning",
                        payload=step_dict,
                        step_cost_usd=progress_dict.get("step_cost_usd"),
                    )
                except queue.Empty:
                    break

            nl_plan, logical_plan, planning_stats = future.result()
            executor.shutdown(wait=False)

            # --- persist planning step_group to DB ----------------------------
            if accumulated_steps:
                try:
                    await save_message(
                        conversation_id, "agent",
                        json.dumps(accumulated_steps),
                        message_type="step_group",
                    )
                except Exception:
                    logger.exception("Failed to save planning step_group")

            # --- save plan messages to DB first (need message_id for stats) ----
            nl_plan_message_id = None
            if nl_plan:
                nl_plan_message_id = await save_message(
                    conversation_id, "agent", nl_plan,
                    message_type="natural-language-plan",
                )
            if logical_plan:
                plan_json = json.dumps(logical_plan, indent=2)
                await save_message(
                    conversation_id, "agent", plan_json,
                    message_type="logical-plan",
                )

            # --- cache planning_stats for the execution phase ----------------
            # QueryExecutionStreamer runs on a separate Execution instance;
            # stash the PhaseStats object so it can be threaded through.
            if planning_stats is not None:
                session_data = active_sessions.get(self.session_id)
                if session_data is not None:
                    session_data["planning_stats"] = planning_stats
                else:
                    active_sessions[self.session_id] = {
                        "planning_stats": planning_stats,
                        "last_access": datetime.now(),
                    }

            # --- persist planning stats to DB ---------------------------------
            planning_stats_dict = None
            if planning_stats is not None:
                try:
                    from carnot.core.models import ExecutionStats, PhaseStats
                    # Build a partial ExecutionStats with only the planning phase
                    partial_stats = ExecutionStats(
                        query=self.query,
                        planning=planning_stats,
                        execution=PhaseStats(phase="execution"),
                    )
                    planning_stats_dict = partial_stats.to_summary_dict()
                    await save_step_stats(
                        conversation_id=conversation_id,
                        session_id=self.session_id,
                        step_type="plan",
                        stats_dict=planning_stats_dict,
                        query=self.query,
                        message_id=nl_plan_message_id,
                    )
                except Exception:
                    logger.exception("Failed to save planning stats to DB")

            logger.info(
                f"Generated plan for session {self.session_id}:\n{nl_plan}\n"
                f"{json.dumps(logical_plan, indent=2) if logical_plan else '(none)'}"
            )

            # --- send planning_stats SSE event ---------------------------------
            if planning_stats_dict is not None:
                try:
                    planning_stats_event = {'type': 'planning_stats', **planning_stats_dict}
                    await self.queue.put(
                        f"data: {json.dumps(planning_stats_event)}\n\n"
                    )
                    await persist_query_event(
                        conversation_id, self.session_id,
                        event_type="planning_stats", source="planning",
                        payload=planning_stats_event,
                        step_cost_usd=planning_stats_dict.get("total_cost_usd"),
                    )
                except Exception:
                    logger.exception("Failed to emit planning_stats SSE event")

            # --- send final plan_complete event --------------------------------
            plan_complete_payload = {'type': 'plan_complete', 'natural_language_plan': nl_plan, 'plan': logical_plan, 'session_id': self.session_id}
            await self.queue.put(
                f"data: {json.dumps(plan_complete_payload)}\n\n"
            )
            if conversation_id is not None:
                await persist_query_event(
                    conversation_id, self.session_id,
                    event_type="plan_complete", source="planning",
                    payload=plan_complete_payload,
                )
                # Planning stream is done; clear the active flag.  If the
                # user approves the plan, QueryExecutionStreamer will re-set
                # it to True when execution begins.
                await _set_query_active(conversation_id, active=False)

        except Exception as exc:
            logger.exception("Plan generation failed")
            error_msg = f"Error generating plan: {exc}"
            if conversation_id is not None:
                try:
                    await save_message(conversation_id, "agent", error_msg, "error")
                except Exception:
                    logger.exception("Failed to save error message")
            error_payload = {'type': 'error', 'message': error_msg}
            await self.queue.put(
                f"data: {json.dumps(error_payload)}\n\n"
            )
            if conversation_id is not None:
                await persist_query_event(
                    conversation_id, self.session_id,
                    event_type="error", source="planning",
                    payload=error_payload,
                )
                # Planning failed — no execution phase will follow, so clear
                # the active flag so poll-based catch-up sees it as complete.
                await _set_query_active(conversation_id, active=False)
        finally:
            await self.queue.put(None)

    async def stream_response_iterator(self):
        """Async iterator consumed by FastAPI's ``StreamingResponse``."""
        self.heartbeat_task = asyncio.create_task(self._heartbeat_sender())
        self.plan_task = asyncio.create_task(self._run_plan_logic())

        try:
            while True:
                item = await self.queue.get()
                if item is None:
                    break
                yield item
        finally:
            self.heartbeat_task.cancel()
            self.plan_task.cancel()
            await asyncio.gather(self.heartbeat_task, self.plan_task, return_exceptions=True)


class QueryRequest(BaseModel):
    query: str
    dataset_ids: list[int]
    session_id: str
    plan: dict | None = None
    cost_budget: float | None = None  # Maximum dollar amount user is willing to spend


def cleanup_old_sessions() -> None:
    now = datetime.now()
    expired_sessions = [
        session_id
        for session_id, session_data in active_sessions.items()
        if now - session_data["last_access"] > SESSION_TIMEOUT
    ]
    for session_id in expired_sessions:
        active_sessions.pop(session_id, None)


async def get_or_create_conversation(
    user_id: str, session_id: str, query: str, dataset_ids: list[int]
) -> int:
    """Find the conversation for *session_id*, creating workspace + conversation if absent.

    Normally the frontend has already called ``POST /workspaces/`` which
    pre-creates the workspace and its first conversation.  This helper
    acts as a safety net so that direct ``/plan`` or ``/execute`` calls
    still work.

    Requires:
        - ``session_id`` is non-empty.

    Returns:
        The ``Conversation.id`` of the (possibly just-created) conversation.

    Raises:
        None.
    """
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Conversation).where(
                Conversation.session_id == session_id,
                Conversation.user_id == user_id,
            )
        )
        conversation = result.scalar_one_or_none()
        if conversation:
            return conversation.id

        # Fallback: create workspace + conversation atomically
        title = f"{query[:50]}..." if len(query) > 50 else query
        dataset_ids_str = ",".join(map(str, dataset_ids))

        workspace = Workspace(
            user_id=user_id,
            session_id=session_id,
            title=title,
            dataset_ids=dataset_ids_str,
        )
        db.add(workspace)
        await db.flush()  # get workspace.id

        conversation = Conversation(
            workspace_id=workspace.id,
            user_id=user_id,
            session_id=session_id,
            title=title,
        )
        db.add(conversation)
        await db.commit()
        await db.refresh(conversation)
        return conversation.id


async def save_message(
    conversation_id: int,
    role: str,
    content: str,
    message_type: str | None = None,
    csv_file: str | None = None,
    row_count: int | None = None,
    cost_budget: float | None = None,
) -> int:
    """Persist a message and return its ``Message.id``."""
    async with AsyncSessionLocal() as db:
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            type=message_type,
            csv_file=csv_file,
            row_count=row_count,
            cost_budget=cost_budget,
        )
        db.add(message)

        result = await db.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        conversation = result.scalar_one_or_none()
        if conversation:
            conversation.updated_at = datetime.now(timezone.utc)  # noqa: UP017

        await db.commit()
        await db.refresh(message)
        return message.id


async def update_conversation_and_workspace_title(
    conversation_id: int, title: str
) -> None:
    """Set the conversation title and propagate to its workspace if still default.

    Called after the first user message is saved so that both the
    conversation and workspace get a meaningful title derived from the
    query text.

    Requires:
        - ``conversation_id`` references a valid conversation.

    Returns:
        None.

    Raises:
        None.  Silently does nothing if the conversation is not found.
    """
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        conversation = result.scalar_one_or_none()
        if not conversation:
            return

        conversation.title = title

        # Propagate to workspace if title is still the default
        ws_result = await db.execute(
            select(Workspace).where(Workspace.id == conversation.workspace_id)
        )
        workspace = ws_result.scalar_one_or_none()
        if workspace and workspace.title == "Untitled Workspace":
            workspace.title = title

        await db.commit()


async def _load_planning_stats_from_db(session_id: str):
    """Load planning ``PhaseStats`` from the most recent plan row for *session_id*.

    Falls back to the DB when the in-memory ``active_sessions`` cache has
    been evicted (e.g. after a server restart between plan and execute).

    Requires:
        - ``session_id`` is a non-empty string.

    Returns:
        A ``PhaseStats`` instance reconstructed from the persisted
        ``stats_json``, or ``None`` if no plan row exists.

    Raises:
        None.  Database errors are logged and ``None`` is returned.
    """
    from carnot.core.models import OperatorStats, PhaseStats

    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(QueryStats)
                .where(QueryStats.session_id == session_id)
                .where(QueryStats.step_type == "plan")
                .order_by(QueryStats.id.desc())
                .limit(1)
            )
            row = result.scalar_one_or_none()
            if row is None or row.stats_json is None:
                return None

            planning_dict = row.stats_json.get("planning", {})
            # Reconstruct PhaseStats from the persisted summary dict.
            op_stats_dicts = planning_dict.get("operator_stats", [])
            op_stats = [OperatorStats.model_validate(d) for d in op_stats_dicts]
            return PhaseStats(
                phase="planning",
                wall_clock_secs=planning_dict.get("wall_clock_secs", 0.0),
                operator_stats=op_stats,
            )
    except Exception:
        logger.exception("Failed to load planning stats from DB for session %s", session_id)
        return None


async def save_step_stats(
    conversation_id: int,
    session_id: str,
    step_type: str,
    stats_dict: dict,
    query: str | None = None,
    message_id: int | None = None,
) -> int:
    """Insert a ``QueryStats`` row for a completed plan or execute step.

    ``query_iteration`` groups all plan revisions and the subsequent
    execution into a single logical cycle.  The iteration number only
    advances **after** an execute step closes the current cycle:

    - Plan 1 → iteration 1  (first plan, no prior execute)
    - Plan 2 → iteration 1  (revision, still same cycle)
    - Execute → iteration 1  (closes this cycle)
    - Plan 3 → iteration 2  (new cycle begins)
    - Execute → iteration 2  (closes second cycle)

    Requires:
        - ``conversation_id`` references a valid conversation.
        - ``step_type`` is ``"plan"`` or ``"execute"``.
        - ``stats_dict`` is the output of ``ExecutionStats.to_summary_dict()``.
          For plan steps, the ``"planning"`` sub-dict is used for metrics.
          For execute steps, the ``"execution"`` sub-dict is used.

    Returns:
        The ``QueryStats.id`` of the newly created row.

    Raises:
        None.  Database errors propagate to the caller.
    """
    # Extract the relevant phase metrics based on step type
    phase = stats_dict.get("planning", {}) if step_type == "plan" else stats_dict.get("execution", {})

    async with AsyncSessionLocal() as db:
        from sqlalchemy import func as sa_func

        # The current iteration = (number of completed execute steps) + 1.
        # All plan steps before the next execute share this iteration,
        # and the execute step that closes the cycle also uses it.
        exec_count_result = await db.execute(
            select(sa_func.count(QueryStats.id))
            .where(QueryStats.conversation_id == conversation_id)
            .where(QueryStats.step_type == "execute")
        )
        current_iteration = exec_count_result.scalar() + 1

        row = QueryStats(
            conversation_id=conversation_id,
            session_id=session_id,
            query=query,
            query_iteration=current_iteration,
            step_type=step_type,
            message_id=message_id,
            cost_usd=phase.get("total_cost_usd"),
            wall_clock_secs=phase.get("wall_clock_secs"),
            input_tokens=phase.get("total_input_tokens"),
            output_tokens=phase.get("total_output_tokens"),
            stats_json=stats_dict,
        )
        db.add(row)
        await db.commit()
        await db.refresh(row)
        return row.id


async def persist_query_event(
    conversation_id: int,
    session_id: str,
    event_type: str,
    payload: dict,
    source: str | None = None,
    step_cost_usd: float | None = None,
) -> None:
    """Append a single SSE event to the ``query_events`` table.

    This is called from both ``PlanningStreamer`` and
    ``QueryExecutionStreamer`` as events are emitted so that workspace
    cost can be reconstructed after a page navigation.

    Requires:
        - ``conversation_id`` references a valid conversation.
        - ``event_type`` is a non-empty string (e.g. ``"step_detail"``).
        - ``payload`` is a JSON-serializable dict.

    Returns:
        None.

    Raises:
        None.  Database errors are logged but swallowed so they never
        break the SSE stream.
    """
    try:
        async with AsyncSessionLocal() as db:
            row = QueryEvent(
                conversation_id=conversation_id,
                session_id=session_id,
                event_type=event_type,
                source=source,
                payload=payload,
                step_cost_usd=step_cost_usd,
            )
            db.add(row)
            await db.commit()
    except Exception:
        logger.exception(
            "Failed to persist query event (type=%s, conv=%s)",
            event_type,
            conversation_id,
        )


async def _set_query_active(conversation_id: int, *, active: bool) -> None:
    """Toggle the ``is_query_active`` flag on a conversation.

    Called at the start of ``PlanningStreamer`` (active=True) and in
    both streamers' terminal paths (active=False) so the poll-based
    catch-up endpoint can report whether a query is still running.

    Requires:
        - ``conversation_id`` references a valid conversation.

    Returns:
        None.

    Raises:
        None.  Database errors are logged but swallowed.
    """
    try:
        async with AsyncSessionLocal() as db:
            conv = await db.get(Conversation, conversation_id)
            if conv:
                conv.is_query_active = active
                await db.commit()
    except Exception:
        logger.exception(
            "Failed to set is_query_active=%s for conversation %s",
            active,
            conversation_id,
        )


async def load_conversation_from_db(
    user_id: str,
    session_id: str,
) -> carnot.Conversation | None:
    """
    Load conversation history from the database and construct a carnot.Conversation object.
    
    Returns None if the conversation doesn't exist.
    """
    async with AsyncSessionLocal() as db:
        # Get the conversation record
        conv_result = await db.execute(
            select(Conversation).where(
                Conversation.session_id == session_id,
                Conversation.user_id == user_id,
            )
        )
        db_conversation = conv_result.scalar_one_or_none()
        
        if not db_conversation:
            return None
        
        # Get all messages for this conversation, ordered by creation time
        messages_result = await db.execute(
            select(Message)
            .where(Message.conversation_id == db_conversation.id)
            .order_by(Message.created_at)
        )
        db_messages = messages_result.scalars().all()
        
        # Convert database messages to conversation message format
        conversation_messages = []
        for msg in db_messages:
            # Only include user and agent messages in the conversation history
            if msg.role in ["user", "agent"]:
                message_dict = {
                    "role": msg.role,
                    "content": msg.content,
                }
                if msg.type:
                    message_dict["type"] = msg.type
                conversation_messages.append(message_dict)
        
        # Get dataset_ids from the parent workspace
        dataset_ids = []
        ws_result = await db.execute(
            select(Workspace).where(Workspace.id == db_conversation.workspace_id)
        )
        workspace = ws_result.scalar_one_or_none()
        if workspace and workspace.dataset_ids:
            dataset_ids = workspace.dataset_ids.split(",")
        
        # Create and return carnot.Conversation object
        return carnot.Conversation(
            user_id=user_id,
            session_id=session_id,
            title=db_conversation.title or "",
            dataset_ids=dataset_ids,
            messages=conversation_messages,
        )

@router.post("/plan")
async def plan_query(
    request: QueryRequest,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Generate a logical execution plan with streaming progress updates.

    Returns an SSE stream that emits:
    - ``step_detail`` events as the planner/data-discovery agent work.
    - A single ``plan_complete`` event with the final NL plan and logical plan.
    - An ``error`` event if anything goes wrong.

    Supports multi-turn conversations via *session_id*.
    """
    if not request.dataset_ids:
        raise HTTPException(status_code=400, detail="At least one dataset must be selected")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Pre-flight check: make sure the user has LLM keys configured.
    # We do this *before* entering the stream so we can return a clean
    # HTTP 400 instead of an SSE error event.
    user_config = await get_user_llm_config(db, user_id)
    if not user_config:
        raise HTTPException(
            status_code=400,
            detail={
                "type": "API_KEY_MISSING",
                "message": "No LLM API keys found for this user.",
            },
        )

    streamer = PlanningStreamer(
        query=request.query,
        dataset_ids=request.dataset_ids,
        user_id=user_id,
        session_id=request.session_id,
        user_config=user_config,
        cost_budget=request.cost_budget,
    )

    return StreamingResponse(
        streamer.stream_response_iterator(),
        media_type="text/event-stream",
    )

@router.post("/execute")
async def execute_query(
    request: QueryRequest,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """
    Execute a Carnot query on selected datasets with streaming progress.
    Supports multi-turn conversations via session_id.
    """
    if not request.dataset_ids:
        raise HTTPException(status_code=400, detail="At least one dataset must be selected")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # retrieve user LLM config
    user_config = await get_user_llm_config(db, user_id)
    if not user_config:
        raise HTTPException(
            status_code=400,
            detail={
                "type": "API_KEY_MISSING",
                "message": "No LLM API keys found for this user.",
            }
        )

    # Instantiate the streamer class
    logger.info(f"Request.plan: {request.plan}")
    logger.info(f"Request.cost_budget: {request.cost_budget}")
    streamer = QueryExecutionStreamer(
        request.query, request.plan, request.dataset_ids, user_id, request.session_id, user_config, request.cost_budget,
    )

    return StreamingResponse(
        # Return the async iterator method from the class
        streamer.stream_response_iterator(),
        media_type="text/event-stream"
    )


@router.get("/events/{conversation_id}")
async def get_query_events(
    conversation_id: int,
    since_id: int = 0,
    limit: int = 200,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return query events persisted since ``since_id``.

    Used by the frontend's poll-based catch-up hook to reconstruct
    streaming state after navigating away and back to a workspace
    whose query is still in-flight.

    Requires:
        - ``conversation_id`` references a conversation owned by the
          authenticated user.
        - ``since_id`` >= 0.

    Returns:
        ``{"events": [...], "is_complete": bool}`` where ``is_complete``
        is ``True`` if the query has finished (``is_query_active`` is
        ``False``).

    Raises:
        HTTPException(404) if the conversation does not exist or is not
        owned by the authenticated user.
    """
    conv = await db.get(Conversation, conversation_id)
    if not conv or conv.user_id != user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")

    result = await db.execute(
        select(QueryEvent)
        .where(QueryEvent.conversation_id == conversation_id)
        .where(QueryEvent.id > since_id)
        .order_by(QueryEvent.id)
        .limit(limit)
    )
    rows = result.scalars().all()

    events = [
        {
            "id": row.id,
            "event_type": row.event_type,
            "source": row.source,
            "payload": row.payload,
            "step_cost_usd": row.step_cost_usd,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
        for row in rows
    ]

    return {"events": events, "is_complete": not conv.is_query_active}


@router.get("/progress/{session_id}")
async def get_progress(session_id: str, since_timestamp: str | None = None):
    """
    Get progress events for a session, optionally filtering by timestamp.
    """
    try:
        fs = fsspec.filesystem(FILESYSTEM)
        session_dir = str(Path(BASE_DIR, ".sessions", session_id) if IS_LOCAL_ENV else S3Path(BASE_DIR, ".sessions", session_id))
        progress_log = str(Path(session_dir, "progress.jsonl") if IS_LOCAL_ENV else S3Path(session_dir, "progress.jsonl"))
        if not fs.exists(progress_log):
            return {"events": []}

        events = []
        with fs.open(progress_log, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        event = json.loads(line)
                        # Filter by timestamp if provided
                        if since_timestamp is None or event['timestamp'] > since_timestamp:
                            events.append(event)
                    except json.JSONDecodeError:
                        continue

        return {"events": events}
    except Exception as e:
        logger.error(f"Error reading progress log: {e}")
        return {"events": [], "error": str(e)}


@router.get("/output/{session_id}")
async def get_output(session_id: str, last_line: int = 0):
    """
    Get terminal output for a session, returns lines after last_line
    """
    # Check if the session is currently active/running
    if session_id in active_sessions and "local_output_path" in active_sessions[session_id]:
        # If active, read from the local file path
        output_file = active_sessions[session_id]["local_output_path"]
        fs = fsspec.filesystem("file") # Always use the local filesystem
    else:
        # If not active, read from the final S3/local destination
        fs = fsspec.filesystem(FILESYSTEM)
        session_dir = str(Path(BASE_DIR, ".sessions", session_id) if IS_LOCAL_ENV else S3Path(BASE_DIR, ".sessions", session_id))
        output_file = str(Path(session_dir, "output.txt") if IS_LOCAL_ENV else S3Path(session_dir, "output.txt"))

    if not fs.exists(output_file):
        return {"lines": [], "total_lines": 0}

    with fs.open(output_file, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
        new_lines = all_lines[last_line:]
        return {"lines": new_lines, "total_lines": len(all_lines)}


@router.get("/download/{filename}")
async def download_query_results(filename: str):
    # Security: only allow downloading CSV files from backend directory
    if not filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid filename - must be a CSV file")

    # Prevent directory traversal
    if "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Get the file path (S3 or local)
    fs = fsspec.filesystem(FILESYSTEM)
    file_path = str(Path(BASE_DIR, ".results", filename) if IS_LOCAL_ENV else S3Path(BASE_DIR, ".results", filename))
    
    if not fs.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    local_path = file_path # Default to the existing path if IS_LOCAL_ENV is true
    should_cleanup = False
    
    # 1. If running remotely (S3), download the file to a local temporary path
    if not IS_LOCAL_ENV:
        try:
            # Create a unique temporary file path on the server
            temp_dir = tempfile.gettempdir()
            local_path = os.path.join(temp_dir, filename)
            
            # Use fsspec to download the S3 file to the local temp path
            s3_fs = fsspec.filesystem("s3")
            s3_fs.get(file_path, local_path)
            
            should_cleanup = True
        except Exception as e:
            logger.error(f"Failed to download S3 file {file_path} to local temp: {e}")
            raise HTTPException(status_code=500, detail="Failed to stage file for download.") from e

    # 2. Return the local path using FileResponse
    # Starlette's FileResponse will now be able to resolve and serve the file.
    return FileResponse(
        path=local_path,
        filename=filename,
        media_type="text/csv",
        # 3. Add cleanup handler for the temporary file
        background=BackgroundTask(lambda: os.remove(local_path)) if should_cleanup else None
    )


# -- Interactive notebook endpoints ------------------------------------------

class NotebookRequest(BaseModel):
    query: str
    dataset_ids: list[int]
    session_id: str
    plan: dict
    cost_budget: float | None = None
    workspace_id: int | None = None       # set by frontend for DB persistence
    conversation_id: int | None = None    # set by frontend for DB persistence
    label: str | None = None              # e.g. "Notebook 1"


class CellExecutionRequest(BaseModel):
    notebook_id: str
    cell_id: str
    code: str | None = None  # current code text; if different from
                              # original_code, triggers param re-parse


class AddCellRequest(BaseModel):
    after_cell_id: str | None = None  # null → insert at top
    cell_type: str = "operator"       # "operator" | "dataset" | "reasoning"
    operator_type: str | None = None  # hint for generating initial code
    code: str | None = None           # optional initial code


class MoveCellRequest(BaseModel):
    cell_id: str
    direction: str  # "up" or "down"


def _extract_quoted(code: str, key: str, *, dotall: bool = False) -> str | None:
    """Extract a quoted ``key=value`` from pseudocode.

    Uses a backreference so the closing quote matches the opening
    quote, allowing embedded quotes of the *other* type.
    Triple-quoted strings (``'''`` / ``\"\"\"``) are tried first.

    Requires:
        - *code* is a non-empty string.
        - *key* is a bare keyword name (no special regex characters).

    Returns:
        The captured value string, or ``None`` if no match is found.

    Raises:
        None.
    """
    flags = re.DOTALL if dotall else 0
    # Try triple-quoted first.
    m = re.search(rf"{key}\s*=\s*'''(.*?)'''", code, flags)
    if m:
        return m.group(1)
    m = re.search(rf'{key}\s*=\s*"""(.*?)"""', code, flags)
    if m:
        return m.group(1)
    # Single-quote with backreference.
    m = re.search(rf"{key}\s*=\s*([\"'])(.*?)\1", code, flags)
    return m.group(2) if m else None


def _parse_code_to_params(
    code: str, node_type: str, operator_type: str | None
) -> dict | None:
    """Extract operator params from user-modified cell pseudocode.

    Uses regex patterns to match the known pseudocode formats generated
    by ``PlanNode.to_code()``.  Returns ``None`` if the code cannot be
    parsed (in which case the backend falls back to the existing params).

    Additionally extracts ``_input_dataset_ids`` and
    ``_output_dataset_id`` when the user edits the ``datasets['...']``
    references, enabling dataset rewiring.

    Requires:
        - ``code`` is a non-empty string.
        - ``node_type`` is one of ``"dataset"``, ``"operator"``,
          ``"reasoning"``.

    Returns:
        A dict of extracted params, or ``None`` if parsing fails.
        May contain special keys ``_input_dataset_ids`` (list of str)
        and ``_output_dataset_id`` (str) for dataset rewiring.

    Raises:
        None.  Parse failures are logged and silently ignored.
    """
    try:
        if node_type == "dataset":
            m = re.search(r'carnot\.load_dataset\(["\'](.+?)["\']\)', code)
            if m:
                return {"dataset_name": m.group(1)}
            return None

        if node_type == "reasoning":
            val = _extract_quoted(code, "query", dotall=True)
            if val:
                return {"task": val}
            return None

        # -- common: extract dataset references for rewiring --------
        result: dict = {}

        # Output dataset: datasets['XXX'] = ...
        out_m = re.search(r"datasets\[([\"'])(.*?)\1\]\s*=", code)
        if out_m:
            result["_output_dataset_id"] = out_m.group(2)

        # Input dataset(s): dataset=datasets['XXX'] or left/right
        input_ids: list[str] = []
        for ref_m in re.finditer(
            r"(?:dataset|left|right)\s*=\s*datasets\[([\"'])(.*?)\1\]", code
        ):
            input_ids.append(ref_m.group(2))
        if input_ids:
            result["_input_dataset_ids"] = input_ids

        # -- operator-specific params ------------------------------
        if operator_type == "SemanticFilter":
            val = _extract_quoted(code, "condition", dotall=True)
            if val:
                result["condition"] = val

        elif operator_type == "SemanticMap":
            for key in ("field", "type", "description"):
                val = _extract_quoted(code, key)
                if val is not None:
                    k = "field_desc" if key == "description" else key
                    result[k] = val

        elif operator_type == "SemanticJoin":
            val = _extract_quoted(code, "condition", dotall=True)
            if val:
                result["condition"] = val

        elif operator_type == "SemanticFlatMap":
            for key in ("field", "type", "description"):
                val = _extract_quoted(code, key)
                if val is not None:
                    k = "field_desc" if key == "description" else key
                    result[k] = val

        elif operator_type == "SemanticGroupBy":
            # Parse group_by=['field1', 'field2']
            gby_m = re.search(r'group_by\s*=\s*\[(.+?)\]', code)
            if gby_m:
                gby_names = re.findall(r"[\"']([^\"']+)[\"']", gby_m.group(1))
                result["gby_fields"] = [{"name": n} for n in gby_names]

            # Parse aggregations=['name(func)', ...]
            agg_m = re.search(r'aggregations\s*=\s*\[(.+?)\]', code)
            if agg_m:
                agg_entries = re.findall(r"[\"']([^\"']+)[\"']", agg_m.group(1))
                agg_fields = []
                for entry in agg_entries:
                    # Format: "field_name(func)"
                    parts = re.match(r'^(.+?)\((.+?)\)$', entry.strip())
                    if parts:
                        agg_fields.append({"name": parts.group(1), "func": parts.group(2)})
                if agg_fields:
                    result["agg_fields"] = agg_fields

        elif operator_type == "SemanticTopK":
            val = _extract_quoted(code, "search", dotall=True)
            if val:
                result["search_str"] = val
            m = re.search(r'k\s*=\s*(\d+)', code)
            if m:
                result["k"] = int(m.group(1))

        elif operator_type == "SemanticAgg":
            val = _extract_quoted(code, "task", dotall=True)
            if val:
                result["task"] = val

        elif operator_type == "Code":
            val = _extract_quoted(code, "task", dotall=True)
            if val:
                result["task"] = val.strip()

        elif operator_type == "Limit":
            m = re.search(r'n\s*=\s*(\d+)', code)
            if m:
                result["n"] = int(m.group(1))

        return result if result else None
    except Exception:
        logger.warning("Failed to parse code to params", exc_info=True)
        return None


def _cleanup_old_notebooks() -> None:
    """Evict notebook states that have not been accessed recently."""
    now = datetime.now()
    expired = [
        nid
        for nid, nb in active_notebooks.items()
        if now - nb.last_access > NOTEBOOK_TIMEOUT
    ]
    for nid in expired:
        active_notebooks.pop(nid, None)


async def _persist_notebook_cells(
    notebook_uuid: str,
    cells: list[dict],
    plan_dict: dict | None = None,
) -> None:
    """Update the ``cells_json`` (and optionally ``plan_json``) for a persisted notebook.

    Called after cell execution, add, delete, or move so the DB snapshot
    stays in sync with in-memory state.  When the physical plan itself
    has been mutated (add/delete/move), pass *plan_dict* to keep
    ``plan_json`` in sync as well — this is required for correct
    rehydration after kernel eviction.

    Requires:
        - ``notebook_uuid`` is a valid UUID that may or may not have a DB row.

    Returns:
        None.

    Raises:
        None.  Errors are logged and silently ignored.
    """
    logger.info(
        "[persist] called for notebook=%s cells=%d has_outputs=%s plan_update=%s",
        notebook_uuid,
        len(cells),
        any(c.get("output") for c in cells),
        plan_dict is not None,
    )
    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Notebook).where(Notebook.notebook_uuid == notebook_uuid)
            )
            nb_row = result.scalar_one_or_none()
            if nb_row is None:
                logger.warning("[persist] no DB row found for notebook=%s — skipping", notebook_uuid)
            else:
                # Sanitise for PostgreSQL JSONB storage (strips null
                # bytes, coerces non-JSON types, etc.).
                nb_row.cells_json = jsonb_serializer.sanitize(cells)
                if plan_dict is not None:
                    nb_row.plan_json = jsonb_serializer.sanitize(plan_dict)
                nb_row.updated_at = datetime.now(timezone.utc)  # noqa: UP017
                await db.commit()
                logger.info(
                    "[persist] committed notebook=%s cells_json_bytes=%d",
                    notebook_uuid,
                    len(json.dumps(nb_row.cells_json)),
                )
    except Exception:
        logger.warning("Failed to persist cells_json for notebook %s", notebook_uuid, exc_info=True)


@router.post("/execute-jupyter")
async def execute_jupyter(
    request: NotebookRequest,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return a notebook-ready physical plan for a query.

    Creates an :class:`Execution` instance, calls
    ``get_physical_plan()``, persists a :class:`NotebookState`, and
    returns the cell descriptors with a ``notebook_id``.  No LLM work is
    performed — this is a synchronous JSON response.

    Requires:
        - ``request.plan`` is a valid logical plan dict.
        - ``request.dataset_ids`` is non-empty.

    Returns:
        ``{ notebook_id, query, cells }``

    Raises:
        HTTPException 400: if inputs are invalid or LLM keys are missing.
    """
    _cleanup_old_notebooks()

    if not request.dataset_ids:
        raise HTTPException(status_code=400, detail="At least one dataset must be selected")
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if not request.plan:
        raise HTTPException(status_code=400, detail="A logical plan is required")

    user_config = await get_user_llm_config(db, user_id)
    if not user_config:
        raise HTTPException(
            status_code=400,
            detail={
                "type": "API_KEY_MISSING",
                "message": "No LLM API keys found for this user.",
            },
        )

    # Load datasets from DB
    datasets = await _load_carnot_datasets(request.dataset_ids)

    # Load conversation so the _plan setter can extract original_query
    # for the reasoning node (needed when query is a follow-up).
    conversation = await load_conversation_from_db(user_id, request.session_id)

    storage = _build_storage()
    exec_instance = carnot.Execution(
        query=request.query,
        datasets=datasets,
        plan=request.plan,
        conversation=conversation,
        tools=[],
        memory=None,
        indices=[],
        llm_config=user_config,
        cost_budget=request.cost_budget,
        storage=storage,
    )

    # Build the notebook state from the Execution's physical plan.
    # No LLM calls happen here — only plan serialisation.
    notebook_id = str(uuid4())
    physical_plan = exec_instance._physical_plan

    # Seed datasets_store with the user-supplied datasets keyed by name
    # so that dataset-load nodes can find them.
    initial_store: dict[str, carnot.Dataset] = {
        ds.name: ds for ds in datasets
    }

    # Initialise all cell statuses to "pending".
    cell_statuses = {
        node.node_id: "pending"
        for node in physical_plan.topo_order()
    }

    nb = NotebookState(
        notebook_id=notebook_id,
        query=request.query,
        physical_plan=physical_plan,
        datasets_store=initial_store,
        cell_statuses=cell_statuses,
        llm_config=user_config,
        user_id=user_id,
        session_id=request.session_id,
        dataset_ids=request.dataset_ids,
        cost_budget=request.cost_budget,
        storage=storage,
        execution=exec_instance,
    )
    active_notebooks[notebook_id] = nb

    cells = nb.get_cells()

    # ── Persist notebook row to DB ────────────────────────
    # Resolve workspace_id + conversation_id from the request or via
    # session_id lookup so the notebook survives page refresh.
    ws_id = request.workspace_id
    conv_id = request.conversation_id
    label = request.label or "Notebook"

    if ws_id is None or conv_id is None:
        # Fallback: look up workspace/conversation by session_id
        result = await db.execute(
            select(Conversation).where(
                Conversation.session_id == request.session_id,
                Conversation.user_id == user_id,
            )
        )
        conv_row = result.scalar_one_or_none()
        if conv_row:
            conv_id = conv_id or conv_row.id
            ws_id = ws_id or conv_row.workspace_id

    if ws_id is not None:
        notebook_row = Notebook(
            workspace_id=ws_id,
            conversation_id=conv_id,
            notebook_uuid=notebook_id,
            label=label,
            query=request.query,
            plan_json=request.plan,
            cells_json=cells,
        )
        db.add(notebook_row)
        await db.commit()
        await db.refresh(notebook_row)
        logger.info("Persisted notebook %s (db id=%s) for workspace %s",
                     notebook_id, notebook_row.id, ws_id)
    else:
        logger.warning("Could not resolve workspace for session %s — notebook not persisted",
                        request.session_id)

    return {
        "notebook_id": notebook_id,
        "query": request.query,
        "cells": cells,
    }


@router.post("/execute-cell")
async def execute_cell(
    request: CellExecutionRequest,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Execute one notebook cell and return its output.

    Looks up the ``NotebookState`` by ``notebook_id``, resolves the
    ``PlanNode`` via ``PhysicalPlan.get_node()``, delegates to
    ``Execution.run_node()``, updates ``datasets_store`` and
    ``cell_statuses``, and streams cell output via SSE.

    Requires:
        - ``request.notebook_id`` references an active or persisted notebook.
        - ``request.cell_id`` is a valid node ID within that notebook's
          physical plan.

    Returns:
        An SSE stream with ``cell_status`` and ``cell_complete`` events.

    Raises:
        HTTPException 404: if the notebook or cell is not found.
    """
    nb = active_notebooks.get(request.notebook_id)
    if nb is None:
        nb = await _rehydrate_notebook(request.notebook_id, user_id, db)
    if nb is None:
        raise HTTPException(status_code=404, detail="Notebook not found or expired")
    nb.last_access = datetime.now()

    # Resolve the node from the physical plan (cell_id == node_id).
    node_id = request.cell_id
    node = nb.physical_plan.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Cell {node_id} not found")

    # If code was modified, invalidate downstream cells
    invalidated_cells: list[str] = []
    if request.code is not None:
        # Build parent_output_map so original_code uses the same format
        # the frontend received (output_dataset_ids, not node IDs).
        parent_output_map = {
            n.node_id: n.output_dataset_id
            for n in nb.physical_plan.nodes
        }
        original_code = node.to_code(parent_output_map=parent_output_map)
        if request.code != original_code:
            # Parse updated params from the modified code (best-effort)
            parsed = _parse_code_to_params(request.code, node.node_type, node.operator_type)
            if parsed:
                # Handle output dataset renaming
                new_out = parsed.pop("_output_dataset_id", None)
                if new_out and new_out != node.output_dataset_id:
                    node.output_dataset_id = new_out

                # Handle input dataset rewiring
                new_inputs = parsed.pop("_input_dataset_ids", None)
                if new_inputs:
                    # Map output_dataset_id → node_id for reverse lookup
                    out_to_node = {
                        n.output_dataset_id: n.node_id
                        for n in nb.physical_plan.nodes
                    }
                    new_parent_ids = [
                        out_to_node[dsid]
                        for dsid in new_inputs
                        if dsid in out_to_node
                    ]
                    if new_parent_ids:
                        node.parent_ids = new_parent_ids

                # Apply remaining operator-specific params
                if parsed:
                    node.params.update(parsed)

            # Invalidate downstream
            invalidated_cells = nb.physical_plan.invalidated_downstream(node_id)
            for inv_id in invalidated_cells:
                nb.cell_statuses[inv_id] = "pending"
                nb.cell_outputs.pop(inv_id, None)
                # Evict cached output
                inv_node = nb.physical_plan.get_node(inv_id)
                nb.datasets_store.pop(inv_node.output_dataset_id, None)

    queue: asyncio.Queue = asyncio.Queue()

    async def _heartbeat():
        try:
            while True:
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                await queue.put(":keep-alive\n\n")
        except asyncio.CancelledError:
            pass

    async def _run_cell():
        try:
            nb.cell_statuses[node_id] = "running"
            await queue.put(
                f"data: {json.dumps({'type': 'cell_status', 'cell_id': node_id, 'status': 'running'})}\n\n"
            )

            exec_instance = nb.execution
            if exec_instance is None:
                await queue.put(
                    f"data: {json.dumps({'type': 'cell_error', 'cell_id': node_id, 'error': 'No execution context'})}\n\n"
                )
                return

            # Run the node in a thread to avoid blocking the event loop
            def _execute():
                updated_store, op_stats = exec_instance.run_node(
                    node_id, nb.datasets_store
                )
                output_dataset = updated_store.get(node.output_dataset_id)
                preview = (
                    exec_instance._build_output_preview(output_dataset)
                    if output_dataset
                    else {}
                )
                # Add answer text for reasoning nodes
                if node.node_type == "reasoning" and output_dataset:
                    preview["answer"] = output_dataset.code_state.get(
                        "final_answer_str", ""
                    )
                return updated_store, op_stats, preview

            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(_execute)

            # Wait for completion
            while not future.done():
                await asyncio.sleep(0.25)

            updated_store, op_stats, preview = future.result()
            executor.shutdown(wait=False)

            # Update the notebook's accumulated datasets
            nb.datasets_store = updated_store

            # Mark the cell as successful and cache its output preview.
            # Sanitise via jsonb_serializer so the preview is safe for
            # both the JSONB column and the SSE payload.
            nb.cell_statuses[node_id] = "success"
            if op_stats is not None:
                preview["operator_stats"] = op_stats.model_dump()
            safe_preview = jsonb_serializer.sanitize(preview)
            nb.cell_outputs[node_id] = safe_preview
            logger.info(
                "[execute_cell] node=%s preview_keys=%s items=%s safe_preview_bytes=%d",
                node_id,
                list(safe_preview.keys()),
                safe_preview.get("items_count"),
                len(json.dumps(safe_preview)),
            )

            # Persist updated cells_json (now including output) to DB.
            # When params were modified (code edit), also persist the
            # updated plan so rehydration picks up the new values.
            plan_update = nb.physical_plan.to_dict() if invalidated_cells else None
            await _persist_notebook_cells(nb.notebook_id, nb.get_cells(), plan_dict=plan_update)

            # Build the response
            payload = {
                "type": "cell_complete",
                "cell_id": node_id,
                "status": "success",
                "output": safe_preview,
                "invalidated_cells": invalidated_cells,
            }

            await queue.put(f"data: {json.dumps(payload, default=str)}\n\n")

        except Exception as exc:
            logger.exception(f"Cell execution failed: {node_id}")
            nb.cell_statuses[node_id] = "error"
            nb.cell_outputs.pop(node_id, None)
            await _persist_notebook_cells(nb.notebook_id, nb.get_cells())
            await queue.put(
                f"data: {json.dumps({'type': 'cell_error', 'cell_id': node_id, 'error': str(exc)})}\n\n"
            )
        finally:
            await queue.put(None)

    async def _stream():
        hb = asyncio.create_task(_heartbeat())
        run = asyncio.create_task(_run_cell())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            hb.cancel()
            run.cancel()
            await asyncio.gather(hb, run, return_exceptions=True)

    return StreamingResponse(_stream(), media_type="text/event-stream")


@router.delete("/notebook/{notebook_id}")
async def delete_notebook(
    notebook_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Tear down an active notebook's in-memory state and delete its DB row.

    Requires:
        - ``notebook_id`` references an active or persisted notebook
          owned by this user.

    Returns:
        ``{ "status": "ok" }``

    Raises:
        HTTPException 404: if the notebook is not found in-memory or in DB.
    """
    nb = active_notebooks.pop(notebook_id, None)

    # Also delete the DB row (if it exists)
    result = await db.execute(
        select(Notebook).where(Notebook.notebook_uuid == notebook_id)
    )
    nb_row = result.scalar_one_or_none()
    if nb_row:
        await db.delete(nb_row)
        await db.commit()

    if nb is None and nb_row is None:
        raise HTTPException(status_code=404, detail="Notebook not found")
    return {"status": "ok"}


@router.get("/notebook/{notebook_id}")
async def get_notebook(
    notebook_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Fetch a notebook's persisted metadata and cells.

    Returns in-memory cells if the notebook is active, otherwise falls
    back to the DB ``cells_json`` snapshot.  The ``active`` flag tells
    the frontend whether the kernel is still alive.

    Requires:
        - ``notebook_id`` references an active or persisted notebook.

    Returns:
        ``{ notebook_id, query, cells, active, label }``

    Raises:
        HTTPException 404: if notebook not found anywhere.
    """
    # Check in-memory first
    nb_mem = active_notebooks.get(notebook_id)
    if nb_mem:
        nb_mem.last_access = datetime.now()
        cells = nb_mem.get_cells()
        logger.info(
            "[get_notebook] in-memory hit notebook=%s cells=%d cells_with_output=%d",
            notebook_id,
            len(cells),
            sum(1 for c in cells if c.get("output")),
        )
        return {
            "notebook_id": notebook_id,
            "query": nb_mem.query,
            "cells": cells,
            "active": True,
            "label": None,  # in-memory doesn't track label
        }

    # Fallback to DB
    result = await db.execute(
        select(Notebook).where(
            Notebook.notebook_uuid == notebook_id,
        )
    )
    nb_row = result.scalar_one_or_none()
    if nb_row is None:
        raise HTTPException(status_code=404, detail="Notebook not found")

    db_cells = nb_row.cells_json or []
    logger.info(
        "[get_notebook] DB fallback notebook=%s cells=%d cells_with_output=%d",
        notebook_id,
        len(db_cells),
        sum(1 for c in db_cells if c.get("output")),
    )
    return {
        "notebook_id": notebook_id,
        "query": nb_row.query,
        "cells": db_cells,
        "active": False,
        "label": nb_row.label,
    }


@router.post("/notebook/{notebook_id}/cells")
async def add_cell(
    notebook_id: str,
    request: AddCellRequest,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Insert a new cell into an active notebook.

    Creates a new ``PlanNode``, splices it into the ``PhysicalPlan``
    DAG after ``after_cell_id``, invalidates downstream cells, and
    returns the updated cells list.

    Requires:
        - ``notebook_id`` references an active or persisted notebook.
        - ``after_cell_id`` (if provided) is a valid node ID.

    Returns:
        ``{ cell, invalidated_cells, updated_cells }``

    Raises:
        HTTPException 404: if the notebook or referenced cell is not found.
    """
    nb = active_notebooks.get(notebook_id)
    if nb is None:
        nb = await _rehydrate_notebook(notebook_id, user_id, db)
    if nb is None:
        raise HTTPException(status_code=404, detail="Notebook not found or expired")
    nb.last_access = datetime.now()

    from carnot.plan.node import PlanNode

    # Generate a unique node ID
    existing_ids = {n.node_id for n in nb.physical_plan.nodes}
    counter = len(existing_ids)
    while f"node-{counter}" in existing_ids:
        counter += 1
    new_node_id = f"node-{counter}"

    # Determine output_dataset_id
    op_type = request.operator_type or ""
    output_dataset_id = f"{op_type or 'Custom'}Operation_{new_node_id}"

    # Determine parent for code generation
    after_node_id = request.after_cell_id
    if after_node_id is None:
        # Insert at top — find first node with no parents
        topo = nb.physical_plan.topo_order()
        if topo:
            after_node_id = topo[0].node_id

    new_node = PlanNode(
        node_id=new_node_id,
        node_type=request.cell_type,
        operator_type=request.operator_type,
        name=request.operator_type or "New Cell",
        description=f"New {request.operator_type or 'custom'} cell",
        params={},
        parent_ids=[after_node_id] if after_node_id else [],
        output_dataset_id=output_dataset_id,
    )

    if after_node_id:
        invalidated = nb.physical_plan.insert_node(after_node_id, new_node)
    else:
        nb.physical_plan._nodes[new_node_id] = new_node
        invalidated = []

    # Set status for new cell and invalidated cells
    nb.cell_statuses[new_node_id] = "pending"
    for inv_id in invalidated:
        nb.cell_statuses[inv_id] = "pending"
        nb.cell_outputs.pop(inv_id, None)
        inv_node = nb.physical_plan.get_node(inv_id)
        nb.datasets_store.pop(inv_node.output_dataset_id, None)

    cells = nb.get_cells()
    new_cell = next((c for c in cells if c["cell_id"] == new_node_id), None)

    # Persist updated cell layout and plan (structure changed) to DB
    await _persist_notebook_cells(nb.notebook_id, cells, plan_dict=nb.physical_plan.to_dict())

    return {
        "cell": new_cell,
        "invalidated_cells": invalidated,
        "updated_cells": cells,
    }


@router.delete("/notebook/{notebook_id}/cells/{cell_id}")
async def delete_cell(
    notebook_id: str,
    cell_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Remove a cell from an active notebook.

    Deletes the ``PlanNode`` from the ``PhysicalPlan`` DAG, rewires
    dependencies, invalidates downstream cells, and returns the
    updated cells list.

    Requires:
        - ``notebook_id`` references an active or persisted notebook.
        - ``cell_id`` is a valid, non-dataset node ID.

    Returns:
        ``{ invalidated_cells, updated_cells }``

    Raises:
        HTTPException 404/400: if notebook/cell not found or cannot delete.
    """
    nb = active_notebooks.get(notebook_id)
    if nb is None:
        nb = await _rehydrate_notebook(notebook_id, user_id, db)
    if nb is None:
        raise HTTPException(status_code=404, detail="Notebook not found or expired")
    nb.last_access = datetime.now()

    try:
        node = nb.physical_plan.get_node(cell_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Cell {cell_id} not found") from None

    # Prevent deleting structural anchors
    if node.node_type == "dataset":
        raise HTTPException(status_code=400, detail="Cannot delete a dataset cell")
    if node.node_type == "reasoning":
        raise HTTPException(status_code=400, detail="Cannot delete the reasoning cell")

    try:
        invalidated = nb.physical_plan.delete_node(cell_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None

    # Clean up statuses and cached datasets
    nb.cell_statuses.pop(cell_id, None)
    nb.cell_outputs.pop(cell_id, None)
    nb.datasets_store.pop(node.output_dataset_id, None)
    for inv_id in invalidated:
        nb.cell_statuses[inv_id] = "pending"
        nb.cell_outputs.pop(inv_id, None)
        try:
            inv_node = nb.physical_plan.get_node(inv_id)
            nb.datasets_store.pop(inv_node.output_dataset_id, None)
        except KeyError:
            pass

    updated_cells = nb.get_cells()

    # Persist updated cell layout and plan (structure changed) to DB
    await _persist_notebook_cells(nb.notebook_id, updated_cells, plan_dict=nb.physical_plan.to_dict())

    return {
        "invalidated_cells": invalidated,
        "updated_cells": updated_cells,
    }


@router.post("/notebook/{notebook_id}/cells/move")
async def move_cell(
    notebook_id: str,
    request: MoveCellRequest,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Move a cell up or down in the notebook's topological order.

    Swaps the target cell with its neighbor in the topological order.
    Both cells and all downstream cells are invalidated.

    Requires:
        - ``notebook_id`` references an active or persisted notebook.
        - ``request.cell_id`` is a valid node ID.
        - ``request.direction`` is ``"up"`` or ``"down"``.

    Returns:
        ``{ invalidated_cells, updated_cells }``

    Raises:
        HTTPException 400/404: on invalid move or not found.
    """
    nb = active_notebooks.get(notebook_id)
    if nb is None:
        nb = await _rehydrate_notebook(notebook_id, user_id, db)
    if nb is None:
        raise HTTPException(status_code=404, detail="Notebook not found or expired")
    nb.last_access = datetime.now()

    topo = nb.physical_plan.topo_order()
    topo_ids = [n.node_id for n in topo]

    if request.cell_id not in topo_ids:
        raise HTTPException(status_code=404, detail=f"Cell {request.cell_id} not found")

    idx = topo_ids.index(request.cell_id)
    node = topo[idx]

    # Prevent moving dataset or reasoning cells
    if node.node_type == "dataset":
        raise HTTPException(status_code=400, detail="Cannot move a dataset cell")
    if node.node_type == "reasoning":
        raise HTTPException(status_code=400, detail="Cannot move the reasoning cell")

    if request.direction == "up":
        if idx <= 0:
            raise HTTPException(status_code=400, detail="Cell is already at the top")
        swap_node = topo[idx - 1]
        if swap_node.node_type == "dataset":
            raise HTTPException(status_code=400, detail="Cannot move above a dataset cell")
    elif request.direction == "down":
        if idx >= len(topo) - 1:
            raise HTTPException(status_code=400, detail="Cell is already at the bottom")
        swap_node = topo[idx + 1]
        if swap_node.node_type == "reasoning":
            raise HTTPException(status_code=400, detail="Cannot move below the reasoning cell")
    else:
        raise HTTPException(status_code=400, detail="Direction must be 'up' or 'down'")

    # Swap parent_ids between the two nodes
    node.parent_ids, swap_node.parent_ids = swap_node.parent_ids, node.parent_ids

    # Rewire children of both nodes
    for n in nb.physical_plan.nodes:
        new_parents = []
        for pid in n.parent_ids:
            if pid == node.node_id:
                new_parents.append(swap_node.node_id)
            elif pid == swap_node.node_id:
                new_parents.append(node.node_id)
            else:
                new_parents.append(pid)
        n.parent_ids = new_parents

    # Fix self-references — restore the swapped parent_ids properly
    node.parent_ids = [
        swap_node.node_id if pid == node.node_id else pid
        for pid in node.parent_ids
    ]
    swap_node.parent_ids = [
        node.node_id if pid == swap_node.node_id else pid
        for pid in swap_node.parent_ids
    ]

    # Invalidate both cells and downstream
    invalidated = set()
    invalidated.add(node.node_id)
    invalidated.add(swap_node.node_id)
    invalidated.update(nb.physical_plan.invalidated_downstream(node.node_id))
    invalidated.update(nb.physical_plan.invalidated_downstream(swap_node.node_id))

    for inv_id in invalidated:
        nb.cell_statuses[inv_id] = "pending"
        nb.cell_outputs.pop(inv_id, None)
        try:
            inv_node = nb.physical_plan.get_node(inv_id)
            nb.datasets_store.pop(inv_node.output_dataset_id, None)
        except KeyError:
            pass

    updated_cells = nb.get_cells()

    # Persist updated cell layout and plan (structure changed) to DB
    await _persist_notebook_cells(nb.notebook_id, updated_cells, plan_dict=nb.physical_plan.to_dict())

    return {
        "invalidated_cells": sorted(invalidated),
        "updated_cells": updated_cells,
    }
