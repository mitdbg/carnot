import asyncio
import concurrent.futures
import json
import logging
import os
import queue as _queue_mod
import sys
import tempfile
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
    get_db,
)
from app.database import (
    File as FileRecord,
)
from app.env import BASE_DIR, FILESYSTEM, IS_LOCAL_ENV
from app.services.file_service import LocalFileService, S3FileService
from app.services.llm import get_user_llm_config
from carnot.data.dataset import Dataset as CarnotDataset
from carnot.data.item import DataItem
from carnot.execution.progress import ExecutionProgress, PlanningProgress
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

            await self.queue.put(f"data: {json.dumps({'type': 'execution_status', 'message': 'Starting query execution...', 'session_id': session_id})}\n\n")
            await self.queue.put(f"data: {json.dumps({'type': 'execution_status', 'message': 'Loading datasets...'})}\n\n")

            async with AsyncSessionLocal() as db:
                datasets: list[CarnotDataset] = []
                for dataset_id in self.dataset_ids:
                    dataset_result = await db.execute(
                        select(Dataset).where(Dataset.id == dataset_id)
                    )
                    dataset = dataset_result.scalar_one_or_none()
                    if not dataset:
                        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

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
                
                for dataset in datasets:
                    logger.info(f"Dataset: {dataset.name}, Annotation: {dataset.annotation}, Items: {[item.path for item in dataset.items]}")

            await self.queue.put(f"data: {json.dumps({'type': 'execution_status', 'message': f'Loaded {len(datasets)} dataset(s)'})}\n\n")

            all_files = [
                S3Path(item.path) if item.path.startswith("s3://") else Path(item.path)
                for dataset in datasets
                for item in dataset
            ]
            if not all_files:
                await self.queue.put(f"data: {json.dumps({'type': 'error', 'message': 'No files found in selected datasets'})}\n\n")
                return

            await self.queue.put(f"data: {json.dumps({'type': 'execution_status', 'message': f'Processing {len(all_files)} files...'})}\n\n")

            session_exists = session_id in active_sessions
            if session_exists and set(active_sessions[session_id]["dataset_ids"]) != set(self.dataset_ids):
                session_exists = False

            session_dir = Path(BASE_DIR, ".sessions", session_id) if IS_LOCAL_ENV else S3Path(BASE_DIR, ".sessions", session_id)
            file_service.create_dir(str(session_dir))

            await self.queue.put(f"data: {json.dumps({'type': 'execution_status', 'message': 'Preparing data context...'})}\n\n")

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

                await self.queue.put(f"data: {json.dumps({'type': 'execution_status', 'message': f'Processing {text_file_count} text files...'})}\n\n")
            else:
                await self.queue.put(f"data: {json.dumps({'type': 'execution_status', 'message': 'Continuing conversation...'})}\n\n")

            await self.queue.put(f"data: {json.dumps({'type': 'execution_status', 'message': f'Executing query: {self.query}'})}\n\n")

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
            
            # create execution and execute plan
            exec_instance = carnot.Execution(
                query=self.query,
                datasets=datasets,
                plan=self.plan,
                conversation=conversation,
                tools=[],
                memory=None,
                indices=[],
                llm_config=self.user_config,
                progress_log_file=progress_log,
                cost_budget=self.cost_budget,
                storage=_build_storage(),
            )

            await self.queue.put(f"data: {json.dumps({'type': 'execution_status', 'message': 'Running Carnot query processor...'})}\n\n")

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

            # Use run_stream() to get operator-level progress events.
            # The generator is synchronous; we drive it in a background
            # thread and ferry ExecutionProgress events through a
            # thread-safe queue so we can push them to the SSE stream.
            exec_progress_queue: _queue_mod.Queue = _queue_mod.Queue()

            def run_query_with_capture():
                local_output_path = None
                try:
                    sys.stdout = stdout_capture
                    sys.stderr = stderr_capture
                    gen = exec_instance.run_stream()
                    items = None
                    answer = None
                    try:
                        while True:
                            event = next(gen)
                            if isinstance(event, ExecutionProgress):
                                exec_progress_queue.put(event.to_dict())
                    except StopIteration as exc:
                        result = exc.value
                        if result is not None:
                            items, answer = result
                    return items, answer
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

            # Poll for execution progress events while the thread is working
            while not future.done():
                try:
                    progress_dict = exec_progress_queue.get(timeout=0.25)
                    await self.queue.put(
                        f"data: {json.dumps({'type': 'execution_status', 'message': progress_dict.get('message', ''), 'operator_name': progress_dict.get('operator_name', ''), 'operator_index': progress_dict.get('operator_index'), 'total_operators': progress_dict.get('total_operators')})}\n\n"
                    )
                except _queue_mod.Empty:
                    await asyncio.sleep(0.1)

            # Drain any remaining progress events
            while not exec_progress_queue.empty():
                try:
                    progress_dict = exec_progress_queue.get_nowait()
                    await self.queue.put(
                        f"data: {json.dumps({'type': 'execution_status', 'message': progress_dict.get('message', ''), 'operator_name': progress_dict.get('operator_name', ''), 'operator_index': progress_dict.get('operator_index'), 'total_operators': progress_dict.get('total_operators')})}\n\n"
                    )
                except _queue_mod.Empty:
                    break

            items, answer = future.result()
            executor.shutdown(wait=False)

            await self.queue.put(f"data: {json.dumps({'type': 'execution_status', 'message': 'Processing results...'})}\n\n")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"query_results_{timestamp}.csv"
            results_dir = Path(BASE_DIR, ".results") if IS_LOCAL_ENV else S3Path(BASE_DIR, ".results")
            file_service.create_dir(str(results_dir))
            csv_path = results_dir / csv_filename
            fs = fsspec.filesystem(FILESYSTEM)

            try:
                # First, save to our timestamp-based file
                df = pd.DataFrame(items)
                with fs.open(str(csv_path), 'w', encoding='utf-8') as f:
                    df.to_csv(f, index=False)

                if df.empty and (not answer or answer.strip() == ""):
                    message_text = "No results found for your query."
                    await save_message(conversation_id, "agent", message_text, "result")
                    await self.queue.put(f"data: {json.dumps({'type': 'result', 'message': message_text, 'session_id': session_id})}\n\n")
                elif df.empty:
                    message_text = (
                        "Query completed successfully!\n\n"
                        f"Answer Text: {answer}\n\n"
                        "No tabular results found."
                    )
                    await save_message(conversation_id, "agent", message_text, "result")
                    await self.queue.put(f"data: {json.dumps({'type': 'result', 'message': message_text, 'session_id': session_id})}\n\n")
                elif not answer or answer.strip() == "":
                    body = str(df.head())
                    message_text = (
                        "Query completed successfully!\n\n"
                        f"Found {len(df)} result(s):\n\n{body}\n..."
                    )
                    await save_message(conversation_id, "agent", message_text, "result", csv_filename, len(df))
                    await self.queue.put(f"data: {json.dumps({'type': 'result', 'message': message_text, 'csv_file': csv_filename, 'row_count': len(df), 'session_id': session_id})}\n\n")
                else:
                    body = str(df.head())
                    message_text = (
                        "Query completed successfully!\n\n"
                        f"Answer Text: {answer}\n\n"
                        f"Found {len(df)} result(s):\n\n{body}\n..."
                    )
                    await save_message(conversation_id, "agent", message_text, "result", csv_filename, len(df))
                    await self.queue.put(f"data: {json.dumps({'type': 'result', 'message': message_text, 'csv_file': csv_filename, 'row_count': len(df), 'session_id': session_id})}\n\n")

            except Exception as exc:
                logger.exception("Error processing query results")
                error_msg = f"Error processing results: {exc}"
                await save_message(conversation_id, "agent", error_msg, "error")
                await self.queue.put(f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n")

            await self.queue.put(f"data: {json.dumps({'type': 'done', 'message': 'Query execution complete'})}\n\n")

        except Exception as exc:
            logger.exception("Query execution failed")
            error_msg = f"Error executing query: {exc}"
            if conversation_id is not None:
                try:
                    await save_message(conversation_id, "agent", error_msg, "error")
                except Exception:
                    logger.exception("Failed to save error message")
            await self.queue.put(f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n")
        finally:
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

                datasets: list[CarnotDataset] = []
                for dataset_id in self.dataset_ids:
                    dataset_result = await db.execute(
                        select(Dataset).where(Dataset.id == dataset_id)
                    )
                    dataset = dataset_result.scalar_one_or_none()
                    if not dataset:
                        await self.queue.put(
                            f"data: {json.dumps({'type': 'error', 'message': f'Dataset {dataset_id} not found'})}\n\n"
                        )
                        return

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

            # --- conversation bookkeeping --------------------------------------
            conversation_id = await get_or_create_conversation(
                self.user_id, self.session_id, self.query, self.dataset_ids
            )
            conversation = await load_conversation_from_db(self.user_id, self.session_id)
            await save_message(conversation_id, "user", self.query, cost_budget=self.cost_budget)

            await self.queue.put(
                f"data: {json.dumps({'type': 'planning_status', 'message': 'Starting plan generation…', 'session_id': self.session_id})}\n\n"
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
            progress_sync_queue: _queue_mod.Queue = _queue_mod.Queue()

            def _run_plan_stream_with_progress():
                """Drive the plan_stream generator, posting progress to a sync queue."""
                nl_plan = None
                logical_plan = None
                gen = exec_instance.plan_stream()
                try:
                    while True:
                        event = next(gen)
                        if isinstance(event, PlanningProgress):
                            progress_sync_queue.put(event.to_dict())
                except StopIteration as exc:
                    result = exc.value
                    if result is not None:
                        nl_plan, logical_plan = result
                return nl_plan, logical_plan

            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(_run_plan_stream_with_progress)

            # Poll for progress events while the thread is working
            while not future.done():
                try:
                    progress_dict = progress_sync_queue.get(timeout=0.25)
                    await self.queue.put(
                        f"data: {json.dumps({'type': 'planning_status', 'message': progress_dict.get('message', ''), 'phase': progress_dict.get('phase', ''), 'step': progress_dict.get('step'), 'total_steps': progress_dict.get('total_steps')})}\n\n"
                    )
                except _queue_mod.Empty:
                    await asyncio.sleep(0.1)

            # Drain any remaining progress events
            while not progress_sync_queue.empty():
                try:
                    progress_dict = progress_sync_queue.get_nowait()
                    await self.queue.put(
                        f"data: {json.dumps({'type': 'planning_status', 'message': progress_dict.get('message', ''), 'phase': progress_dict.get('phase', ''), 'step': progress_dict.get('step'), 'total_steps': progress_dict.get('total_steps')})}\n\n"
                    )
                except _queue_mod.Empty:
                    break

            nl_plan, logical_plan = future.result()
            executor.shutdown(wait=False)

            # --- save results to DB -------------------------------------------
            if nl_plan:
                await save_message(
                    conversation_id, "agent", nl_plan,
                    message_type="natural-language-plan",
                )
            if logical_plan:
                plan_json = json.dumps(logical_plan, indent=2)
                await save_message(
                    conversation_id, "agent", plan_json,
                    message_type="logical-plan",
                )

            logger.info(
                f"Generated plan for session {self.session_id}:\n{nl_plan}\n"
                f"{json.dumps(logical_plan, indent=2) if logical_plan else '(none)'}"
            )

            # --- send final plan_complete event --------------------------------
            await self.queue.put(
                f"data: {json.dumps({'type': 'plan_complete', 'natural_language_plan': nl_plan, 'plan': logical_plan, 'session_id': self.session_id})}\n\n"
            )

        except Exception as exc:
            logger.exception("Plan generation failed")
            error_msg = f"Error generating plan: {exc}"
            if conversation_id is not None:
                try:
                    await save_message(conversation_id, "agent", error_msg, "error")
                except Exception:
                    logger.exception("Failed to save error message")
            await self.queue.put(
                f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
            )
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

        title = f"{query[:50]}..." if len(query) > 50 else query
        dataset_ids_str = ",".join(map(str, dataset_ids))

        conversation = Conversation(
            user_id=user_id,
            session_id=session_id,
            title=title,
            dataset_ids=dataset_ids_str,
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
) -> None:
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
        
        # Parse dataset_ids from comma-separated string
        dataset_ids = []
        if db_conversation.dataset_ids:
            dataset_ids = db_conversation.dataset_ids.split(",")
        
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
    - ``planning_status`` events as the planner/data-discovery agent work.
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
