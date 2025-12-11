import asyncio
import json
import logging
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import fsspec
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

import carnot
from app.auth import get_user_hash
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
from app.env import BACKEND_ROOT, BASE_DIR, FILESYSTEM, IS_LOCAL_ENV
from app.services.file_service import LocalFileService, S3FileService
from app.services.llm import get_user_llm_config

router = APIRouter()
logger = logging.getLogger(__name__)
file_service = LocalFileService() if IS_LOCAL_ENV else S3FileService()
SESSION_TIMEOUT = timedelta(minutes=30)


class OutputCapture:
    """
    Captures output and writes it to a file (local or S3) while also echoing it to the original stream.
    """
    def __init__(self, filepath: str, original_stream):
        self.filepath = filepath
        self.original_stream = original_stream
        self.file_handle = None
        self.file_stream = None
        self._open_stream()

    def _open_stream(self):
        """Open the stream using fsspec."""
        try:
            # open the stream in append mode ('a') since stdout/stderr writes are sequential
            # and may be interleaved. 'w' would overwrite the previous stream's output.
            fs = fsspec.filesystem(FILESYSTEM)
            self.file_handle = fs.open(self.filepath, mode='a', encoding='utf-8')
            self.file_stream = self.file_handle.__enter__()
        except Exception as e:
            self.original_stream.write(f"Error opening output stream for {self.filepath}: {e}\n")
            raise

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


def extract_plan_from_output(output) -> str | None:
    """Extract the CodeAgent planning steps from the DataRecordCollection."""
    try:
        data_records = getattr(output, "data_records", None)
        if not data_records:
            return None

        seen: set[str] = set()
        ordered_plans: list[str] = []

        for record in data_records:
            try:
                context_obj = record["context"]
            except Exception:
                context_obj = None

            plan_value = getattr(context_obj, "plan", None) if context_obj is not None else None
            if isinstance(plan_value, str):
                plan_str = plan_value.strip()
                if plan_str and plan_str not in seen:
                    ordered_plans.append(plan_str)
                    seen.add(plan_str)

        if ordered_plans:
            return "\n\n".join(ordered_plans)
    except Exception:
        logger.debug("Failed to extract plan from output", exc_info=True)

    return None


class QueryRequest(BaseModel):
    query: str
    dataset_ids: list[int]
    session_id: str


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
    session_id: str, query: str, dataset_ids: list[int]
) -> int:
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Conversation).where(Conversation.session_id == session_id)
        )
        conversation = result.scalar_one_or_none()
        if conversation:
            return conversation.id

        title = f"{query[:50]}..." if len(query) > 50 else query
        dataset_ids_str = ",".join(map(str, dataset_ids))

        conversation = Conversation(
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
    csv_file: str | None = None,
    row_count: int | None = None,
) -> None:
    async with AsyncSessionLocal() as db:
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            csv_file=csv_file,
            row_count=row_count,
        )
        db.add(message)

        result = await db.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        conversation = result.scalar_one_or_none()
        if conversation:
            conversation.updated_at = datetime.now(timezone.utc)

        await db.commit()


async def stream_query_execution(
    query: str, dataset_ids: list[int], session_id: str, user_config: dict,
):
    try:
        cleanup_old_sessions()

        session_id = session_id or str(uuid4())
        conversation_id = await get_or_create_conversation(
            session_id, query, dataset_ids
        )
        await save_message(conversation_id, "user", query)

        yield f"data: {json.dumps({'type': 'status', 'message': 'Starting query execution...', 'session_id': session_id})}\n\n"
        await asyncio.sleep(0.1)
        yield f"data: {json.dumps({'type': 'status', 'message': 'Loading datasets...'})}\n\n"
        await asyncio.sleep(0.1)

        async with AsyncSessionLocal() as db:
            datasets = []
            for dataset_id in dataset_ids:
                dataset_result = await db.execute(
                    select(Dataset).where(Dataset.id == dataset_id)
                )
                dataset = dataset_result.scalar_one_or_none()
                if not dataset:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Dataset {dataset_id} not found'})}\n\n"
                    return

                files_result = await db.execute(
                    select(FileRecord)
                    .join(DatasetFile, FileRecord.id == DatasetFile.file_id)
                    .where(DatasetFile.dataset_id == dataset_id)
                )
                files = files_result.scalars().all()
                datasets.append([file.file_path for file in files])

        yield f"data: {json.dumps({'type': 'status', 'message': f'Loaded {len(datasets)} dataset(s)'})}\n\n"
        await asyncio.sleep(0.1)

        all_files = [Path(path) for dataset in datasets for path in dataset]
        if not all_files:
            yield f"data: {json.dumps({'type': 'error', 'message': 'No files found in selected datasets'})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'status', 'message': f'Processing {len(all_files)} files...'})}\n\n"
        await asyncio.sleep(0.1)

        session_exists = session_id in active_sessions
        if session_exists and set(active_sessions[session_id]["dataset_ids"]) != set(
            dataset_ids
        ):
            session_exists = False

        session_dir = Path(BASE_DIR, "sessions", session_id)
        file_service.create_dir(str(session_dir))

        yield f"data: {json.dumps({'type': 'status', 'message': 'Preparing data context...'})}\n\n"
        await asyncio.sleep(0.1)

        # NOTE: this copies files to a session-specific directory; we cannot make copies of large datasets
        if not session_exists:
            text_file_count = 0
            for file_path in all_files:
                if file_path.suffix.lower() in {".pdf", ".jpg", ".jpeg", ".png", ".gif", ".zip", ".exe", ".bin"}:
                    continue

                destination = session_dir / file_path.name
                try:
                    destination.write_bytes(file_path.read_bytes())
                    text_file_count += 1
                except OSError as exc:
                    logger.debug("Skipping file %s: %s", file_path, exc)

            if text_file_count == 0:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No text files found in selected datasets'})}\n\n"
                return

            yield f"data: {json.dumps({'type': 'status', 'message': f'Processing {text_file_count} text files...'})}\n\n"
            await asyncio.sleep(0.1)
        else:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Continuing conversation...'})}\n\n"
            await asyncio.sleep(0.1)

        if session_exists:
            ctx = active_sessions[session_id]["context"]
        else:
            context_id = f"session_{session_id[:8]}"
            ctx = carnot.TextFileContext(
                path=str(session_dir),
                id=context_id,
                description=f"Query on {len(datasets)} dataset(s)",
                llm_config=user_config,
            )

        yield f"data: {json.dumps({'type': 'status', 'message': f'Executing query: {query}'})}\n\n"
        await asyncio.sleep(0.1)

        # setup progress logging and clear old progress log if it exists
        fs = fsspec.filesystem(FILESYSTEM)
        progress_log = str(Path(session_dir, "progress.jsonl"))
        if fs.exists(progress_log):
            fs.rm(progress_log, recursive=False)

        compute_ctx = ctx.compute(query)
        config = carnot.QueryProcessorConfig(
            policy=carnot.MaxQuality(),
            llm_config=user_config,
            progress=True,  # Enable console progress
            session_id=session_id,  # Add session ID for tracking
            progress_log_file=progress_log,  # Add progress log file path
        )

        yield f"data: {json.dumps({'type': 'status', 'message': 'Running Carnot query processor...'})}\n\n"
        await asyncio.sleep(0.1)

        output_log = str(Path(session_dir, "output.txt"))
        if fs.exists(output_log):
            fs.rm(output_log, recursive=False)

        original_stdout = sys.stdout
        original_stderr = sys.stderr
        def run_query_with_capture():
            stdout_capture = OutputCapture(output_log, original_stdout)
            stderr_capture = OutputCapture(output_log, original_stderr)
            try:
                sys.stdout = stdout_capture
                sys.stderr = stderr_capture
                return compute_ctx.run(config=config)
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                stdout_capture.close()
                stderr_capture.close()

        loop = asyncio.get_event_loop()
        output = await loop.run_in_executor(None, run_query_with_capture)

        active_sessions[session_id] = {
            "context": compute_ctx,
            "last_access": datetime.now(),
            "dataset_ids": dataset_ids,
            "session_dir": str(session_dir),
        }

        plan_text = extract_plan_from_output(output)
        if plan_text:
            await save_message(conversation_id, "plan", plan_text)
            yield f"data: {json.dumps({'type': 'plan', 'message': plan_text, 'session_id': session_id})}\n\n"
            await asyncio.sleep(0.1)

        yield f"data: {json.dumps({'type': 'status', 'message': 'Processing results...'})}\n\n"
        await asyncio.sleep(0.1)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"query_results_{timestamp}.csv"
        results_dir = Path(BASE_DIR, "results")
        file_service.create_dir(str(results_dir))
        csv_path = results_dir / csv_filename

        try:
            # First, save to our timestamp-based file
            df = output.to_df()
            df.to_csv(csv_path, index=False)

            # Try to extract the actual filename from Carnot's output
            csv_filename_from_output = None
            if not df.empty:
                # Check all columns for CSV filename mentions
                # Pattern to match CSV filenames (e.g., "filtered_enron_emails.csv" or "output.csv")
                csv_pattern = r'\b([a-zA-Z0-9_\-]+\.csv)\b'
                for col in df.columns:
                    for value in df[col]:
                        if isinstance(value, str):
                            # Look for any CSV filename in the text
                            matches = re.findall(csv_pattern, value, re.IGNORECASE)
                            if matches:
                                # Use the last match (most likely the output file)
                                csv_filename_from_output = matches[-1]
                                break
                    if csv_filename_from_output:
                        break
                
                # If we found a filename, check if that file exists and use it
                if csv_filename_from_output:
                    actual_csv_path = BACKEND_ROOT / csv_filename_from_output
                    if actual_csv_path.exists() and actual_csv_path != csv_path:
                        # Use the file that Carnot created
                        csv_filename = csv_filename_from_output
                        csv_path = actual_csv_path
                        df = pd.read_csv(csv_path)  # Re-read from the actual file
                    else:
                        # File doesn't exist, rename our file to match
                        csv_path.rename(actual_csv_path)
                        csv_path = actual_csv_path
                        csv_filename = csv_filename_from_output

            if df.empty:
                message_text = "No results found for your query."
                await save_message(conversation_id, "result", message_text)
                yield f"data: {json.dumps({'type': 'result', 'message': message_text, 'session_id': session_id})}\n\n"
            else:
                if len(df.columns) >= 2:
                    result_column = df.iloc[:, 1]
                    lines = [
                        f"  {index + 1}. {value}"
                        for index, value in enumerate(result_column.tolist())
                    ]
                    body = "\n".join(lines)
                else:
                    body = df.to_string(index=False)

                message_text = (
                    "Query completed successfully!\n\n"
                    f"Found {len(df)} result(s):\n\n{body}"
                )
                await save_message(
                    conversation_id, "result", message_text, csv_filename, len(df)
                )
                yield f"data: {json.dumps({'type': 'result', 'message': message_text, 'csv_file': csv_filename, 'row_count': len(df), 'session_id': session_id})}\n\n"

        except Exception as exc:
            logger.exception("Error processing query results")
            error_msg = f"Error processing results: {exc}"
            await save_message(conversation_id, "error", error_msg)
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'message': 'Query execution complete'})}\n\n"

    except Exception as exc:
        logger.exception("Query execution failed")
        error_msg = f"Error executing query: {exc}"
        if "conversation_id" in locals():
            try:
                await save_message(conversation_id, "error", error_msg)
            except Exception:
                logger.exception("Failed to save error message")
        yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"

@router.post("/execute")
async def execute_query(
    request: QueryRequest,
    auth_data: tuple = Depends(get_user_hash), 
    db: Session = Depends(get_db),
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
    user_hash, _ = auth_data
    user_config = await get_user_llm_config(db, user_hash)
    if not user_config:
        raise HTTPException(
            status_code=400, 
            detail={
                "type": "API_KEY_MISSING",
                "message": "No LLM API keys found for this user. Please configure them in Settings.",
            }
        )

    return StreamingResponse(
        stream_query_execution(request.query, request.dataset_ids, request.session_id, user_config),
        media_type="text/event-stream"
    )

@router.get("/progress/{session_id}")
async def get_progress(session_id: str, since_timestamp: str | None = None):
    """
    Get progress events for a session, optionally filtering by timestamp
    """
    try:
        fs = fsspec.filesystem(FILESYSTEM)
        session_dir = str(Path(BASE_DIR, "sessions", session_id))
        progress_log = str(Path(session_dir, "progress.jsonl"))
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
    try:
        fs = fsspec.filesystem(FILESYSTEM)
        session_dir = str(Path(BASE_DIR, "sessions", session_id))
        output_file = str(Path(session_dir, "output.txt"))
        if not fs.exists(output_file):
            return {"lines": [], "total_lines": 0}

        with fs.open(output_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            new_lines = all_lines[last_line:]
            return {"lines": new_lines, "total_lines": len(all_lines)}
    except Exception as e:
        logger.error(f"Error reading output log: {e}")
        return {"lines": [], "total_lines": 0}


@router.get("/download/{filename}")
async def download_query_results(filename: str):
    """
    Download a query results CSV file
    """
    # Security: only allow downloading CSV files from backend directory
    if not filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid filename - must be a CSV file")

    # Prevent directory traversal
    if "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Get the backend directory path
    fs = fsspec.filesystem(FILESYSTEM)
    file_path = str(Path(BASE_DIR, "results", filename))
    if not fs.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="text/csv"
    )
