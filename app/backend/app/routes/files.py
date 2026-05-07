import logging
import os
import uuid
from io import BytesIO

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.auth import get_current_user
from app.database import AsyncSessionLocal, UploadJob, get_db
from app.database import File as FileRecord
from app.env import BASE_DIR, DATA_DIR, IS_LOCAL_ENV, SHARED_DATA_DIR
from app.models.schemas import DirectoryCreate, FileBatchDelete, PaginatedFileList
from app.services.file_service import (
    ARCHIVE_EXTENSIONS,
    DEFAULT_PAGE_SIZE,
    LocalFileService,
    S3FileService,
    _extract_archive,
    normalize_path,
)

logger = logging.getLogger('uvicorn.error')

router = APIRouter()
file_service = LocalFileService() if IS_LOCAL_ENV else S3FileService()

# Max items to return in a single page to prevent overload
MAX_ITEMS_PER_REQUEST = 200

# In-memory progress tracker: job_id -> number of files uploaded so far.
# Updated by the background thread; read by the status endpoint for live progress.
_job_progress: dict[str, int] = {}


class _UploadFileLike:
    """Minimal shim so raw bytes can be passed to file_service methods that expect an UploadFile."""
    def __init__(self, filename: str, data: bytes) -> None:
        self.filename = filename
        self.file = BytesIO(data)


def _sync_upload_job(job_id: str, file_bytes: bytes, filename: str, upload_dir: str) -> list[str]:
    """
    Synchronous work that runs inside a threadpool so it never blocks the event loop.
    Extracts the archive (if applicable), writes every file to storage, and returns
    the list of uploaded destination paths.
    """
    mock_file = _UploadFileLike(filename, file_bytes)
    if any(filename.lower().endswith(ext) for ext in ARCHIVE_EXTENSIONS):
        streams, paths = _extract_archive(mock_file, upload_dir)
    else:
        streams = [BytesIO(file_bytes)]
        paths = [os.path.join(upload_dir, filename)]

    _job_progress[job_id] = 0
    uploaded_paths: list[str] = []
    for stream, path in zip(streams, paths, strict=True):
        dir_path = os.path.dirname(path)
        if not file_service.exists(dir_path):
            file_service.create_dir(dir_path)
        file_service._write_file_to_path(stream, path)
        uploaded_paths.append(path)
        _job_progress[job_id] = len(uploaded_paths)

    return uploaded_paths


async def _run_upload_job(
    job_id: str,
    file_bytes: bytes,
    filename: str,
    upload_dir: str,
    user_id: str,
    shared: bool,
) -> None:
    """
    Background coroutine — runs after the 202 response is sent to the client.
    Performs the actual file extraction + storage upload, then records
    the resulting file paths in the database.
    """
    async with AsyncSessionLocal() as db:
        job = await db.get(UploadJob, job_id)
        job.status = "running"
        await db.commit()

        try:
            uploaded_paths = await run_in_threadpool(
                _sync_upload_job, job_id, file_bytes, filename, upload_dir
            )

            # Bulk-insert FileRecord rows in batches to avoid a single huge transaction.
            db_batch_size = 500
            for i in range(0, len(uploaded_paths), db_batch_size):
                batch = uploaded_paths[i : i + db_batch_size]
                db.add_all([FileRecord(user_id=user_id, file_path=p, shared=shared) for p in batch])
                await db.flush()

            job.status = "completed"
            job.total_files = len(uploaded_paths)
            job.processed_files = len(uploaded_paths)
            await db.commit()

        except Exception as exc:
            logger.error("Upload job %s failed: %s", job_id, exc, exc_info=True)
            await db.rollback()
            job = await db.get(UploadJob, job_id)
            job.status = "failed"
            job.error = str(exc)
            await db.commit()

        finally:
            _job_progress.pop(job_id, None)


@router.get("/browse", response_model=PaginatedFileList)
async def browse_directory(
    path: str | None = None,
    limit: int = DEFAULT_PAGE_SIZE,
    continuation_token: str | None = None,
    user_id: str = Depends(get_current_user)
):
    """
    Browse directory contents with pagination support for large directories.
    
    Args:
        path: Directory path to browse. If None, returns root level.
        limit: Maximum number of items to return (default 50, max 200).
        continuation_token: Token from previous response to fetch next page.
        user_id: Current authenticated user (injected).
    
    Returns:
        PaginatedFileList with items, next_token, and has_more flag.
    """
    try:
        # Cap limit to prevent abuse
        limit = min(limit, MAX_ITEMS_PER_REQUEST)
        
        # return the root level (i.e. "data/") if no path is provided
        if path is None or path == "":
            result = file_service.list_directory_paginated(BASE_DIR, limit=limit, continuation_token=continuation_token)
            result.items = [fp for fp in result.items if not fp.is_hidden]
            return result

        # normalize the incoming path from the frontend
        normalized_path = normalize_path(path)

        # if this path is {DATA_DIR}, return the results under the user's data directory
        if normalized_path.rstrip("/") == normalize_path(DATA_DIR).rstrip("/"):
            normalized_path = os.path.join(DATA_DIR, user_id)
            if not file_service.exists(normalized_path):
                file_service.create_dir(normalized_path)

        # ensure that path ends with a slash for directory listing consistency
        if not normalized_path.endswith("/") and normalized_path != "":
             normalized_path += "/"

        # confirm that path exists and is a directory / s3 prefix
        if not file_service.exists(normalized_path):
            raise HTTPException(status_code=404, detail=f"Path {normalized_path} not found")
        
        if not file_service.is_dir(normalized_path):
            raise HTTPException(status_code=400, detail=f"Path {normalized_path} is not a directory or s3 prefix")

        # get paginated list of directory contents
        result = file_service.list_directory_paginated(
            normalized_path,
            limit=limit,
            continuation_token=continuation_token
        )
        
        # Filter hidden files and sort (directories first, then alphabetically)
        result.items = [item for item in result.items if not item.is_hidden]
        result.items.sort(key=lambda file: (not file.is_directory, file.path.lower()))

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error browsing directory: {str(e)}") from e


@router.post("/upload", status_code=202)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    path: str = Form(""),
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Accept a file (including archives) and immediately return a job ID.
    The actual extraction and storage upload happens as a background task so the
    HTTP connection is never held open long enough to trigger the ALB idle timeout.
    Poll GET /upload/status/{job_id} to track progress.
    """
    # Validate and resolve the destination path before reading the file bytes.
    normalized_path = normalize_path(path)

    if normalized_path.rstrip("/") == normalize_path(DATA_DIR).rstrip("/"):
        normalized_path = os.path.join(DATA_DIR, user_id)

    if normalized_path.rstrip("/") == normalize_path(BASE_DIR).rstrip("/"):
        raise HTTPException(status_code=400, detail="Cannot upload files directly to the base directory.")

    shared = (
        normalized_path.rstrip("/") == normalize_path(SHARED_DATA_DIR).rstrip("/")
        or normalized_path.startswith(SHARED_DATA_DIR)
    )

    # Read all bytes now, while the request is still open.
    file_bytes = await file.read()
    filename = file.filename

    # Persist the job record so the client can poll for status immediately.
    job_id = str(uuid.uuid4())
    db.add(UploadJob(id=job_id, user_id=user_id, status="pending"))
    await db.commit()

    # Kick off the upload after the response is sent — the connection is now free.
    background_tasks.add_task(
        _run_upload_job,
        job_id,
        file_bytes,
        filename,
        normalized_path,
        user_id,
        shared,
    )

    return {"job_id": job_id, "status": "pending"}


@router.get("/upload/status/{job_id}")
async def get_upload_status(
    job_id: str,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Return the current status of an async upload job.

    Fields:
        status        — pending | running | completed | failed
        total_files   — total number of files to upload (null until extraction finishes)
        processed_files — number of files successfully written to storage so far
        error         — error message if status is 'failed'
    """
    job = await db.get(UploadJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Upload job not found.")
    if job.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to view this upload job.")

    # Use the in-memory counter for live progress while the job is running;
    # fall back to the DB value once the job is finished.
    live_processed = _job_progress.get(job_id, job.processed_files)

    return {
        "job_id": job.id,
        "status": job.status,
        "total_files": job.total_files,
        "processed_files": live_processed,
        "error": job.error,
    }


@router.post("/delete")
async def delete_files(data: FileBatchDelete, db: AsyncSession = Depends(get_db)):
    """
    Delete a batch of files from storage and the database.
    """
    if not data.files:
        raise HTTPException(status_code=400, detail="No file paths provided for deletion.")

    # Separate items into files and directories
    files_to_delete, dirs_to_delete = [], []    
    for file_path in data.files:
        if file_service.is_dir(file_path):
            dirs_to_delete.append(file_path)
        else:
            files_to_delete.append(file_path)

    deleted_count = 0
    errors = []

    # delete individual files and their DB records first
    for file_path in files_to_delete:
        try:
            # NOTE: if there is an error in the SQL execution after this point, the file service
            #       and database will be inconsistent; we should update file_service.delete_file(file_path)
            #       to be idempotent; such that future deletions of the same path do not error out.
            file_service.delete_file(file_path)

            # delete the file record from the database; ensure the path matches the record exactly
            stmt = select(FileRecord).where(FileRecord.file_path == file_path)
            result = await db.execute(stmt)
            record = result.scalars().first()
            if record:
                await db.delete(record)
                deleted_count += 1
            else:
                # log if the record doesn't exist but continue trying others
                logger.warning(f"Database record not found for file path: {file_path}")

        except Exception as e:
            # collect errors and continue processing the rest of the batch
            errors.append({"file_path": file_path, "error": str(e)})
            logger.error(f"Failed to delete file {file_path}: {str(e)}")

    # commit deletions of individual files
    await db.commit()

    # remove the directory structures
    for dir_path in dirs_to_delete:
        try:
            file_service.delete_directory(dir_path)
        except Exception as e:
            errors.append({"directory_path": dir_path, "error": str(e)})
            logger.error(f"Failed to delete directory {dir_path}: {e}")
    
    if errors:
        # if there are errors, return a partial success/failure response
        raise HTTPException(
            status_code=400,
            detail=f"Successfully deleted {deleted_count} file(s) but failed for {len(errors)} file(s).",
            headers={"X-Deletion-Errors": str(errors)}
        )

    return {"message": f"Successfully deleted {deleted_count} file(s)."}


@router.post("/create-directory")
async def create_directory(data: DirectoryCreate, user_id: str = Depends(get_current_user)):
    """
    Create a new directory for the user.
    """
    try:
        # normalize the incoming path from the frontend
        normalized_path = normalize_path(data.path)

        # do not create directories within the BASE_DIR directly
        if normalized_path.rstrip("/") == normalize_path(BASE_DIR).rstrip("/"):
            raise HTTPException(status_code=400, detail="Cannot create directory immediately under the base directory.")

        # inject user_id if we are at the top level of DATA_DIR
        if normalized_path.rstrip("/") == normalize_path(DATA_DIR).rstrip("/"):
            normalized_path = os.path.join(normalized_path, user_id)

        # construct the final path by joining only the new directory name
        full_path = os.path.join(normalized_path, data.name)
        file_service.create_dir(full_path)

        return {"message": "Success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating directory: {str(e)}") from e


@router.post("/expand-paths")
async def expand_paths(paths: list[str], user_id: str = Depends(get_current_user)):
    """
    Expand a list of paths (files and directories) into a flat list of file paths.
    
    This is used when creating datasets from selected folders - the frontend can
    select folder paths, and this endpoint expands them to all contained files.
    
    Args:
        paths: List of file and/or directory paths to expand.
        user_id: Current authenticated user (injected).
    
    Returns:
        List of file paths (directories are expanded to their contained files).
    """
    try:
        expanded_files = set()
        
        for path in paths:
            normalized_path = normalize_path(path)
            
            if file_service.is_dir(normalized_path):
                # Expand directory to all subfiles
                subfiles = file_service.list_all_subfiles(normalized_path)
                expanded_files.update(subfiles)
            elif file_service.exists(normalized_path):
                # It's a regular file
                expanded_files.add(normalized_path)
            # Skip non-existent paths silently
        
        return {"files": list(expanded_files)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error expanding paths: {str(e)}") from e
