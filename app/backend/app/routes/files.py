import logging
import os

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.database import File as FileRecord
from app.database import get_db
from app.env import BASE_DIR, DATA_DIR, IS_LOCAL_ENV, SHARED_DATA_DIR
from app.models.schemas import DirectoryCreate, FileBatchDelete, PaginatedFileList
from app.services.file_service import DEFAULT_PAGE_SIZE, LocalFileService, S3FileService, normalize_path

logger = logging.getLogger('uvicorn.error')

router = APIRouter()
file_service = LocalFileService() if IS_LOCAL_ENV else S3FileService()

# Max items to return in a single page to prevent overload
MAX_ITEMS_PER_REQUEST = 200


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


@router.post("/upload")
async def upload_file(file: UploadFile = File(...), path: str = Form(""), user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """
    Upload a file to the server.
    """
    try:
        # normalize the incoming path from the frontend
        normalized_path = normalize_path(path)

        # if this path is {DATA_DIR}, then upload the file under the user's data directory
        if normalized_path.rstrip("/") == normalize_path(DATA_DIR).rstrip("/"):
            normalized_path = os.path.join(DATA_DIR, user_id)

        # do not upload files within the BASE_DIR directly
        if normalized_path.rstrip("/") == normalize_path(BASE_DIR).rstrip("/"):
            raise HTTPException(status_code=400, detail="Cannot upload files directly to the base directory.")

        # save file to file system
        file_paths = file_service.save_uploaded_file(file, normalized_path)

        # determine if the file is shared based on the provided path
        shared = (
            normalized_path.rstrip("/") == normalize_path(SHARED_DATA_DIR).rstrip("/")
            or normalized_path.startswith(SHARED_DATA_DIR)
        )

        # store file metadata in database
        uploaded_files = [
            FileRecord(user_id=user_id, file_path=file_path, shared=shared)
            for file_path in file_paths
        ]
        db.add_all(uploaded_files)
        await db.commit()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}") from e


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
