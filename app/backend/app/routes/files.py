import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import File as FileRecord
from app.database import get_db
from app.env import BASE_DIR, IS_LOCAL_ENV
from app.models.schemas import FileItem
from app.services.file_service import LocalFileService, S3FileService

logger = logging.getLogger('uvicorn.error')

router = APIRouter()
file_service = LocalFileService() if IS_LOCAL_ENV else S3FileService()

def normalize_path(path: str) -> str:
    """
    Normalizes a path, preserving protocol (like s3://) and ensuring consistent
    single slashes and no trailing slash (unless the path is just the root protocol).
    """
    if not path:
        return ""

    # separate protocol (e.g., s3://bucket, file://) from the path segment
    protocol = ""
    path_segment = path

    # check for protocol or local absolute path starting with "/"
    if "://" in path:
        parts = path.split("://", 1)
        protocol = parts[0] + "://"
        path_segment = parts[1]
    elif path.startswith("/"):
        # for local paths, treat the initial '/' as a special character to preserve
        path_segment = path.lstrip("/")

    # normalize the path segment: remove multiple slashes and trailing slash
    # This also handles the case where the protocol part might have had extra slashes
    path_segment = path_segment.replace("//", "/")
    path_segment = path_segment.rstrip("/")

    # recombine
    if protocol:
        # for protocols, the combination looks like: s3://bucket/key/
        return protocol + path_segment
    elif path.startswith("/") and path_segment:
        # for absolute local paths: /carnot/data
        return "/" + path_segment
    elif path.startswith("/") and not path_segment:
        # If path was just "/", return "/"
        return "/"

    return path_segment

@router.get("/browse", response_model=list[FileItem])
async def browse_directory(path: str | None = None):
    """
    Browse directory contents (uploaded files and user's data directory)
    """
    try:
        # return the root level (i.e. "data/") if no path is provided
        if path is None or path == "":
            file_items = file_service.list_directory(BASE_DIR)
            logger.info(
                f"browse directory | Path: {path} | BASE_DIR: {BASE_DIR} | Returning: {file_items}"
            )
            return file_items

        # ✨ FIX: Normalize the incoming path from the frontend
        normalized_path = normalize_path(path)

        # confirm that path exists and is a directory / s3 prefix
        if not file_service.exists(normalized_path):
            raise HTTPException(status_code=404, detail=f"Path {normalized_path} not found")
        
        if not file_service.is_dir(normalized_path):
            raise HTTPException(status_code=400, detail=f"Path {normalized_path} is not a directory or s3 prefix")

        # get list of directory contents and then sort them so that directories come first
        items = file_service.list_directory(normalized_path)
        items.sort(key=lambda file: (not file.is_directory, file.path.lower()))

        logger.info(
            f"browse directory | Path: {path} | normalized_path: {normalized_path} | Returning: {items}"
        )

        return items

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error browsing directory: {str(e)}") from e


@router.post("/upload")
async def upload_file(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    """
    Upload a file to the server.
    """
    try:
        # save file to file system
        file_paths = file_service.save_uploaded_file(file)

        # store file metadata in database
        uploaded_files = [FileRecord(file_path=file_path) for file_path in file_paths]
        db.add_all(uploaded_files)
        await db.commit()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}") from e


@router.get("/upload")
async def list_uploaded_files(db: AsyncSession = Depends(get_db)):
    """
    List all uploaded files
    """
    try:
        result = await db.execute(select(FileRecord).order_by(FileRecord.upload_date.desc()))
        files = result.scalars().all()
        return [
            {
                "id": f.id,
                "file_path": f.file_path,
                "upload_date": f.upload_date
            }
            for f in files
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}") from e
