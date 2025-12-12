import logging
import os
import shutil
import tarfile
import zipfile
from abc import ABC, abstractmethod
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import IO

import boto3
from cloudpathlib import S3Path
from fastapi import HTTPException, UploadFile

from app.env import BASE_DIR, DATA_DIR
from app.models.schemas import FileItem

logger = logging.getLogger('uvicorn.error')

# predefined set of archive extensions
ARCHIVE_EXTENSIONS = (
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz",
    ".tar.xz",
    ".txz",
)

# TODO: investigate using fsspec to avoid (completely) reinventing our own file service?

def _safe_join(base: Path, target: Path) -> Path:
    resolved_base = base.resolve()
    resolved_target = target.resolve()
    try:
        resolved_target.relative_to(resolved_base)
    except ValueError as err:
        raise HTTPException(status_code=400, detail="Archive contains unsafe paths") from err
    return resolved_target


def _extract_zip_to_streams(archive_path: Path, data_dir: Path | S3Path) -> tuple[list[IO[bytes]], list[str]]:
    """Extracts a ZIP archive and returns a list of BytesIO streams for file contents."""
    streams, upload_paths = [], []
    try:
        with zipfile.ZipFile(archive_path) as zip_ref:
            for member in zip_ref.infolist():
                if member.is_dir():
                    continue

                # check for unsafe paths; will throw an exception if unsafe
                fake_destination = Path("/tmp/uploads/extracted")
                member_path = fake_destination / member.filename
                _ = _safe_join(fake_destination, member_path)

                # read content into an in-memory buffer
                with zip_ref.open(member, 'r') as member_file:
                    content = member_file.read()
                    streams.append(BytesIO(content))
                    upload_paths.append(str(data_dir / member.filename))
                    
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail=f"Invalid ZIP archive: {exc}") from exc
        
    return streams, upload_paths


def _extract_tar_to_streams(archive_path: Path, data_dir: Path | S3Path) -> tuple[list[IO[bytes]], list[str]]:
    """Extracts a TAR archive and returns a list of BytesIO streams for file contents."""
    streams, upload_paths = [], []
    try:
        with tarfile.open(archive_path, "r:*") as tar_ref:
            for member in tar_ref.getmembers():
                if member.isdir():
                    continue

                # check for unsafe paths; will throw an exception if unsafe
                fake_destination = Path("/tmp/uploads/extracted")
                member_path = fake_destination / member.name
                _ = _safe_join(fake_destination, member_path)

                # read content into an in-memory buffer
                file_obj = tar_ref.extractfile(member)
                if file_obj:
                    content = file_obj.read()
                    streams.append(BytesIO(content))
                    upload_paths.append(str(data_dir / member.name))
                    
    except tarfile.TarError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid TAR archive: {exc}") from exc
        
    return streams, upload_paths


def _extract_archive(archive: UploadFile, data_dir: str) -> tuple[list[IO[bytes]], list[str]]:
    try:
        # copy archive to a temporary location
        temp_dir = Path("/tmp/uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_archive_path = temp_dir / archive.filename
        with open(temp_archive_path, "wb") as buffer:
            shutil.copyfileobj(archive.file, buffer)

        # extract archive contents to in-memory streams
        file_streams = []
        data_dir_path = S3Path(data_dir) if data_dir.startswith("s3://") else Path(data_dir)
        if archive.filename.lower().endswith(".zip"):
            file_streams, upload_paths = _extract_zip_to_streams(temp_archive_path, data_dir_path)
        else:
            file_streams, upload_paths = _extract_tar_to_streams(temp_archive_path, data_dir_path)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to extract archive: {exc}") from exc
    finally:
        # clean up temporary archive file
        if temp_archive_path.exists():
            os.remove(temp_archive_path)

    return file_streams, upload_paths


class BaseFileService(ABC):
    """Abstract base class for file services"""

    @abstractmethod
    def exists(self, path: str) -> bool:
        pass

    @abstractmethod
    def create_dir(self, path: str) -> None:
        pass

    @abstractmethod
    def is_dir(self, path: str) -> bool:
        pass

    @abstractmethod
    def list_directory(self, path: str) -> list[FileItem]:
        pass

    @abstractmethod
    def list_all_subfiles(self, path: str) -> list[str]:
        pass

    @abstractmethod
    def read_file(self, path: str) -> str:
        pass

    @abstractmethod
    def _write_file_to_path(self, file: UploadFile, path: str) -> None:
        pass

    def save_uploaded_file(self, file: UploadFile) -> list[str]:
        """Save an uploaded file to the upload directory. Returns the uploaded file paths."""
        # get list of files and the paths they will be uploaded to
        file_bytes_streams, upload_paths = [file.file], [os.path.join(DATA_DIR, file.filename)]
        if any(file.filename.lower().endswith(ext) for ext in ARCHIVE_EXTENSIONS):
            file_bytes_streams, upload_paths = _extract_archive(file, DATA_DIR)

        # TODO: handle case where name collision occurs
        for path in upload_paths:
            if self.exists(path):
                pass

        # upload files
        for file_bytes_stream, path in zip(file_bytes_streams, upload_paths, strict=True):
            self._write_file_to_path(file_bytes_stream, path)

        return upload_paths


class LocalFileService(BaseFileService):
    """File service for local filesystem"""
    def exists(self, path: str) -> bool:
        return os.path.exists(path)

    def create_dir(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

    def is_dir(self, path: str) -> bool:
        return os.path.isdir(path)

    def list_directory(self, path: str) -> list[FileItem]:
        items = []
        for entry in os.listdir(path):
            entry_path = os.path.join(path, entry)
            display_name = entry.split(BASE_DIR)[-1].lstrip("/")
            is_dir = os.path.isdir(entry_path)
            stat = os.stat(entry_path)
            items.append(FileItem(
                path=entry_path,
                display_name=display_name,
                is_directory=is_dir,
                size=stat.st_size if not is_dir else None,
                modified=datetime.fromtimestamp(stat.st_mtime)
            ))

        return items

    def list_all_subfiles(self, path: str) -> list[str]:
        """
        NOTE: this method assumes path is an absolute path; without this assumption file_paths
        will not have correct absolute paths.
        """
        file_paths = []
        for root, _, files in os.walk(path):
            for file in files:
                file_paths.append(os.path.join(root, file))
        return file_paths

    def read_file(self, path: str) -> str:
        with open(path, encoding="utf-8") as f:
            return f.read()

    def _write_file_to_path(self, file_bytes_stream: IO[bytes], path: str) -> None:
        """Save an uploaded file to the given path"""
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file_bytes_stream, buffer)


class S3FileService(BaseFileService):
    """File service for AWS S3"""
    def __init__(self):
        self.s3 = boto3.client('s3')
        self.s3_bucket = DATA_DIR.replace("s3://", "").split("/")[0]

    def _get_s3_key_from_path(self, path: str) -> str:
        return "/".join(path.replace("s3://", "").split("/")[1:])

    def exists(self, path: str) -> bool:
        """Check if a file or directory exists"""
        s3_prefix = self._get_s3_key_from_path(path)
        response = self.s3.list_objects_v2(Bucket=self.s3_bucket, Prefix=s3_prefix, MaxKeys=1)
        return 'Contents' in response

    def create_dir(self, path: str) -> None:
        """S3 prefixes are created implicitly on file upload"""
        pass

    def is_dir(self, path: str) -> bool:
        """Check if a path is a directory; S3 prefixes are always treated as directories"""
        return True

    def list_directory(self, path: str) -> list[FileItem]:
        """List contents of an s3 prefix"""
        items = []
        prefix = self._get_s3_key_from_path(path)
        paginator = self.s3.get_paginator('list_objects_v2')
        result_iterator = paginator.paginate(Bucket=self.s3_bucket, Prefix=prefix, Delimiter='/')

        for page in result_iterator:
            # CommonPrefixes contains "directories"
            for prefix in page.get('CommonPrefixes', []):
                path = f"s3://{self.s3_bucket}/{prefix['Prefix']}"
                display_name = path.split(BASE_DIR)[-1].lstrip("/")
                items.append(FileItem(
                    path=path,
                    display_name=display_name,
                    is_directory=True,
                ))

            # Contents contains files
            for obj in page.get('Contents', []):
                if obj['Key'] == prefix:
                    continue
                path = f"s3://{self.s3_bucket}/{obj['Key']}"
                display_name = path.split(BASE_DIR)[-1].lstrip("/")
                items.append(FileItem(
                    path=path,
                    display_name=display_name,
                    is_directory=False,
                    size=obj['Size'],
                    modified=obj['LastModified'],
                ))

        return items

    def list_all_subfiles(self, path: str) -> list[str]:
        """List all files under the given s3 prefix"""
        file_paths = []
        prefix = self._get_s3_key_from_path(path)
        paginator = self.s3.get_paginator('list_objects_v2')
        result_iterator = paginator.paginate(Bucket=self.s3_bucket, Prefix=prefix)

        for page in result_iterator:
            for obj in page.get('Contents', []):
                file_paths.append(f"s3://{self.s3_bucket}/{obj['Key']}")

        return file_paths

    def read_file(self, path: str) -> str:
        """Read the contents of a file from s3"""
        s3_key = self._get_s3_key_from_path(path)
        response = self.s3.get_object(Bucket=self.s3_bucket, Key=s3_key)
        content = response['Body'].read().decode('utf-8')
        return content

    def _write_file_to_path(self, file_bytes_stream: IO[bytes], path: str) -> None:
        """Save an uploaded file to the given s3 path"""
        s3_key = self._get_s3_key_from_path(path)
        self.s3.upload_fileobj(file_bytes_stream, self.s3_bucket, s3_key)
