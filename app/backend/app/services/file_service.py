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

from app.env import BASE_DIR, DATA_DIR, SHARED_DATA_DIR
from app.models.schemas import FileItem, PaginatedFileList

logger = logging.getLogger('uvicorn.error')

# Default page size for paginated directory listings
DEFAULT_PAGE_SIZE = 50

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
def normalize_path(path: str) -> str:
    """
    Normalizes a path, preserving protocol (like s3://) and ensuring consistent single slashes.
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

    # recombine
    if protocol:
        # for protocols, the combination looks like: s3://bucket/key/
        return protocol + path_segment
    elif path.startswith("/") and path_segment:
        # for absolute local paths: /carnot/data/
        return "/" + path_segment
    elif path.startswith("/") and not path_segment:
        # If path was just "/", return "/"
        return "/"

    return path_segment

def virtualize_filepath(path: str) -> str:
    """
    Convert absolute file paths to virtual paths which:
    - remove the BASE_DIR prefix
    - remove the user-id from non-shared paths

    The typical path structure is:
    - {BASE_DIR}/data/{user_id}/path/to/file.txt  for user-specific files
    - {BASE_DIR}/shared/path/to/file.txt          for shared files

    Instead, we want to return:
    - data/path/to/file.txt    for user-specific files
    - shared/path/to/file.txt  for shared files
    """
    normalized_path = normalize_path(path)
    normalized_base_dir = normalize_path(BASE_DIR).rstrip("/")
    normalized_data_dir = normalize_path(DATA_DIR).rstrip("/")
    normalized_shared_data_dir = normalize_path(SHARED_DATA_DIR).rstrip("/")

    if normalized_path.startswith(normalized_shared_data_dir):
        virtual_path = normalized_path.replace(normalized_base_dir + "/", "", 1)
    elif normalized_path.startswith(normalized_data_dir):
        relative_path = normalized_path.replace(normalized_data_dir + "/", "", 1)
        # remove the user_id segment
        parts = relative_path.split("/", 1)
        virtual_path = "data/" + parts[1] if len(parts) == 2 else "data/"
    else:
        virtual_path = normalized_path.replace(normalized_base_dir + "/", "", 1)

    return virtual_path


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
                if member.is_dir() or "__MACOSX" in member.filename:
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
                if member.isdir() or "__MACOSX" in member.name:
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

        # get archive filename without extension
        archive_name = None
        for ext in ARCHIVE_EXTENSIONS:
            if archive.filename.lower().endswith(ext):
                archive_name = archive.filename.split(ext)[0]
                break

        # extract archive contents to in-memory streams
        file_streams = []
        data_dir_path = S3Path(data_dir) if data_dir.startswith("s3://") else Path(data_dir)
        data_dir_path = data_dir_path / archive_name
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
    def list_directory_paginated(
        self,
        path: str,
        limit: int = DEFAULT_PAGE_SIZE,
        continuation_token: str | None = None
    ) -> PaginatedFileList:
        """List directory contents with pagination support for large directories."""
        pass

    @abstractmethod
    def delete_directory(self, path: str) -> None:
        pass

    @abstractmethod
    def list_all_subfiles(self, path: str) -> list[str]:
        pass

    @abstractmethod
    def delete_file(self, path: str) -> None:
        pass

    @abstractmethod
    def read_file(self, path: str, bytes: bool = False) -> str:
        pass

    @abstractmethod
    def _write_file_to_path(self, file: UploadFile, path: str) -> None:
        pass

    def save_uploaded_file(self, file: UploadFile, upload_dir: str) -> list[str]:
        """Save an uploaded file to the upload directory. Returns the uploaded file path(s)."""
        # get list of files and the paths they will be uploaded to
        file_bytes_streams, upload_paths = [file.file], [os.path.join(upload_dir, file.filename)]
        if any(file.filename.lower().endswith(ext) for ext in ARCHIVE_EXTENSIONS):
            file_bytes_streams, upload_paths = _extract_archive(file, upload_dir)

        # TODO: handle case where name collision occurs
        for path in upload_paths:
            if self.exists(path):
                pass

        # upload files
        for file_bytes_stream, path in zip(file_bytes_streams, upload_paths, strict=True):
            dir = os.path.dirname(path)
            if not self.exists(dir):
                self.create_dir(dir)
            self._write_file_to_path(file_bytes_stream, path)

        return upload_paths


class LocalFileService(BaseFileService):
    """File service for local filesystem"""
    def __init__(self):
        self.create_dir(DATA_DIR)
        self.create_dir(SHARED_DATA_DIR)

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
            is_dir = os.path.isdir(entry_path)
            stat = os.stat(entry_path)
            items.append(FileItem(
                path=entry_path,
                display_name=entry + ("/" if is_dir else ""),
                is_directory=is_dir,
                is_hidden=entry.startswith("."),
                size=stat.st_size if not is_dir else None,
                modified=datetime.fromtimestamp(stat.st_mtime)
            ))

        return items

    def list_directory_paginated(
        self,
        path: str,
        limit: int = DEFAULT_PAGE_SIZE,
        continuation_token: str | None = None
    ) -> PaginatedFileList:
        """List directory contents with pagination for local filesystem."""
        entries = sorted(os.listdir(path))
        total_count = len(entries)
        
        # Decode continuation token (simple offset-based for local)
        start_idx = 0
        if continuation_token:
            try:
                import base64
                start_idx = int(base64.b64decode(continuation_token).decode('utf-8'))
            except (ValueError, TypeError):
                start_idx = 0
        
        # Slice entries for this page
        end_idx = min(start_idx + limit, total_count)
        page_entries = entries[start_idx:end_idx]
        
        items = []
        for entry in page_entries:
            entry_path = os.path.join(path, entry)
            is_dir = os.path.isdir(entry_path)
            stat = os.stat(entry_path)
            items.append(FileItem(
                path=entry_path,
                display_name=entry + ("/" if is_dir else ""),
                is_directory=is_dir,
                is_hidden=entry.startswith("."),
                size=stat.st_size if not is_dir else None,
                modified=datetime.fromtimestamp(stat.st_mtime)
            ))
        
        # Generate next token if more items exist
        next_token = None
        has_more = end_idx < total_count
        if has_more:
            import base64
            next_token = base64.b64encode(str(end_idx).encode('utf-8')).decode('utf-8')
        
        return PaginatedFileList(
            items=items,
            next_token=next_token,
            total_count=total_count if start_idx == 0 else None,
            has_more=has_more
        )

    def delete_directory(self, path: str) -> None:
        if self.is_dir(path):
            shutil.rmtree(path)

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

    def delete_file(self, path: str) -> None:
        os.remove(path)

    def read_file(self, path: str, bytes: bool = False) -> str:
        read_kwargs = {"mode": "rb"} if bytes else {"encoding": "utf-8"}
        with open(path, **read_kwargs) as file:
            content = file.read()
        return content

    def _write_file_to_path(self, file_bytes_stream: IO[bytes], path: str) -> None:
        """Save an uploaded file to the given path"""
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file_bytes_stream, buffer)


class S3FileService(BaseFileService):
    """File service for AWS S3"""
    def __init__(self):
        self.s3 = boto3.client('s3')
        self.s3_bucket = DATA_DIR.replace("s3://", "").split("/")[0]

        self.create_dir(DATA_DIR)
        self.create_dir(SHARED_DATA_DIR)

    def _get_s3_key_from_path(self, path: str) -> str:
        return "/".join(path.replace("s3://", "").split("/")[1:])

    def exists(self, path: str) -> bool:
        """Check if a file or directory exists"""
        s3_prefix = self._get_s3_key_from_path(path)
        response = self.s3.list_objects_v2(Bucket=self.s3_bucket, Prefix=s3_prefix, MaxKeys=1)
        return 'Contents' in response

    def create_dir(self, path: str) -> None:
        """Place an empty object to represent a directory in s3"""
        s3_key = self._get_s3_key_from_path(path).rstrip("/") + "/"
        self.s3.put_object(Bucket=self.s3_bucket, Key=s3_key)

    def is_dir(self, path: str) -> bool:
        """Check if a path is a directory; S3 prefixes are always treated as directories"""
        return path.endswith("/")

    def list_directory(self, path: str) -> list[FileItem]:
        """List contents of an s3 prefix"""
        items = []
        prefix = self._get_s3_key_from_path(path).rstrip("/") + "/"
        prefix = "" if prefix == "/" else prefix
        paginator = self.s3.get_paginator('list_objects_v2')
        result_iterator = paginator.paginate(Bucket=self.s3_bucket, Prefix=prefix, Delimiter='/')

        for page in result_iterator:
            # CommonPrefixes contains "directories"
            for cp in page.get('CommonPrefixes', []):
                path = f"s3://{self.s3_bucket}/{cp['Prefix']}"
                display_name = path.rstrip("/").split("/")[-1]
                items.append(FileItem(
                    path=path,
                    display_name=display_name.rstrip("/") + "/",
                    is_directory=True,
                    is_hidden=display_name.startswith("."),
                ))

            # Contents contains files
            for obj in page.get('Contents', []):
                if obj['Key'] == prefix:
                    continue
                path = f"s3://{self.s3_bucket}/{obj['Key']}"
                display_name = path.rstrip("/").split("/")[-1]
                items.append(FileItem(
                    path=path,
                    display_name=display_name,
                    is_directory=False,
                    is_hidden=display_name.startswith("."),
                    size=obj['Size'],
                    modified=obj['LastModified'],
                ))

        return items

    def list_directory_paginated(
        self,
        path: str,
        limit: int = DEFAULT_PAGE_SIZE,
        continuation_token: str | None = None
    ) -> PaginatedFileList:
        """List contents of an S3 prefix with pagination support.
        
        Uses S3's native ContinuationToken for efficient pagination without
        fetching all objects first.
        """
        items = []
        prefix = self._get_s3_key_from_path(path).rstrip("/") + "/"
        prefix = "" if prefix == "/" else prefix
        
        # Build request parameters
        request_params = {
            'Bucket': self.s3_bucket,
            'Prefix': prefix,
            'Delimiter': '/',
            'MaxKeys': limit
        }
        
        if continuation_token:
            request_params['ContinuationToken'] = continuation_token
        
        # Make single paginated request
        response = self.s3.list_objects_v2(**request_params)
        
        # CommonPrefixes contains "directories"
        for cp in response.get('CommonPrefixes', []):
            dir_path = f"s3://{self.s3_bucket}/{cp['Prefix']}"
            display_name = dir_path.rstrip("/").split("/")[-1]
            items.append(FileItem(
                path=dir_path,
                display_name=display_name.rstrip("/") + "/",
                is_directory=True,
                is_hidden=display_name.startswith("."),
            ))

        # Contents contains files
        for obj in response.get('Contents', []):
            if obj['Key'] == prefix:
                continue
            file_path = f"s3://{self.s3_bucket}/{obj['Key']}"
            display_name = file_path.rstrip("/").split("/")[-1]
            items.append(FileItem(
                path=file_path,
                display_name=display_name,
                is_directory=False,
                is_hidden=display_name.startswith("."),
                size=obj['Size'],
                modified=obj['LastModified'],
            ))
        
        # Check if there are more results
        has_more = response.get('IsTruncated', False)
        next_token = response.get('NextContinuationToken') if has_more else None
        
        # Note: S3 doesn't provide total count without listing all objects
        # We only provide it if we know we're at the end
        total_count = len(items) if not has_more and not continuation_token else None
        
        return PaginatedFileList(
            items=items,
            next_token=next_token,
            total_count=total_count,
            has_more=has_more
        )

    def delete_directory(self, path: str) -> None:
        """Delete a directory (prefix) from s3"""
        prefix = self._get_s3_key_from_path(path).rstrip("/") + "/"
        paginator = self.s3.get_paginator('list_objects_v2')
        result_iterator = paginator.paginate(Bucket=self.s3_bucket, Prefix=prefix)

        objects_to_delete = []
        for page in result_iterator:
            for obj in page.get('Contents', []):
                objects_to_delete.append({'Key': obj['Key']})

        # batch delete
        for i in range(0, len(objects_to_delete), 1000):
            batch = objects_to_delete[i:i + 1000]
            self.s3.delete_objects(Bucket=self.s3_bucket, Delete={'Objects': batch})

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

    def delete_file(self, path: str) -> None:
        """Delete a file from s3"""
        s3_key = self._get_s3_key_from_path(path)
        self.s3.delete_object(Bucket=self.s3_bucket, Key=s3_key)

    def read_file(self, path: str, bytes: bool = False) -> str:
        """Read the contents of a file from s3"""
        s3_key = self._get_s3_key_from_path(path)
        response = self.s3.get_object(Bucket=self.s3_bucket, Key=s3_key)
        content = response['Body'].read().decode('utf-8')
        return content if bytes else content.decode('utf-8')

    def _write_file_to_path(self, file_bytes_stream: IO[bytes], path: str) -> None:
        """Save an uploaded file to the given s3 path"""
        s3_key = self._get_s3_key_from_path(path)
        self.s3.upload_fileobj(file_bytes_stream, self.s3_bucket, s3_key)
