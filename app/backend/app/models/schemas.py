from datetime import datetime

from pydantic import BaseModel


# Config schemas
class AppConfig(BaseModel):
    base_dir: str
    data_dir: str
    shared_data_dir: str

# Settings schemas
class AppSettings(BaseModel):
    env_filepath: str | None = None

# File schemas
class FileItem(BaseModel):
    path: str
    display_name: str
    is_directory: bool
    is_hidden: bool = False
    size: int | None = None
    modified: datetime | None = None


class PaginatedFileList(BaseModel):
    """Paginated response for file browsing with large directories."""
    items: list[FileItem]
    next_token: str | None = None
    total_count: int | None = None  # Only provided on first page if available
    has_more: bool = False


class FileBatchDelete(BaseModel):
    files: list[str]

class DirectoryCreate(BaseModel):
    path: str
    name: str

# Dataset schemas
class DatasetCreate(BaseModel):
    name: str
    shared: bool
    annotation: str
    files: list[str]

class DatasetFileResponse(BaseModel):
    id: int
    file_path: str
    file_name: str
    
    class Config:
        from_attributes = True

class DatasetResponse(BaseModel):
    id: int
    name: str
    annotation: str
    created_at: datetime
    updated_at: datetime
    file_count: int = 0
    
    class Config:
        from_attributes = True

class DatasetDetailResponse(BaseModel):
    id: int
    name: str
    annotation: str
    created_at: datetime
    updated_at: datetime
    files: list[DatasetFileResponse]
    
    class Config:
        from_attributes = True

class DatasetUpdate(BaseModel):
    name: str | None = None
    annotation: str | None = None
    files: list[str] | None = None

# Search schemas
class SearchQuery(BaseModel):
    query: str
    paths: list[str] | None = None

class SearchResult(BaseModel):
    file_path: str
    file_name: str
    virtual_path: str
    relevance_score: float | None = None
    snippet: str | None = None
