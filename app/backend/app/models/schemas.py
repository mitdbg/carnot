from datetime import datetime

from pydantic import BaseModel


# Config schemas
class AppConfig(BaseModel):
    base_dir: str

# Settings schemas
class AppSettings(BaseModel):
    env_filepath: str | None = None

# File schemas
class FileItem(BaseModel):
    path: str
    display_name: str
    is_directory: bool
    size: int | None = None
    modified: datetime | None = None

class FileBatchDelete(BaseModel):
    files: list[str]

# Dataset schemas
class DatasetCreate(BaseModel):
    name: str
    user_id: str
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
    relevance_score: float | None = None
    snippet: str | None = None
