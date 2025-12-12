"""Progress event logging for web session tracking."""
import json
import logging
from dataclasses import dataclass
from enum import Enum

import fsspec

logger = logging.getLogger(__name__)


class ProgressLevel(str, Enum):
    OUTER = "outer"  
    INNER = "inner"  


class ProgressStatus(str, Enum):
    STARTED = "started"
    PROGRESS = "progress"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ProgressEvent:
    timestamp: str
    session_id: str
    level: ProgressLevel | str
    operator_id: str
    operator_name: str
    status: ProgressStatus | str
    current: int = 0
    total: int = 0
    percentage: float = 0.0
    cost: float = 0.0
    message: str = ""
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "level": self.level.value if isinstance(self.level, ProgressLevel) else self.level,
            "operator_id": self.operator_id,
            "operator_name": self.operator_name,
            "status": self.status.value if isinstance(self.status, ProgressStatus) else self.status,
            "progress": {
                "current": self.current,
                "total": self.total,
                "percentage": self.percentage,
            },
            "cost": self.cost,
            "message": self.message,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class ProgressLogger:
    def __init__(self, log_file_path: str | None):
        self.log_file_path = log_file_path
        self.file_handle = None
        self.file_stream = None
        if self.log_file_path:
            self._open_stream()

    def _open_stream(self):
        """Open the stream using fsspec in append mode ('a')."""
        try:
            fs = fsspec.filesystem("s3" if self.log_file_path.startswith("s3://") else "file")
            self.file_handle = fs.open(self.log_file_path, mode='a', encoding='utf-8')
            self.file_stream = self.file_handle.__enter__()

        except Exception:
            raise

    def log_event(self, event: ProgressEvent):
        if not self.file_stream:
            return

        try:
            # write and flush the new event data
            self.file_stream.write(event.to_json() + '\n')
            self.file_stream.flush()

        except Exception:
            pass

    def close(self):
        """Close the file stream, committing the final data chunk to S3 if applicable."""
        if self.file_handle:
            self.file_handle.__exit__(None, None, None)
            self.file_handle = None
            self.file_stream = None


# NOTE: keeping old class definition w/locking around in case we need it later
# class ProgressLogger:
#     def __init__(self, log_file_path: str | None):
#         self.log_file_path = Path(log_file_path) if log_file_path else None
#         if self.log_file_path:
#             # Ensure parent directory exists
#             self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

#     def log_event(self, event: ProgressEvent):
#         if not self.log_file_path:
#             return

#         try:
#             with open(self.log_file_path, 'a') as f:
#                 fcntl.flock(f.fileno(), fcntl.LOCK_EX)
#                 try:
#                     f.write(event.to_json() + '\n')
#                     f.flush()
#                 finally:
#                     fcntl.flock(f.fileno(), fcntl.LOCK_UN)
#         except Exception as e:
#             logger.warning(f"Failed to write progress event: {e}")
