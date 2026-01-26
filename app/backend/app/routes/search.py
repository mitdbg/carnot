import io
import logging
import os
from pathlib import Path

import palimpzest as pz
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database import get_db
from app.env import DATA_DIR, IS_LOCAL_ENV, SHARED_DATA_DIR
from app.models.schemas import SearchQuery, SearchResult
from app.services.file_service import LocalFileService, S3FileService, virtualize_filepath
from app.services.llm import get_user_llm_config

router = APIRouter()
file_service = LocalFileService() if IS_LOCAL_ENV else S3FileService()
logger = logging.getLogger('uvicorn.error')

SKIP_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".zip", ".exe", ".bin"}

class TextFileWithPath(BaseModel):
    file_path: str
    file_name: str
    contents: str

def get_text_from_pdf(pdf_bytes):
    pdf = PdfReader(io.BytesIO(pdf_bytes))
    all_text = ""
    for page in pdf.pages:
        all_text += page.extract_text() + "\n"
    return all_text

class RecursiveTextFileDataset(pz.IterDataset):
    def __init__(self, id: str, paths: list[str] | None, user_id: str) -> None:
        paths = paths or [os.path.join(DATA_DIR, user_id), SHARED_DATA_DIR]
        super().__init__(id=id, schema=TextFileWithPath)
        self.filepaths = [
            fp 
            for path in paths
            for fp in file_service.list_all_subfiles(path)
            if Path(fp).suffix.lower() not in SKIP_SUFFIXES
        ]

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> dict:
        filepath = self.filepaths[idx]
        filename = os.path.basename(filepath)

        try:
            if filepath.endswith(".pdf"):
                pdf_contents = file_service.read_file(filepath, bytes=True)
                contents = get_text_from_pdf(pdf_contents)
            else:
                contents = file_service.read_file(filepath)
        except Exception:
            contents = ""

        return {
            "file_path": filepath,
            "file_name": filename,
            "contents": contents,
        }


def run_semantic_search(query_text: str, search_paths: list[str] | None, user_id: str, user_config: dict) -> list[SearchResult]:
    os.environ["OPENAI_API_KEY"] = user_config.get("OPENAI_API_KEY", "")
    ds = RecursiveTextFileDataset(id="search", paths=search_paths, user_id=user_id)
    ds = ds.sem_filter(f"The file matches the query: {query_text}")
    config = pz.QueryProcessorConfig(
        policy=pz.MaxQuality(),
        llm_config=user_config,
        progress=False,
    )

    output_ds = ds.run(config=config)

    results = []
    for item in output_ds:
        content = getattr(item, "contents", "")
        snippet = content[:200] + "..." if len(content) > 200 else content
        file_path = getattr(item, "file_path", "")
        results.append(
            SearchResult(
                file_path=file_path,
                file_name=getattr(item, "file_name", ""),
                virtual_path=virtualize_filepath(file_path),
                relevance_score=1.0,
                snippet=snippet,
            )
        )
    return results


@router.post("/", response_model=list[SearchResult])
async def search_files(
    query: SearchQuery,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # retrieve user LLM config
        user_config = await get_user_llm_config(db, user_id)
        if not user_config:
            raise HTTPException(
                status_code=400, 
                detail="No LLM API keys found for this user."
            )

        search_paths = query.paths or None
        results: list[SearchResult] = run_semantic_search(query.query, search_paths, user_id, user_config)

        unique: list[SearchResult] = []
        seen_paths = set()
        for result in results:
            if result.file_path not in seen_paths:
                seen_paths.add(result.file_path)
                unique.append(result)

        return unique

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=f"Error searching files: {exc}") from exc
