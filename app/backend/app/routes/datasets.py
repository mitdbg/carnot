import os

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.database import Dataset, DatasetFile, get_db
from app.database import File as FileRecord
from app.models.schemas import (
    DatasetCreate,
    DatasetDetailResponse,
    DatasetResponse,
    DatasetUpdate,
)

router = APIRouter()

@router.get("/", response_model=list[DatasetResponse])
async def list_datasets(user_id: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """
    List all datasets with file count
    """
    try:
        # Query datasets with file count
        result = await db.execute(
            select(
                Dataset,
                func.count(DatasetFile.file_id).label("file_count")
            )
            .where(or_(Dataset.user_id == user_id, Dataset.shared))
            .outerjoin(DatasetFile, Dataset.id == DatasetFile.dataset_id)
            .group_by(Dataset.id)
            .order_by(Dataset.created_at.desc())
        )

        datasets = []
        for dataset, file_count in result:
            datasets.append(DatasetResponse(
                id=dataset.id,
                name=dataset.name,
                annotation=dataset.annotation,
                created_at=dataset.created_at,
                updated_at=dataset.updated_at,
                file_count=file_count or 0
            ))

        return datasets
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing datasets: {str(e)}") from e

@router.post("/", response_model=DatasetDetailResponse)
async def create_dataset(
    dataset: DatasetCreate,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new dataset
    """
    try:
        # Check if dataset name already exists
        result = await db.execute(
            select(Dataset).where(
                Dataset.name == dataset.name,
                or_(Dataset.user_id == user_id, Dataset.shared)
            )
        )
        existing = result.scalar_one_or_none()

        if existing:
            raise HTTPException(status_code=400, detail="Dataset name already exists")

        # Create dataset and write to DB to get db_dataset.id
        db_dataset = Dataset(
            name=dataset.name,
            user_id=user_id,
            shared=dataset.shared,
            annotation=dataset.annotation
        )
        db.add(db_dataset)
        await db.flush()

        # Add files (expand directories if needed)
        dataset_files = []
        for filepath in dataset.files:
            # Get or create File record
            file_result = await db.execute(
                select(FileRecord).where(FileRecord.file_path == filepath)
            )
            db_file_record = file_result.scalar_one_or_none()

            if not db_file_record:
                db_file_record = FileRecord(file_path=filepath, user_id=user_id)
                db.add(db_file_record)
                await db.flush()

            # Create DatasetFile junction record
            dataset_file = DatasetFile(
                dataset_id=db_dataset.id,
                file_id=db_file_record.id,
            )
            db.add(dataset_file)
            dataset_files.append(dataset_file)

        if not dataset_files:
            raise HTTPException(status_code=400, detail="No valid files found in selection")

        await db.commit()
        await db.refresh(db_dataset)

        # Fetch files for response with join to File table
        result = await db.execute(
            select(DatasetFile, FileRecord)
            .join(FileRecord, DatasetFile.file_id == FileRecord.id)
            .where(DatasetFile.dataset_id == db_dataset.id)
        )
        file_rows = result.all()

        return DatasetDetailResponse(
            id=db_dataset.id,
            name=db_dataset.name,
            annotation=db_dataset.annotation,
            created_at=db_dataset.created_at,
            updated_at=db_dataset.updated_at,
            files=[
                {
                    "id": file.id,
                    "file_path": file.file_path,
                    "file_name": os.path.basename(file.file_path),
                }
                for _, file in file_rows
            ]
        )

    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Error creating dataset: {str(e)}") from e

@router.get("/{dataset_id}", response_model=DatasetDetailResponse)
async def get_dataset(
    dataset_id: int,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get dataset details with files
    """
    try:
        # Get dataset
        result = await db.execute(
            select(Dataset).where(
                Dataset.id == dataset_id,
                or_(Dataset.user_id == user_id, Dataset.shared)
            )
        )
        dataset = result.scalar_one_or_none()

        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Get files with join to File table
        result = await db.execute(
            select(DatasetFile, FileRecord)
            .join(FileRecord, DatasetFile.file_id == FileRecord.id)
            .where(DatasetFile.dataset_id == dataset_id)
        )
        file_rows = result.all()

        return DatasetDetailResponse(
            id=dataset.id,
            name=dataset.name,
            annotation=dataset.annotation,
            created_at=dataset.created_at,
            updated_at=dataset.updated_at,
            files=[
                {
                    "id": file.id,
                    "file_path": file.file_path,
                }
                for _, file in file_rows
            ]
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting dataset: {str(e)}") from e

@router.put("/{dataset_id}", response_model=DatasetDetailResponse)
async def update_dataset(
    dataset_id: int,
    dataset_update: DatasetUpdate,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update dataset
    """
    try:
        # Get dataset
        result = await db.execute(
            select(Dataset).where(
                Dataset.id == dataset_id,
                or_(Dataset.user_id == user_id, Dataset.shared)
            )
        )
        dataset = result.scalar_one_or_none()

        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Update fields
        if dataset_update.name is not None:
            # Check if new name already exists
            result = await db.execute(
                select(Dataset).where(
                    Dataset.name == dataset_update.name,
                    or_(Dataset.user_id == user_id, Dataset.shared),
                    Dataset.id != dataset_id
                )
            )
            existing = result.scalar_one_or_none()
            if existing:
                raise HTTPException(status_code=400, detail="Dataset name already exists")
            dataset.name = dataset_update.name

        if dataset_update.annotation is not None:
            dataset.annotation = dataset_update.annotation

        if dataset_update.files is not None:
            # Delete existing dataset-file associations
            await db.execute(
                delete(DatasetFile).where(DatasetFile.dataset_id == dataset_id)
            )

            # Add new files
            for filepath in dataset_update.files:
                # Get or create File record
                file_result = await db.execute(
                    select(FileRecord).where(FileRecord.file_path == filepath)
                )
                db_file_record = file_result.scalar_one_or_none()

                if not db_file_record:
                    db_file_record = FileRecord(file_path=filepath)
                    db.add(db_file_record)
                    await db.flush()

                # Create DatasetFile junction record
                dataset_file = DatasetFile(
                    dataset_id=dataset.id,
                    file_id=db_file_record.id,
                )
                db.add(dataset_file)

        await db.commit()
        await db.refresh(dataset)

        # Get updated files with join to File table
        result = await db.execute(
            select(DatasetFile, FileRecord)
            .join(FileRecord, DatasetFile.file_id == FileRecord.id)
            .where(DatasetFile.dataset_id == dataset_id)
        )
        file_rows = result.all()

        return DatasetDetailResponse(
            id=dataset.id,
            name=dataset.name,
            annotation=dataset.annotation,
            created_at=dataset.created_at,
            updated_at=dataset.updated_at,
            files=[
                {
                    "id": file.id,
                    "file_path": file.file_path,
                }
                for _, file in file_rows
            ]
        )

    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Error updating dataset: {str(e)}") from e

@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: int,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete dataset
    """
    try:
        # Get dataset
        result = await db.execute(
            select(Dataset).where(
                Dataset.id == dataset_id,
                or_(Dataset.user_id == user_id, Dataset.shared))
        )
        dataset = result.scalar_one_or_none()

        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Delete dataset (cascades to files)
        await db.delete(dataset)
        await db.commit()

        return {"message": "Dataset deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting dataset: {str(e)}") from e
