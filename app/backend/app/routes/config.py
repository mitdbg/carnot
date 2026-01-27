from fastapi import APIRouter, HTTPException

from app.env import BASE_DIR, DATA_DIR, SHARED_DATA_DIR
from app.models.schemas import AppConfig

router = APIRouter()


@router.get("/", response_model=AppConfig)
async def get_config(path: str | None = None):
    """
    Return the config for the application.
    """
    try:
        return AppConfig(base_dir=BASE_DIR, data_dir=DATA_DIR, shared_data_dir=SHARED_DATA_DIR)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching config: {str(e)}") from e
