from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.database import UserSettings, get_db
from app.security import decrypt_value, encrypt_value

router = APIRouter()

# Pydantic model for the incoming JSON request
class ApiKeysRequest(BaseModel):
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    TOGETHER_API_KEY: str = ""

async def upsert_settings(db: AsyncSession, user_id: str, updates: dict):
    result = await db.execute(select(UserSettings).where(UserSettings.user_id == user_id))
    user_settings = result.scalar_one_or_none()
    if not user_settings:
        user_settings = UserSettings(user_id=user_id)
        db.add(user_settings)

    # encrypt values before setting attributes
    for key, value in updates.items():
        if value: 
            encrypted_value = encrypt_value(value)
            setattr(user_settings, key, encrypted_value)

    await db.commit()
    await db.refresh(user_settings)


@router.get("/")
async def get_settings(
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(UserSettings).where(UserSettings.user_id == user_id))
    user_settings = result.scalar_one_or_none()

    if not user_settings:
        return {}

    # decrypt to check length, then mask
    def decrypt_and_mask(encrypted_token):
        if not encrypted_token:
            return ""
        try:
            # we decrypt just to verify we have a value and show last 4 chars
            plain_text = decrypt_value(encrypted_token)
            if not plain_text: 
                return ""
            return f"...{plain_text[-4:]}" if len(plain_text) > 4 else "******"
        except Exception:
            return ""

    return {
        "OPENAI_API_KEY": decrypt_and_mask(user_settings.openai_api_key),
        "ANTHROPIC_API_KEY": decrypt_and_mask(user_settings.anthropic_api_key),
        "GEMINI_API_KEY": decrypt_and_mask(user_settings.gemini_api_key),
        "TOGETHER_API_KEY": decrypt_and_mask(user_settings.together_api_key),
    }


@router.post("/keys")
async def update_keys(
    keys: ApiKeysRequest,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    updates = {
        "openai_api_key": keys.OPENAI_API_KEY,
        "anthropic_api_key": keys.ANTHROPIC_API_KEY,
        "gemini_api_key": keys.GEMINI_API_KEY,
        "together_api_key": keys.TOGETHER_API_KEY
    }

    await upsert_settings(db, user_id, updates)
    return {"status": "success"}

@router.post("/env")
async def upload_env(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    # Read file content
    content = await file.read()
    content_str = content.decode("utf-8")

    updates = {}
    valid_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "TOGETHER_API_KEY"]

    for line in content_str.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key in valid_keys:
                updates[key.lower()] = value

    if not updates:
        raise HTTPException(status_code=400, detail="No valid API keys found in .env file")

    await upsert_settings(db, user_id, updates)
    return {"status": "success", "updated": list(updates.keys())}
