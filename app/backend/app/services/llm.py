from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import UserSettings
from app.security import decrypt_value


async def get_user_llm_config(db: AsyncSession, user_id: str) -> dict:
    """Retrieves and decrypts the user's LLM API keys from the database."""
    result = await db.execute(select(UserSettings).where(UserSettings.user_id == user_id))
    settings = result.scalar_one_or_none()

    config = {}
    if settings:
        config['OPENAI_API_KEY'] = decrypt_value(settings.openai_api_key)
        config['ANTHROPIC_API_KEY'] = decrypt_value(settings.anthropic_api_key)
        config['GEMINI_API_KEY'] = decrypt_value(settings.gemini_api_key)
        config['TOGETHER_API_KEY'] = decrypt_value(settings.together_api_key)

    return {k: v for k, v in config.items() if v}
