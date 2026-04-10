from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import UserSettings
from app.env import IS_LOCAL_ENV
from app.security import decrypt_value, get_merged_secrets


async def get_user_llm_config(db: AsyncSession, user_id: str) -> dict:
    """
    Retrieves the user's LLM API keys.

    For remote environments, org-level default keys (if any) are loaded first,
    then user-specific keys are merged on top so personal overrides always win.
    """
    if IS_LOCAL_ENV:
        result = await db.execute(select(UserSettings).where(UserSettings.user_id == user_id))
        settings = result.scalar_one_or_none()

        config = {}
        if settings:
            config['OPENAI_API_KEY'] = decrypt_value(settings.openai_api_key)
            config['ANTHROPIC_API_KEY'] = decrypt_value(settings.anthropic_api_key)
            config['GEMINI_API_KEY'] = decrypt_value(settings.gemini_api_key)
            config['TOGETHER_API_KEY'] = decrypt_value(settings.together_api_key)

        return {k: v for k, v in config.items() if v}

    else:
        # get the merged set of secrets (org defaults + user overrides, with user taking precedence)
        _, _, merged = await get_merged_secrets(user_id)

        config = {
            'OPENAI_API_KEY': merged.get('openai_api_key'),
            'ANTHROPIC_API_KEY': merged.get('anthropic_api_key'),
            'GEMINI_API_KEY': merged.get('gemini_api_key'),
            'TOGETHER_API_KEY': merged.get('together_api_key')
        }

        # return only the keys that have non-null values
        return {k: v for k, v in config.items() if v}
