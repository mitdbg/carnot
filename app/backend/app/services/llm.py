from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import UserSettings
from app.env import IS_LOCAL_ENV
from app.security import decrypt_value, get_user_secrets


async def get_user_llm_config(db: AsyncSession, user_id: str) -> dict:
    """
    Retrieves the user's LLM API keys from AWS Secrets Manager.
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
        secrets = await get_user_secrets(user_id)
        config = {
            'OPENAI_API_KEY': secrets.get('openai_api_key'),
            'ANTHROPIC_API_KEY': secrets.get('anthropic_api_key'),
            'GEMINI_API_KEY': secrets.get('gemini_api_key'),
            'TOGETHER_API_KEY': secrets.get('together_api_key')
        }

        # return only the keys that have non-null values
        return {k: v for k, v in config.items() if v}
