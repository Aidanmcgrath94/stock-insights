from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # extra="ignore": .env may hold entries not used by Settings (e.g.
    # LOG_LEVEL, read directly from the environment by logging_config)
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str
    finnhub_api_key: str
    openai_model: str = "gpt-4o-mini"


@lru_cache
def get_settings() -> Settings:
    return Settings()
