"""Admin API configuration."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore",
    )

    postgres_host: str
    postgres_port: int
    postgres_db: str
    postgres_user: str
    postgres_password: str

    admin_username: str
    admin_password: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
