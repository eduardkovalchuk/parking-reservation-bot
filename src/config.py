"""Application configuration using pydantic-settings."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI
    openai_api_key: str
    openai_chat_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # Weaviate
    weaviate_host: str
    weaviate_port: int
    weaviate_grpc_port: int
    weaviate_collection_name: str

    # PostgreSQL
    postgres_host: str
    postgres_port: int
    postgres_db: str
    postgres_user: str
    postgres_password: str

    # RAG
    retrieval_k: int = 4
    chunk_size: int = 450
    chunk_overlap: int = 70


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
