from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: SecretStr
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"

    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_name: str = "financial_documents"

    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_top_k: int = 5
    max_upload_size_mb: int = 50

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
