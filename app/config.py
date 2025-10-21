"""Configuration management for RAG-Anything API."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    # API Configuration
    openai_api_key: str
    openai_model: str = "gpt-5-nano"  # Default to gpt-5-nano
    openai_embedding_model: str = "text-embedding-3-small"  # Default embedding model

    # Database Configuration
    database_async_url: str = "postgresql+asyncpg://rag_user:rag_password@postgres:5432/rag_db"
    database_sync_url: str = "postgresql://rag_user:rag_password@postgres:5432/rag_db"

    # Service Configuration
    service_port: int = 8000
    service_host: str = "0.0.0.0"
    log_level: str = "INFO"
    debug: bool = False

    # LLM Configuration
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048

    # RAG Configuration
    rag_top_k: int = 5
    rag_similarity_threshold: float = 0.4
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Vector Store Configuration
    vector_dimension: int = 1536  # OpenAI embeddings dimension


# Global settings instance
settings = Settings()
