# config/settings.py
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Absolute path to the project root .env file
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """

    # --- LLM and API Keys ---
    GEMINI_API_KEY: str = Field(...)
    PRIMARY_MODEL: str = Field(default="gemini-2.5-pro")
    SECONDARY_MODEL: str = Field(default="gemini-2.5-flash")
    EMBEDDING_MODEL: str = Field(default="text-embedding-004")

    # --- JSON VALIDITY CHECKER---
    JSON_VALIDATOR: str = Field(default="gemini-flash-lite-latest")

    # --- Pinecone / Vector DB Config ---
    PINECONE_API_KEY: str = Field(...)
    PINECONE_ENVIRONMENT: str = Field(...)
    PINECONE_INDEX_NAME: str = Field(...)
    EMBEDDING_DIMENSION: int = Field(default=768)

    # --- Performance / Cost Control ---
    RAG_TOP_K: int = Field(default=8)

    # --- Observability ---
    # Default is INFO. Can be set to DEBUG, WARNING, ERROR in .env
    LOG_LEVEL: str = Field(default="INFO")

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Initialize settings singleton
settings = Settings()
