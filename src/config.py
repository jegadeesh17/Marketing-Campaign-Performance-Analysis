"""Centralized application configuration management via Pydantic BaseSettings."""

from functools import lru_cache
from pathlib import Path

from typing import Any

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    """Application settings with environment variable overrides and sensible defaults."""

    app_env: str = Field(
        default="production",
        description="Deployment environment (development, staging, production)",
    )
    api_host: str = Field(
        default="0.0.0.0",
        description="Host binding interface for Uvicorn ASGI server",
    )
    api_port: int = Field(
        default=8000,
        description="Port for the FastAPI microservice",
    )
    port: int = Field(
        default=8000,
        description="Target port for Google Cloud Run container deployment",
    )
    log_level: str = Field(
        default="INFO",
        description="Application logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    model_dir: str = Field(
        default="models",
        description="Directory path containing serialized .joblib model artifacts",
    )
    db_host: str = Field(
        default="localhost",
        description="PostgreSQL hostname for raw campaign metrics store",
    )
    db_port: int = Field(
        default=5432,
        description="PostgreSQL port",
    )
    db_name: str = Field(
        default="marketing_campaign",
        description="PostgreSQL database name",
    )
    db_user: str = Field(
        default="postgres",
        description="PostgreSQL username",
    )
    db_password: str = Field(
        default="",
        description="PostgreSQL password",
    )

    model_config = SettingsConfigDict(
        env_file=(ENV_PATH, ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    @model_validator(mode="before")
    @classmethod
    def empty_strings_to_default(cls, data: Any) -> Any:
        """Normalize empty string or whitespace-only inputs so fields fall back to defaults."""
        if isinstance(data, dict):
            return {
                k: v
                for k, v in data.items()
                if v is not None and not (isinstance(v, str) and v.strip() == "")
            }
        return data


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings singleton instance."""
    return Settings()
