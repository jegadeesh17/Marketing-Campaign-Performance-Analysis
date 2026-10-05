"""Database connection and engine configuration using centralized settings."""

from urllib.parse import quote_plus
from sqlalchemy import create_engine
from src.config import get_settings


def get_db_url(dbname: str | None = None) -> str:
    """Construct PostgreSQL database URL using centralized Settings.

    Args:
        dbname: Optional target database name override. If None, uses settings.db_name.
    """
    settings = get_settings()
    db_user = settings.db_user
    db_password = settings.db_password
    db_password_encoded = f":{quote_plus(db_password)}" if db_password else ""
    db_host = settings.db_host
    db_port = settings.db_port
    effective_dbname = dbname if dbname is not None else settings.db_name

    return f"postgresql://{db_user}{db_password_encoded}@{db_host}:{db_port}/{effective_dbname}"


def get_engine(dbname: str | None = None):
    """Create and return a SQLAlchemy engine instance for the specified database."""
    return create_engine(get_db_url(dbname))
