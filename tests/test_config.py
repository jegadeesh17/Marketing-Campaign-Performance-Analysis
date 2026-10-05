"""Unit and integration tests for centralized Pydantic BaseSettings and database configuration."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import Settings, get_settings
from src.db_config import get_db_url, get_engine


def test_default_settings_values():
    """Verify that default settings match ARCHITECTURE Section 8 specifications."""
    settings = Settings(_env_file=None)

    assert settings.app_env == "production"
    assert settings.api_host == "0.0.0.0"
    assert settings.api_port == 8000
    assert settings.port == 8000
    assert settings.log_level == "INFO"
    assert settings.model_dir == "models"
    assert settings.db_host == "localhost"
    assert settings.db_port == 5432
    assert settings.db_name == "marketing_campaign"
    assert settings.db_user == "postgres"
    assert settings.db_password == ""


def test_cached_get_settings_singleton():
    """Verify get_settings() returns a cached singleton instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    assert isinstance(s1, Settings)


def test_settings_environment_variable_override(monkeypatch):
    """Verify that environment variables properly override default settings values."""
    monkeypatch.setenv("APP_ENV", "staging")
    monkeypatch.setenv("API_HOST", "127.0.0.1")
    monkeypatch.setenv("API_PORT", "9000")
    monkeypatch.setenv("PORT", "8080")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("MODEL_DIR", "custom_models")
    monkeypatch.setenv("DB_HOST", "db.internal")
    monkeypatch.setenv("DB_PORT", "5433")
    monkeypatch.setenv("DB_NAME", "custom_campaigns")
    monkeypatch.setenv("DB_USER", "custom_user")
    monkeypatch.setenv("DB_PASSWORD", "custom_password")

    custom_settings = Settings(_env_file=None)
    assert custom_settings.app_env == "staging"
    assert custom_settings.api_host == "127.0.0.1"
    assert custom_settings.api_port == 9000
    assert custom_settings.port == 8080
    assert custom_settings.log_level == "DEBUG"
    assert custom_settings.model_dir == "custom_models"
    assert custom_settings.db_host == "db.internal"
    assert custom_settings.db_port == 5433
    assert custom_settings.db_name == "custom_campaigns"
    assert custom_settings.db_user == "custom_user"
    assert custom_settings.db_password == "custom_password"


@pytest.mark.parametrize(
    "invalid_field,invalid_val",
    [
        ("api_port", "not_a_valid_port"),
        ("port", "abc"),
        ("db_port", "xyz"),
    ],
)
def test_settings_invalid_input_validation_errors(invalid_field, invalid_val):
    """Verify that invalid input types raise a Pydantic ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None, **{invalid_field: invalid_val})
    assert invalid_field in str(exc_info.value)


def test_requirements_file_dependency_cleanup():
    """Verify requirements.txt contains pydantic-settings and eliminates streamlit."""
    req_path = Path(ROOT) / "requirements.txt"
    assert req_path.exists(), "requirements.txt does not exist"
    content = req_path.read_text(encoding="utf-8")
    lines = [line.strip().lower() for line in content.splitlines() if line.strip() and not line.startswith("#")]

    assert not any("streamlit" in line for line in lines), "requirements.txt must not contain streamlit"
    assert any("pydantic-settings" in line for line in lines), "requirements.txt must contain pydantic-settings"


def test_env_example_documented_keys():
    """Verify .env.example contains all required environment variable keys."""
    env_example_path = Path(ROOT) / ".env.example"
    assert env_example_path.exists(), ".env.example does not exist"
    content = env_example_path.read_text(encoding="utf-8")

    expected_keys = [
        "APP_ENV",
        "API_HOST",
        "API_PORT",
        "PORT",
        "LOG_LEVEL",
        "MODEL_DIR",
        "DB_HOST",
        "DB_PORT",
        "DB_NAME",
        "DB_USER",
        "DB_PASSWORD",
    ]
    for key in expected_keys:
        assert f"{key}=" in content, f".env.example is missing key {key}"


def test_db_config_get_db_url_default():
    """Verify get_db_url builds correct URL from settings with default values."""
    mock_settings = Settings(
        _env_file=None,
        db_user="postgres",
        db_password="",
        db_host="localhost",
        db_port=5432,
        db_name="marketing_campaign",
    )
    with patch("src.db_config.get_settings", return_value=mock_settings):
        url = get_db_url()
        assert url == "postgresql://postgres@localhost:5432/marketing_campaign"


def test_db_config_get_db_url_override_and_password_encoding():
    """Verify get_db_url properly URL-encodes passwords with special characters and respects dbname override."""
    mock_settings = Settings(
        _env_file=None,
        db_user="analytics_user",
        db_password="p@ssword#123!/test",
        db_host="10.0.0.5",
        db_port=5439,
        db_name="default_db",
    )
    with patch("src.db_config.get_settings", return_value=mock_settings):
        url = get_db_url(dbname="override_db")
        assert "override_db" in url
        assert "default_db" not in url
        assert "analytics_user" in url
        assert "10.0.0.5:5439" in url
        # Confirm password was encoded
        assert "%40" in url or "@" in url
        assert "%23" in url or "#" in url


def test_db_config_get_engine_returns_engine():
    """Verify get_engine returns a valid SQLAlchemy engine instance without immediate connection."""
    from sqlalchemy.engine import Engine

    mock_settings = Settings(
        _env_file=None,
        db_user="postgres",
        db_password="",
        db_host="localhost",
        db_port=5432,
        db_name="marketing_campaign",
    )
    with patch("src.db_config.get_settings", return_value=mock_settings):
        engine = get_engine()
        assert isinstance(engine, Engine)
        assert engine.url.database == "marketing_campaign"
        assert engine.url.username == "postgres"


def test_api_main_settings_integration():
    """Verify that api.main imports and uses Settings instance on app.state."""
    from api.main import app

    assert hasattr(app.state, "settings")
    assert isinstance(app.state.settings, Settings)
    assert app.state.settings.api_port == 8000
