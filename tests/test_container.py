"""Unit and topology tests for Docker containerization and Docker Compose orchestration.

Validates multi-stage Dockerfile syntax, non-root user security configuration,
unified single-container Uvicorn entrypoint, .dockerignore build hygiene,
and docker-compose.yml service topology in accordance with ADR-0007 and ARCHITECTURE Sec 6.3 & 7.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE_PATH = ROOT / "Dockerfile"
COMPOSE_PATH = ROOT / "docker-compose.yml"
DOCKERIGNORE_PATH = ROOT / ".dockerignore"


def validate_dockerfile_content(content: str) -> dict[str, Any]:
    """Inspect and validate Dockerfile directives against production hardening invariants.

    Raises ValueError if invariants are violated.
    """
    lines = [line.strip() for line in content.splitlines()]

    # Invariant 1: Multi-stage build structure
    has_builder = any(re.match(r"^FROM\s+\S+\s+AS\s+builder\b", line, re.IGNORECASE) for line in lines)
    has_runner = any(
        re.match(r"^FROM\s+\S+\s+AS\s+(runner|runtime)\b", line, re.IGNORECASE) for line in lines
    )
    if not (has_builder and has_runner):
        raise ValueError("Dockerfile must specify multi-stage build with 'builder' and 'runner/runtime' stages")

    # Invariant 2: Non-root security user
    has_user_directive = any(re.match(r"^USER\s+([a-zA-Z0-9_-]+)", line) for line in lines)
    if not has_user_directive:
        raise ValueError("Dockerfile runtime stage must switch to non-root USER before entrypoint")

    # Invariant 3: Prohibit legacy Streamlit artifacts (ADR-0007)
    for line in lines:
        if "streamlit" in line.lower() and not line.startswith("#"):
            raise ValueError(f"Dockerfile contains forbidden streamlit reference: {line}")
        if "8501" in line and not line.startswith("#"):
            raise ValueError(f"Dockerfile contains forbidden Streamlit port 8501: {line}")

    # Invariant 4: Entrypoint must run unified Uvicorn ASGI server
    has_uvicorn_entrypoint = any(
        "uvicorn" in line and "api.main:app" in line and ("ENTRYPOINT" in line or "CMD" in line)
        for line in lines
    )
    if not has_uvicorn_entrypoint:
        raise ValueError("Dockerfile must define Uvicorn entrypoint targeting api.main:app")

    return {
        "has_builder": has_builder,
        "has_runner": has_runner,
        "has_user_directive": has_user_directive,
        "has_uvicorn_entrypoint": has_uvicorn_entrypoint,
    }


def validate_compose_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Inspect and validate docker-compose configuration dictionary against production invariants.

    Raises ValueError if invariants are violated.
    """
    if not isinstance(data, dict):
        raise ValueError("Compose config root must be a dictionary")

    services = data.get("services")
    if not isinstance(services, dict):
        raise ValueError("Compose config must define a 'services' mapping")

    # Invariant 1: Must define 'api' service
    if "api" not in services:
        raise ValueError("Compose config must define a unified 'api' service")

    # Invariant 2: Strictly no Streamlit service or dual frontend container
    for svc_name in services:
        if "streamlit" in svc_name.lower():
            raise ValueError(f"Forbidden Streamlit service '{svc_name}' detected in docker-compose.yml")

    # Invariant 3: API service port mapping must include 8000 and exclude 8501
    api_ports = services["api"].get("ports", [])
    ports_str = " ".join(str(p) for p in api_ports)
    if "8000" not in ports_str:
        raise ValueError("API service in docker-compose.yml must expose/map port 8000")
    if "8501" in ports_str:
        raise ValueError("Forbidden Streamlit port 8501 mapped in docker-compose.yml")

    # Invariant 4: API service must have healthcheck probe
    api_healthcheck = services["api"].get("healthcheck")
    if not api_healthcheck:
        raise ValueError("API service in docker-compose.yml must define a healthcheck probe")

    return {
        "service_count": len(services),
        "services": list(services.keys()),
        "has_db": "db" in services,
    }


# ==============================================================================
# Dockerfile Invariant Tests
# ==============================================================================


def test_dockerfile_exists_and_multistage():
    """Verify Dockerfile exists at repository root and implements multi-stage build."""
    assert DOCKERFILE_PATH.exists(), f"Dockerfile missing at {DOCKERFILE_PATH}"

    content = DOCKERFILE_PATH.read_text(encoding="utf-8")
    result = validate_dockerfile_content(content)

    assert result["has_builder"] is True
    assert result["has_runner"] is True

    # Base image should be python slim (3.11, 3.12, or 3.13)
    assert re.search(r"FROM\s+python:3\.(11|12|13)-slim\s+AS\s+builder", content)
    assert re.search(r"FROM\s+python:3\.(11|12|13)-slim\s+AS\s+runner", content)


def test_dockerfile_non_root_security_user():
    """Verify Dockerfile provisions and activates a non-root application user."""
    content = DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert "groupadd" in content
    assert "useradd" in content
    assert "USER appuser" in content
    assert "--chown=appuser:appgroup" in content
    assert "chown -R appuser:appgroup /app" in content


def test_dockerfile_entrypoint_and_port():
    """Verify Dockerfile exposes port 8000 and defines single Uvicorn entrypoint."""
    content = DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert "EXPOSE 8000" in content
    assert 'ENTRYPOINT ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]' in content
    assert "HEALTHCHECK" in content
    assert "http://localhost:8000/health" in content


def test_dockerfile_required_directories_copied():
    """Verify that all core runtime directories and artifacts are copied into image."""
    content = DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert "api/" in content
    assert "src/" in content
    assert "models/" in content
    assert "data/" in content
    assert "/opt/venv" in content


def test_dockerfile_no_streamlit_artifacts():
    """Verify Dockerfile strictly eliminates Streamlit per ADR-0007."""
    content = DOCKERFILE_PATH.read_text(encoding="utf-8")

    # Streamlit shouldn't be executed or exposed
    for line in content.splitlines():
        clean_line = line.strip()
        if not clean_line.startswith("#"):
            assert "streamlit" not in clean_line.lower(), f"Unexpected streamlit reference: {clean_line}"
            assert "8501" not in clean_line, f"Unexpected port 8501 reference: {clean_line}"


# ==============================================================================
# Docker Compose Topology Tests
# ==============================================================================


def test_docker_compose_exists_and_valid_yaml():
    """Verify docker-compose.yml exists, parses cleanly, and adheres to single-container topology."""
    assert COMPOSE_PATH.exists(), f"docker-compose.yml missing at {COMPOSE_PATH}"

    with open(COMPOSE_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    result = validate_compose_dict(data)
    assert "api" in result["services"]
    assert result["has_db"] is True


def test_docker_compose_api_service_configuration():
    """Verify API service ports, healthcheck, and environment configuration."""
    with open(COMPOSE_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    api_svc = data["services"]["api"]

    # Port mapping
    ports = api_svc.get("ports", [])
    assert any("8000:8000" in str(p) for p in ports)

    # Healthcheck
    healthcheck = api_svc.get("healthcheck", {})
    test_cmd = str(healthcheck.get("test", ""))
    assert "http://localhost:8000/health" in test_cmd

    # Environment variables
    env = api_svc.get("environment", [])
    env_str = str(env)
    assert "APP_ENV" in env_str
    assert "API_PORT=8000" in env_str
    assert "MODEL_DIR=models" in env_str


def test_docker_compose_cli_config_execution():
    """Execute 'docker compose config' command if docker CLI is available on host."""
    docker_bin = shutil.which("docker")
    if not docker_bin:
        pytest.skip("Docker CLI not available on current host system")

    result = subprocess.run(
        [docker_bin, "compose", "config"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"docker compose config failed:\n{result.stderr}"
    assert "marketing-campaign-api" in result.stdout


# ==============================================================================
# .dockerignore Hygiene Tests
# ==============================================================================


def test_dockerignore_rules():
    """Verify .dockerignore excludes caches, virtualenvs, local env, and development artifacts."""
    assert DOCKERIGNORE_PATH.exists(), f".dockerignore missing at {DOCKERIGNORE_PATH}"

    content = DOCKERIGNORE_PATH.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip() and not line.strip().startswith("#")]

    assert ".git" in lines
    assert "__pycache__" in lines
    assert ".pytest_cache" in lines
    assert ".venv/" in lines or ".venv" in lines
    assert ".env" in lines
    assert "notebooks/" in lines or "notebooks" in lines
    assert "tests/" in lines or "tests" in lines

    # Essential source files must NOT be ignored
    assert "api/" not in lines
    assert "src/" not in lines
    assert "models/" not in lines
    assert "requirements.txt" not in lines


# ==============================================================================
# Invalid-Input & Negative Invariant Tests
# ==============================================================================


def test_invalid_dockerfile_rejections():
    """Verify that Dockerfile validator raises ValueError on non-compliant configurations."""
    # Case 1: Missing multi-stage
    single_stage = """
    FROM python:3.11-slim
    WORKDIR /app
    COPY . .
    USER appuser
    ENTRYPOINT ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
    """
    with pytest.raises(ValueError, match="multi-stage build"):
        validate_dockerfile_content(single_stage)

    # Case 2: Running as root (missing USER)
    no_user = """
    FROM python:3.11-slim AS builder
    RUN pip install -r requirements.txt
    FROM python:3.11-slim AS runner
    WORKDIR /app
    COPY . .
    ENTRYPOINT ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
    """
    with pytest.raises(ValueError, match="non-root USER"):
        validate_dockerfile_content(no_user)

    # Case 3: Streamlit entrypoint
    streamlit_dockerfile = """
    FROM python:3.11-slim AS builder
    RUN pip install -r requirements.txt
    FROM python:3.11-slim AS runner
    WORKDIR /app
    COPY . .
    USER appuser
    ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", "--server.port", "8501"]
    """
    with pytest.raises(ValueError, match="forbidden streamlit"):
        validate_dockerfile_content(streamlit_dockerfile)

    # Case 4: Missing Uvicorn entrypoint
    no_uvicorn = """
    FROM python:3.11-slim AS builder
    RUN pip install -r requirements.txt
    FROM python:3.11-slim AS runner
    WORKDIR /app
    COPY . .
    USER appuser
    CMD ["python", "something_else.py"]
    """
    with pytest.raises(ValueError, match="Uvicorn entrypoint"):
        validate_dockerfile_content(no_uvicorn)


def test_invalid_compose_topology_rejections():
    """Verify that compose topology validator raises ValueError on invalid configurations."""
    # Case 1: Streamlit service present
    bad_compose_streamlit = {
        "services": {
            "api": {"ports": ["8000:8000"], "healthcheck": {"test": "curl"}},
            "streamlit": {"ports": ["8501:8501"]},
        }
    }
    with pytest.raises(ValueError, match="Forbidden Streamlit service"):
        validate_compose_dict(bad_compose_streamlit)

    # Case 2: Forbidden port 8501 mapped
    bad_compose_port = {
        "services": {
            "api": {"ports": ["8000:8000", "8501:8501"], "healthcheck": {"test": "curl"}}
        }
    }
    with pytest.raises(ValueError, match="Forbidden Streamlit port 8501"):
        validate_compose_dict(bad_compose_port)

    # Case 3: Missing api service
    bad_compose_no_api = {
        "services": {
            "web": {"ports": ["8000:8000"]}
        }
    }
    with pytest.raises(ValueError, match="unified 'api' service"):
        validate_compose_dict(bad_compose_no_api)

    # Case 4: Missing healthcheck on API service
    bad_compose_no_health = {
        "services": {
            "api": {"ports": ["8000:8000"]}
        }
    }
    with pytest.raises(ValueError, match="healthcheck probe"):
        validate_compose_dict(bad_compose_no_health)

    # Case 5: Malformed root
    with pytest.raises(ValueError, match="must be a dictionary"):
        validate_compose_dict("not-a-dict")  # type: ignore
