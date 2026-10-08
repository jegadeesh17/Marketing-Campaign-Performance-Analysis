# ==============================================================================
# Multi-Stage Production Dockerfile for Marketing Campaign Intelligence Microservice
# Architecture: Unified single-container serving FastAPI and Dark-Mode Glassmorphic SPA (/app)
# Reference: docs/DECISIONS.md (ADR-0007), docs/ARCHITECTURE.md Section 6.3 & 7
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build Dependencies
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system compilation dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Install python dependencies in isolated virtual environment
COPY requirements.txt ./
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------------------
# Stage 2: Minimal Non-Root Runtime
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS runner

WORKDIR /app

# Environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    APP_ENV=production

# Install curl for healthcheck probe
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Create non-root security user and group
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy source code, models, and data artifacts with non-root ownership
COPY --chown=appuser:appgroup api/ api/
COPY --chown=appuser:appgroup src/ src/
COPY --chown=appuser:appgroup models/ models/
COPY --chown=appuser:appgroup data/ data/

# Ensure workspace ownership
RUN chown -R appuser:appgroup /app

# Drop root privileges
USER appuser

# Expose unified application port
EXPOSE 8000

# Periodic healthcheck probe inspecting /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Single Uvicorn ASGI server entrypoint respecting dynamic Cloud Run PORT
ENTRYPOINT ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
