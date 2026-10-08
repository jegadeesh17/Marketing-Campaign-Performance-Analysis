# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Reactive Auto-Forecast debouncing (320ms) and dynamic synchronization indicator (`.sync-pill`) in workstation UI (`api/index.html`).
- Scenario Delta Tracking (`#res-revenue-delta`) displaying comparative variance (`+₹... (+...% vs last run)`) on the revenue KPI banner.
- Inline input error states (`.input-error`) and dynamic error highlighting for perimeter invariant violations.
- Deep-linking URL parameter synchronization (`window.history.replaceState`) and executive "Copy Brief" clipboard action.
- Operational Readiness Probe endpoint (`GET /ready`) returning HTTP 200/503 based on in-memory model initialization status (`M3-TASK-01`).
- Structured JSON logging middleware in `api/main.py` recording ISO-8601 timestamps, HTTP method, path, status code, and execution latency without leaking request bodies (`M3-TASK-02`).
- Automated end-to-end smoke verification script (`scripts/smoke_test.py`) with live server testing and offline in-process `--mock` execution, alongside integration test coverage in `tests/test_smoke.py` (`M3-TASK-03`).
- Dark-Mode Glassmorphic Single-Page Web App (`api/index.html`) served natively at `GET /app` with root redirect (`M2-TASK-02`).
- Production multi-stage `Dockerfile` and `docker-compose.yml` orchestrating unified single-container Uvicorn runtime on port `${PORT:-8000}` with non-root security user (`M2-TASK-03`).
- Automated GitHub Actions CI Pipeline (`.github/workflows/ci.yml`) validating syntax, fast test execution, and full test suite (`M2-TASK-04`).
- Centralized environment configuration via Pydantic `BaseSettings` in `src/config.py` with defensive empty-string fallback (`M2-TASK-01`).
- High-Throughput Batch Prediction Endpoints (`/forecast_revenue/batch` and `/predict_profitability/batch`) supporting up to 500 records per request with structured validation reporting (`M1-TASK-04`).
- Vectorized Batch Campaign Feature Construction in Inference Engine (`src/inference.py`), deriving one-hot channel flags, cyclical month encodings, and zero-safe ratios (`M1-TASK-03`).
- Strict Pydantic v2 Invariant Boundary Validation in `CampaignInput` enforcing logical constraints (`clicks <= impressions`, `conversions <= clicks`, `leads <= clicks`, `month in 1..12`, non-negative metrics, brand normalization) with HTTP 422 rejections (`M1-TASK-01`).

### Changed
- Reorganized campaign workstation parameters into a progressive top-to-bottom conversion cascade with live step conversion indicators.
- Promoted multi-channel delivery selectors out of progressive disclosure into a first-class form card.
- Added Indian beauty retail seasonality markers and festive sale indicators to the execution month slider.
- Added a responsive Diagnostics Split View mode to inspect conversion drop-off and break-even sensitivity side-by-side on desktop viewports.
- Enhanced operational liveness probe (`GET /health`) with process uptime tracking and version metadata (`M3-TASK-01`).
- Aligned demo showcase narrative in `docs/DEMO.md` with glassmorphic `/app` architecture, probes, batch inference curl commands, and single-port deployment steps (`M3-TASK-03`).
- Completely decommissioned Streamlit framework and dependencies, replacing it with the FastAPI-served glassmorphic web application (`M2-TASK-01`, `M2-TASK-02`, ADR-0007).
- Refactored model serving in `api/main.py` from synchronous per-request disk reads to in-memory model singletons loaded during FastAPI async lifespan startup (`M1-TASK-02`).

### Fixed
- Corrected live preview Cost Per Lead (CPL) calculation in `api/index.html` from `acquisitionCost / leads` to `(conversions * acquisitionCost) / leads`, aligning preview with domain unit economics.
- Replaced disruptive browser `alert()` popups with an accessible, non-blocking toast notification system and inline sync status indicators.
- Fixed WCAG AA color contrast on funnel step badges (`#1d4ed8` on `#eff6ff`) to meet accessibility standards.
- Resolved clean-runner dependency gaps by pinning `pyyaml`, `sqlalchemy`, and `httpx` in `requirements.txt`.
- Silenced pytest-asyncio deprecation warnings in `pytest.ini` with `asyncio_default_fixture_loop_scope = function`.

### Removed
- Decommissioned legacy Streamlit interface (`app/app.py`), resolving code duplication and removing dead dependencies.
- Excluded unreferenced model variants (`revenue_regressor_tuned.joblib`) from Docker builds via `.dockerignore`.

