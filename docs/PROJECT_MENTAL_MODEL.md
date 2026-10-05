# Project Mental Model — Marketing Campaign Performance Analysis

## Change Overview
Upgrade the Marketing Campaign Performance Analysis platform from a prototype data-science script and local API demo to an enterprise-grade production ML microservice. The change eliminates model loading I/O bottlenecks in FastAPI via lifespan singletons, introduces strict Pydantic v2 input validation and bulk batch prediction endpoints, containerizes the stack with a production-grade multi-stage Dockerfile and Docker Compose, and implements robust observability, health probes, and CI/CD pipelines.

## Project Context
- **Repository:** `MarketingCampaignAnalysis`
- **Remote Origin:** `https://github.com/jegadeesh17/Marketing-Campaign-Performance-Analysis.git`
- **Base Branch:** `main`
- **Base Commit:** `c0c4220ea1ede5968d7369dd6fda558928f6f897` (`fix(compat): add fallback class for scikit-learn _RemainderColsList unpickling`)
- **Working Branch:** `feature/production-hardening`
- **Codebase Map:** [`docs/CODEBASE_MAP.md`](./CODEBASE_MAP.md)

## Posture
- **Posture:** **Production**
- **Standards:** Strict schema validation, in-memory model lifecycle caching, zero silent exception swallowing, reproducible Docker containerization, comprehensive pytest test coverage, and automated GitHub Actions CI.

## Target Persona
- **Primary Persona:** Growth Marketing Leads, Performance Marketers, and FinOps Analysts evaluating multi-brand paid ad campaigns across Nykaa, Purplle, and Tira.
- **Secondary Persona:** ML / Platform Engineers integrating automated campaign revenue forecasting and profitability classification into ad bidding pipelines or enterprise BI dashboards.

## Agreed Milestones
- **M1: In-Memory Lifespan Serving & API Hardening**
  - Refactor `api/main.py` using FastAPI lifespan context to load XGBoost regression and classification pipelines as in-memory singletons on startup, eliminating per-request disk reads.
  - Implement strict Pydantic v2 input validation guarding against negative spend, zero impressions, invalid months, and unsupported categories.
  - Expose `/forecast_revenue/batch` and `/predict_profitability/batch` endpoints for high-throughput campaign evaluation.
  - Expand pytest unit and API test coverage for boundary values and error handling.
- **M2: Production Containerization, Glassmorphic UI & Cloud Deployment Configuration**
  - Author production multi-stage `Dockerfile` and `docker-compose.yml` orchestrating FastAPI and Streamlit with health checks.
  - Upgrade Streamlit dashboard (`app/app.py`) with a magical glassmorphic UI design (frosted-glass cards with `backdrop-filter: blur()`, glowing gradient borders, modern typography, and refined KPI scorecards).
  - Integrate Pydantic `BaseSettings` for robust environment variable management.
  - Establish automated GitHub Actions CI workflow (`.github/workflows/ci.yml`) enforcing linting, type-checking, pytest verification, and Cloud Run deployment staging.
- **M3: Live Cloud Deployment, Observability & Verification**
  - Deploy service live to GCP Cloud Run (`asia-south1.run.app`) matching existing portfolio deployments.
  - Add structured JSON logging, request tracing, and latency/throughput metrics.
  - Add production health (`/health`) and readiness (`/ready`) probes.
  - Finalize production documentation, live URL smoke-tests, deployment guides, and complete adversarial review.

---

## Part 1: Requirements and UX

### 1. Primary Journey & Demonstration Narrative
- **Purpose:** Showcase an enterprise-grade ML forecasting and analytics system for multi-brand e-commerce marketing (Nykaa, Purplle, Tira), deployed live on GCP Cloud Run.
- **Entry Points:**
  - **Interactive Dashboard (Streamlit):** Stakeholders and interviewers interact with a 3-column forecasting simulator adjusting ad spend, impressions, clicks, channel mix, and target segments to visualize forecasted revenue, profitability status, and historical ROI benchmarks.
  - **Production REST API (FastAPI):**
    - `POST /forecast_revenue` & `POST /predict_profitability`: Low-latency single-campaign evaluation powered by in-memory model singletons.
    - `POST /forecast_revenue/batch` & `POST /predict_profitability/batch`: High-throughput batch inference endpoint accepting a list of campaign records, returning structured predictions.
    - `GET /health` & `GET /ready`: Production readiness probes reporting model loading status, version metadata, and memory readiness.
  - **Runtime Orchestration:** Multi-stage OCI container deployed to GCP Cloud Run, and locally runnable via Docker Compose.

### 2. Errors, Edge Cases & Validation Strategy (Recommended Standard)
- **Strict Boundary Validation (422 Unprocessable Entity):**
  - Numeric invariants: `impressions >= 0`, `clicks >= 0`, `leads >= 0`, `conversions >= 0`, `acquisition_cost >= 0`, `engagement_score >= 0`.
  - Logical invariants: `clicks <= impressions` (clicks cannot exceed impressions) and `conversions <= clicks` (conversions cannot exceed clicks).
  - Date boundaries: `1 <= month <= 12`.
  - Categorical constraints: Brand must be one of `['nykaa', 'purplle', 'tira']` (case-insensitive normalization). Unknown brands reject with an explicit 422 list of valid brands.
- **Leakage & Zero-Division Immunity:**
  - Zero-safe feature derivations in `src/inference.py`: Zero impressions yields `CTR = 0.0`; zero clicks yields `conversion_rate = 0.0` and `CPL = 0.0`. No `NaN`, `Inf`, or unhandled `ZeroDivisionError` bubbles up.
- **Batch Error Handling:**
  - Batch endpoints validate all items; if any item is malformed, return structured 422 detail indicating the exact row index and validation error.

### 3. Explicit Non-Goals
- **No Model Retraining:** The existing trained XGBoost models (`revenue_regressor.joblib` and `profit_classifier.joblib`) are frozen and reused as-is.
- **No External Live Data Connectors:** No real-time webhooks or sync integrations with Google Ads / Meta Ads APIs.
- **No Authentication / Multi-Tenancy:** Single-tenant showcase service without JWT or API key gating.

