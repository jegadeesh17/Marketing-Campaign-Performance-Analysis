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
- **M2: Production Containerization & CI/CD Pipeline**
  - Author production multi-stage `Dockerfile` and `docker-compose.yml` orchestrating FastAPI, Streamlit, and PostgreSQL staging.
  - Integrate Pydantic `BaseSettings` for robust environment variable management.
  - Establish automated GitHub Actions CI (`.github/workflows/ci.yml`) enforcing linting, type-checking, and pytest verification.
- **M3: Observability, Metrics & Production Documentation**
  - Add structured JSON logging, request tracing, and latency/throughput metrics.
  - Add production health (`/health`) and readiness (`/ready`) probes.
  - Finalize production documentation, deployment guides, and complete adversarial review.

---

## Part 1: Requirements and UX
*(To be populated during Phase 1 Spec Interview)*

