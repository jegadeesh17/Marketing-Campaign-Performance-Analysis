# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- High-Throughput Batch Prediction Endpoints (`/forecast_revenue/batch` and `/predict_profitability/batch`) supporting up to 500 records per request with structured validation reporting (`M1-TASK-04`).
- Vectorized Batch Campaign Feature Construction in Inference Engine (`src/inference.py`), deriving one-hot channel flags, cyclical month encodings, and zero-safe ratios (`M1-TASK-03`).
- Strict Pydantic v2 Invariant Boundary Validation in `CampaignInput` enforcing logical constraints (`clicks <= impressions`, `conversions <= clicks`, `leads <= clicks`, `month in 1..12`, non-negative metrics, brand normalization) with HTTP 422 rejections (`M1-TASK-01`).

### Changed
- Refactored model serving in `api/main.py` from synchronous per-request disk reads to in-memory model singletons loaded during FastAPI async lifespan startup (`M1-TASK-02`).

