# Codebase Map for Marketing Campaign Performance Analysis
Living map of repository architecture, conventions, test baseline, and production hardening interfaces.

*Date: 2026-10-05*

---

## 1. Summary

Marketing Campaign Performance Analysis is an end-to-end machine learning analytics and forecasting platform for multi-brand campaign performance (Nykaa, Purplle, Tira). The system ingests raw campaign metrics into PostgreSQL (with fallback to local CSV samples), performs leakage-safe data cleaning and feature engineering (CTR, conversion rate, CPL, cyclical month encoding, multi-channel flags), and trains two XGBoost pipelines: an XGBoost Regressor forecasting campaign revenue and an XGBoost Classifier predicting campaign profitability. The models are served through a FastAPI REST API ([`api/main.py`](../api/main.py)) and an interactive 3-column Streamlit dashboard ([`app/app.py`](../app/app.py)).

---

## 2. Stack

- **Language:** Python 3.13.2
- **Frameworks & Web Services:**
  - FastAPI 0.115.6 (REST API service)
  - Uvicorn 0.32.1 (ASGI server)
  - Streamlit (interactive web dashboard)
  - Pydantic v2 (request/response schemas via FastAPI)
- **Machine Learning & Data Processing:**
  - Scikit-learn (pipelines, ColumnTransformer, preprocessing transformers, metrics)
  - XGBoost (XGBRegressor, XGBClassifier)
  - Imbalanced-learn (imbalance handling)
  - Pandas, NumPy (data manipulation and vectorized transforms)
  - Joblib (model serialization and deserialization)
  - Matplotlib, Seaborn (exploratory visual analysis)
- **Database & Storage:**
  - PostgreSQL (relational store for raw campaign metrics)
  - SQLAlchemy (SQL connection and ORM abstraction)
- **Testing:**
  - Pytest 8.3.4 (unit and API test suite)
- **Package Manager & Dependency Manifest:**
  - `pip` via [`requirements.txt`](../requirements.txt)

---

## 3. Layout

```text
MarketingCampaignAnalysis/
├── .env.example                               # Environment template for DB credentials
├── .gitignore                                 # Git ignore rules for caches, env, and full datasets
├── README.md                                  # Project overview, quickstart, and architecture summary
├── pytest.ini                                 # Pytest testpaths and pythonpath configuration
├── requirements.txt                           # Pinned and unpinned project dependencies
├── api/                                       # FastAPI serving application
│   ├── __init__.py                            # API package marker
│   └── main.py                                # FastAPI endpoints: /health, /forecast_revenue, /predict_profitability
├── app/                                       # Streamlit presentation dashboard
│   └── app.py                                 # 3-column interactive forecasting web interface
├── data/                                      # Data storage, SQL schema, and dataset documentation
│   ├── DATA_SETUP.md                          # Guide for sample vs. full CSV dataset resolution
│   ├── queries.sql                            # SQL DDL for raw_campaign_data table
│   ├── nykaa_campaign_data_sample.csv         # Sample dataset for Nykaa brand
│   ├── nykaa_campaign_data_with_nulls.csv     # Imputation test dataset for Nykaa
│   ├── purplle_campaign_data_sample.csv       # Sample dataset for Purplle brand
│   ├── purplle_campaign_data_with_nulls.csv   # Imputation test dataset for Purplle
│   ├── tira_campaign_data_sample.csv          # Sample dataset for Tira brand
│   └── tira_campaign_data_with_nulls.csv      # Imputation test dataset for Tira
├── docs/                                      # Specifications, walkthroughs, and visuals
│   ├── DEMO.md                                # 5-minute interview demonstration script
│   ├── PROJECT_SPEC.md                        # Formal engineering and data specification
│   ├── CODEBASE_MAP.md                        # Living codebase architecture and conventions map
│   └── eda/                                   # Visual analysis plots
│       ├── correlation_heatmap.png            # Heatmap of numerical feature correlations
│       ├── roi_by_campaign_type.png           # Boxplot of ROI across campaign types
│       └── roi_distribution.png               # Histogram & KDE of campaign ROI distribution
├── models/                                    # Serialized machine learning pipelines (tracked in Git/LFS)
│   ├── profit_classifier.joblib               # Trained XGBoost classification pipeline
│   ├── revenue_regressor.joblib               # Trained XGBoost regression pipeline
│   └── revenue_regressor_tuned.joblib         # Tuned regression variant
├── notebooks/                                 # Exploratory and prototyping notebooks
│   └── Marketing Campaign Performance Analysis.ipynb # Prototyping and EDA notebook
├── reports/                                   # Model evaluation metrics and artifacts
│   ├── evaluation.md                          # Holdout evaluation report for regression and classification
│   └── metrics.json                           # Programmatic metric summary (R², RMSE, Accuracy, F1)
├── scripts/                                   # Automation and reporting utility scripts
│   ├── export_evaluation.py                   # Script generating reports/evaluation.md and metrics.json
│   └── generate_eda.py                        # Script generating docs/eda/*.png visual artifacts
├── src/                                       # Core pipeline and feature engineering source modules
│   ├── __init__.py                            # Source package marker
│   ├── data_ingestion.py                      # Multi-brand CSV to PostgreSQL ingestion pipeline
│   ├── data_preprocessing.py                  # Cleaning, imputation, and cyclical feature engineering
│   ├── db_config.py                           # SQLAlchemy database engine configuration
│   ├── inference.py                           # Feature construction utilities for single-row inference
│   └── train_models.py                        # Model training, grid search, and pipeline serialization
└── tests/                                     # Automated test suite
    ├── test_api.py                            # FastAPI route status and prediction integration tests
    └── test_inference.py                      # Feature derivation and shape unit tests
```

### Key Entry Points
- **FastAPI API server:** `uvicorn api.main:app --port 8000`
- **Streamlit dashboard:** `streamlit run app/app.py`
- **Model training pipeline:** `python src/train_models.py`
- **Data ingestion:** `python src/data_ingestion.py`
- **Evaluation report export:** `python scripts/export_evaluation.py`
- **EDA plots generation:** `python scripts/generate_eda.py`
- **Automated test suite:** `pytest`

---

## 4. Conventions

### 4.1 Naming Conventions
- **Code files and modules:** Strict `snake_case.py` (e.g., [`src/data_preprocessing.py`](../src/data_preprocessing.py), [`src/inference.py`](../src/inference.py), [`tests/test_api.py`](../tests/test_api.py)).
- **Classes and Pydantic Models:** `PascalCase` (e.g., `CampaignInput` in [`api/main.py`](../api/main.py#L21), `_RemainderColsList` in [`app/app.py`](../app/app.py#L9)).
- **Functions and Variables:** `snake_case` (e.g., `build_campaign_row()` in [`src/inference.py`](../src/inference.py#L13), `load_and_clean_data()` in [`src/data_preprocessing.py`](../src/data_preprocessing.py#L39), `train_pipelines()` in [`src/train_models.py`](../src/train_models.py#L16)).
- **Constants:** `UPPER_SNAKE_CASE` (e.g., `CHANNELS` in [`src/inference.py`](../src/inference.py#L10), `DATA_DIR` in [`src/data_preprocessing.py`](../src/data_preprocessing.py#L6), `ROOT` in [`api/main.py`](../api/main.py#L12)).

### 4.2 Formatting & Lint Configuration
- **Indentation:** 4 spaces per indentation level.
- **Quotes:** Double quotes (`"`) preferred for strings and docstrings; single quotes (`'`) used in legacy script dictionary lookups.
- **Line Length:** Kept under 120 characters where possible.
- **Lint/Format configs:** No standalone `.flake8`, `ruff.toml`, or `.black` configuration file exists; code adheres to standard PEP 8 conventions.

### 4.3 Typing & Annotations
- Modern Python 3.10+ typing syntax using `from __future__ import annotations` (e.g., [`api/main.py`](../api/main.py#L3), [`src/inference.py`](../src/inference.py#L3), [`scripts/export_evaluation.py`](../scripts/export_evaluation.py#L3)).
- Standard built-in collection types: `list[str]`, `dict`, `tuple[pd.DataFrame, pd.DataFrame]`, primitive return types `-> dict`, `-> None`.
- Pydantic field validation using `Field(...)` constraints (e.g., `month: int = Field(default=5, ge=1, le=12)` in [`api/main.py`](../api/main.py#L27)).

### 4.4 Error Handling & Fallbacks
- **FastAPI HTTP status mapping:** When models cannot be found or loaded, endpoints raise `HTTPException(status_code=503, detail=str(exc)) from exc` (e.g., [`api/main.py`](../api/main.py#L58)).
- **Resilient Data Fallback:** In [`src/data_preprocessing.py`](../src/data_preprocessing.py#L24-L36), `_load_raw_dataframe()` attempts to query PostgreSQL (`raw_campaign_data`); if the DB connection fails, it catches the exception and automatically loads raw CSV files from `data/`.
- **Zero-Division Protection:** Explicit ternary guards are implemented across feature calculations (e.g., `ctr = clicks / impressions if impressions > 0 else 0` in [`src/inference.py`](../src/inference.py#L22), [`src/data_preprocessing.py`](../src/data_preprocessing.py#L73)).
- **Imputation Guards:** Categorical null values default to `.mode()[0]` or `'Unknown'` if empty ([`src/data_preprocessing.py`](../src/data_preprocessing.py#L48)); numerical nulls impute median values.

### 4.5 Module Organization & Path Resolution
- Pipeline components are decoupled into distinct operational scripts under `src/`.
- Scripts and entry points dynamically append the repository root to `sys.path` to ensure importability regardless of current working directory:
  ```python
  # Example from api/main.py and tests/test_api.py
  ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
  if ROOT not in sys.path:
      sys.path.insert(0, ROOT)
  ```

---

## 5. Tests

- **Test Runner:** `pytest` (version 8.3.4)
- **Configuration:** [`pytest.ini`](../pytest.ini):
  ```ini
  [pytest]
  testpaths = tests
  pythonpath = .
  ```
- **Test Directory:** `tests/`
- **Naming Conventions:**
  - Files: `test_*.py`
  - Functions: `test_<function_or_scenario>` (e.g., `test_health`, `test_forecast_revenue`, `test_build_campaign_row_shapes`)
  - Fixtures: Defined in test modules (e.g., `client` fixture with mocked models in [`tests/test_api.py`](../tests/test_api.py#L13))
- **Execution Commands:**
  - Run entire test suite: `pytest` or `pytest -v`
  - Run a single test file: `pytest tests/test_api.py` or `pytest tests/test_inference.py`
  - Run a single test method: `pytest tests/test_api.py::test_health`
- **Markers:** No `slow` or custom pytest markers exist in `pytest.ini` or test implementations.

---

## 6. Baseline

- **Executed Command:** `pytest -v`
- **Execution Date:** 2026-10-05
- **Exit Code:** `0`
- **Summary Line:** `10 passed in 5.35s`
- **Failing Tests:** None (0 failing tests)
- **Test Breakdown:**
  - `tests/test_api.py::test_health` — PASSED
  - `tests/test_api.py::test_forecast_revenue` — PASSED
  - `tests/test_api.py::test_predict_profitability` — PASSED
  - `tests/test_inference.py::test_build_campaign_row_shapes` — PASSED
  - `tests/test_inference.py::test_channel_flags_present` — PASSED
  - `tests/test_inference.py::test_derived_metrics_computed` — PASSED
  - `tests/test_inference.py::test_month_cyclical_encoding` — PASSED
  - `tests/test_inference.py::test_zero_safe_divisions` — PASSED
  - `tests/test_inference.py::test_categorical_defaults` — PASSED
  - `tests/test_inference.py::test_classifier_row_is_copy` — PASSED

---

## 7. Existing Behavior (Production Hardening Target Area)

The planned production hardening change focuses on FastAPI serving, in-memory lifespan model loading, batch predictions, Dockerization, and CI/CD. The existing behaviors and interfaces that must remain stable are:

| Interface / Public Behavior | Implementing Module | Existing Test Coverage |
|-----------------------------|---------------------|------------------------|
| `GET /health` returns `status: "ok"`, `revenue_model: bool`, `profit_model: bool` | [`api/main.py`](../api/main.py#L43-L49) | [`tests/test_api.py::test_health`](../tests/test_api.py#L30-L31) |
| `POST /forecast_revenue` accepts `CampaignInput` payload, returns `{"forecasted_revenue": <float>}` | [`api/main.py`](../api/main.py#L52-L60) | [`tests/test_api.py::test_forecast_revenue`](../tests/test_api.py#L34-L38) |
| `POST /predict_profitability` accepts `CampaignInput` payload, passes predicted revenue to classifier, returns `forecasted_revenue`, `profitable` (bool), `status` ("PROFITABLE" or "LOSS") | [`api/main.py`](../api/main.py#L63-L78) | [`tests/test_api.py::test_predict_profitability`](../tests/test_api.py#L40-L45) |
| `CampaignInput` schema defaults and validation (brand, impressions, clicks, leads, conversions, channels, month in 1..12) | [`api/main.py`](../api/main.py#L21-L35) | Implicitly via `test_forecast_revenue`, `test_predict_profitability` |
| `build_campaign_row(payload: dict)` returns `(df_reg, df_cls)` with derived metrics (`ctr`, `conversion_rate`, `cpl`), `month_sin`, `month_cos`, binary `channel_*` flags | [`src/inference.py`](../src/inference.py#L13-L52) | [`tests/test_inference.py`](../tests/test_inference.py) (7 dedicated unit tests) |
| Streamlit dashboard model loading and inference (`models/revenue_regressor.joblib` and `models/profit_classifier.joblib`) | [`app/app.py`](../app/app.py#L52-L60) | Untested by pytest (manual verification) |

---

## 8. Setup

### 8.1 Environment Variables
Configured via `.env` file (see template in [`.env.example`](../.env.example)):
- `DB_HOST`: Hostname for PostgreSQL instance (default: `localhost`)
- `DB_PORT`: Port for PostgreSQL instance (default: `5432`)
- `DB_NAME`: Database name (default: `marketing_campaign`)
- `DB_USER`: Database username (default: `postgres`)
- `DB_PASSWORD`: Password for database user

### 8.2 External Services & Local Fallbacks
- **PostgreSQL:** Optional for model inference and evaluation fallback, required only for running `python src/data_ingestion.py`.
- **Sample CSVs:** Located in `data/*_campaign_data_sample.csv`, allowing end-to-end retraining, testing, and EDA generation without external database infrastructure.
- **Model Artifacts:** Pre-trained artifacts `models/revenue_regressor.joblib` and `models/profit_classifier.joblib` reside in the repository.

---

## 9. Risks

1. **Per-Request Model Disk I/O:**
   In [`api/main.py`](../api/main.py#L56, #L67-L68), `_load_model()` is invoked on every single HTTP request. This causes unnecessary disk I/O, latency spikes, and potential memory leaks under concurrency. Migrating to FastAPI lifespan management is required.
2. **Coupled Test Fixture Patching:**
   [`tests/test_api.py`](../tests/test_api.py#L24) explicitly patches `api.main._load_model`. If `_load_model` is altered, removed, or bypassed during the lifespan refactor, the test fixture will fail or fail to mock the models. The test fixture must be updated alongside the lifespan implementation (e.g., populating `app.state.models` or managing lifespan in `TestClient`).
3. **Single-Item Payload Limitation:**
   [`src/inference.py`](../src/inference.py#L13) assumes a single dictionary payload and constructs 1-row DataFrames. Adding batch prediction endpoints requires supporting either a list of payloads or a vectorized dataframe builder without breaking the single-record signature expected by `app/app.py` and existing tests.
4. **Scikit-learn Deserialization Compatibility:**
   Serialized pipelines contain custom transformers and `scikit-learn` internal classes (e.g., `_RemainderColsList` addressed in [`app/app.py`](../app/app.py#L8-L11)). Docker images must pin compatible scikit-learn/joblib versions to avoid unpickling errors.
5. **Missing Containerization & CI/CD Infrastructure:**
   No `Dockerfile`, `docker-compose.yml`, `.dockerignore`, or `.github/workflows/` directory currently exists in the repository. Adding these must preserve local execution patterns while standardizing containerized deployment.

---

## 10. Documentation

- **Specification:** [`docs/PROJECT_SPEC.md`](PROJECT_SPEC.md) (comprehensive system specification and functional requirements)
- **Walkthrough & Demo:** [`docs/DEMO.md`](DEMO.md) (5-minute interview demo script)
- **Data Setup:** [`data/DATA_SETUP.md`](../data/DATA_SETUP.md) (dataset resolution order and setup instructions)
- **Evaluation Report:** [`reports/evaluation.md`](../reports/evaluation.md) (regression and classification holdout metrics)
- **Metrics JSON:** [`reports/metrics.json`](../reports/metrics.json) (machine-readable metrics)
- **Project Readme:** [`README.md`](../README.md) (repository documentation)
- **CHANGELOG:** None
- **Architecture Decision Records (ADRs):** None (no `docs/adr/` or `docs/decisions/` directory)
