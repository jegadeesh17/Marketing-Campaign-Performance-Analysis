# Multi-Brand Marketing Campaign Performance Analysis

An end-to-end machine learning and analytics platform for multi-brand marketing campaigns: it cleans raw campaign data from three brands, engineers features, trains a revenue regressor and a profit classifier, and serves forecasts through a FastAPI service with a dark-mode web app.

On holdout data the revenue model reaches R² 0.7189 and the profit classifier a weighted F1 of 0.9689 (see [Model performance](#model-performance)).

## Features

- **Multi-brand ingestion:** loads raw campaign CSVs from Nykaa, Purplle and Tira into PostgreSQL, with a CSV fallback that works without a database.
- **Feature engineering:** cyclical month encoding, CTR, conversion rate and cost-per-lead ratios, and multi-label channel parsing.
- **Revenue regression:** an XGBoost regressor that forecasts campaign revenue.
- **Profit classification:** an XGBoost classifier with `scale_pos_weight` for the imbalanced profit and loss classes.
- **Web app:** forecasts and live health status in a dark-mode single-page app served at `/app`.
- **Batch endpoints:** up to 500 records per request, validated with Pydantic v2 invariants such as `clicks <= impressions`.
- **Operations:** liveness (`/health`) and readiness (`/ready`) probes, structured JSON request logging and a smoke-test script.
- **Leakage-safe pipeline:** strict train/test isolation, with class weights computed from training labels only.

## Quick start

Requires Python 3.11.

```bash
git clone https://github.com/jegadeesh17/Marketing-Campaign-Performance-Analysis.git
cd Marketing-Campaign-Performance-Analysis
python -m venv .venv
source .venv/Scripts/activate       # Windows Git Bash; use .venv/bin/activate elsewhere
pip install -r requirements.txt
uvicorn api.main:app --port 8000
```

Open http://localhost:8000/app. To run in a container instead: `docker compose up --build`.

The two trained models (`revenue_regressor.joblib`, `profit_classifier.joblib`) are tracked with Git LFS (see `.gitattributes`); other model files are git-ignored. The repository contains `*_campaign_data_sample.csv` files for each brand. For the full data, place `nykaa_campaign_data.csv`, `purplle_campaign_data.csv` and `tira_campaign_data.csv` in `data/` ([data/DATA_SETUP.md](data/DATA_SETUP.md)).

## Usage

Train and explore in the notebook, which is the source of truth for the pipeline:

```bash
jupyter notebook "notebooks/Marketing Campaign Performance Analysis.ipynb"
```

To load the data into PostgreSQL (needs the `DB_*` variables below): `python src/data_ingestion.py`.

Call the API:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness probe with process uptime and model status |
| `/ready` | GET | Readiness probe; 200 once the models are loaded, 503 before |
| `/app` | GET | The single-page web app |
| `/forecast_revenue` | POST | Revenue forecast for one campaign (INR) |
| `/predict_profitability` | POST | Profitability classification for one campaign |
| `/forecast_revenue/batch` | POST | Batch revenue forecast, up to 500 records |
| `/predict_profitability/batch` | POST | Batch profitability classification |

## Model performance

Latest holdout results, from `reports/evaluation.md` (regenerated after training):

| Metric | Score |
|---|---|
| Revenue regression R² | 0.7189 |
| Revenue regression RMSE | 252,360 |
| Profit classification accuracy | 0.9686 |
| Profit classification weighted F1 | 0.9689 |

The notebook's holdout classification report shows recall 0.99 for unprofitable campaigns (label 0) and 0.96 for profitable ones (label 1). `reports/evaluation.md` does not store per-class recall, so read both before judging the classifier by accuracy alone.

## Running tests

```bash
python -m pytest -q                  # full suite
python -m pytest -q -m "not slow"    # fast suite (no tests are marked slow yet)
python scripts/smoke_test.py --mock  # end-to-end smoke check, in process
```

## Configuration

Copy `.env.example` to `.env` and edit the values.

| Variable | Purpose |
|---|---|
| `APP_ENV` | Deployment environment: development, staging or production |
| `API_HOST` | Interface the Uvicorn server binds to (default `0.0.0.0`) |
| `API_PORT` | Port for the FastAPI service (default 8000) |
| `PORT` | Port for Cloud Run container deployment (default 8000) |
| `LOG_LEVEL` | Logging level: DEBUG, INFO, WARNING or ERROR |
| `MODEL_DIR` | Directory containing the `.joblib` model files (default `models`) |
| `DB_HOST` | PostgreSQL host for the optional campaign warehouse |
| `DB_PORT` | PostgreSQL port (default 5432) |
| `DB_NAME` | PostgreSQL database name (default `marketing_campaign`) |
| `DB_USER` | PostgreSQL username (default `postgres`) |
| `DB_PASSWORD` | PostgreSQL password |

## Project structure

```
api/        FastAPI service and the single-page web app
data/       sample campaign CSVs and data setup notes
docs/       specifications, architecture, decisions and demo script
models/     trained model files
notebooks/  the analysis and training notebook
reports/    generated evaluation report and metrics
scripts/    exploratory-analysis export, evaluation export and smoke test
src/        configuration, ingestion, preprocessing, training and inference
tests/      pytest suite
```

## Documentation

- [docs/README.md](docs/README.md): index of the documents
- [docs/PROJECT_SPEC.md](docs/PROJECT_SPEC.md): technical specification of the whole project
- [docs/SPEC.md](docs/SPEC.md): specification of the production-hardening work
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): components, API contracts and deployment topology
- [docs/DECISIONS.md](docs/DECISIONS.md): architecture decision records
- [docs/DEMO.md](docs/DEMO.md): five-minute demo script
- [CHANGELOG.md](CHANGELOG.md): history of changes

## How this was built

Designed and built by jegadeesh17 with [SoftwareEngineeringAgents](https://github.com/jegadeesh17/SoftwareEngineeringAgents), a multi-agent workflow for Claude Code. I reviewed and tested all code.

## License

[MIT](LICENSE)
