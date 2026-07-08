# Marketing Campaign Analysis — Technical Specification

---

## Document Control

| Field | Value |
|-------|-------|
| **Document** | PROJECT_SPEC.md |
| **Version** | 1.0 |
| **Status** | Active |
| **Last updated** | 2026-07-08 |
| **Repository** | [github.com/jegadeesh17/Marketing-Campaign-Performance-Analysis](https://github.com/jegadeesh17/Marketing-Campaign-Performance-Analysis) |
| **Related docs** | [README.md](../README.md), [DEMO.md](./DEMO.md), [reports/evaluation.md](../reports/evaluation.md) |

---

## 1. Executive Summary

Marketing Campaign Analysis is an **end-to-end ML analytics platform** for multi-brand campaign intelligence (Nykaa, Purplle, Tira). It forecasts **campaign revenue** (XGBoost regression) and predicts **profitability** (XGBoost classification with imbalance handling), served through PostgreSQL storage, FastAPI endpoints, and a Streamlit forecasting dashboard.

**Interview pitch:**

> *"I forecast campaign revenue with XGBoost at R² 0.72 and profitability with weighted F1 0.97 across Nykaa, Purplle, and Tira — with leakage-safe feature engineering, FastAPI endpoints, and pytest coverage."*

---

## 2. Scope

### 2.1 In Scope

| # | Capability |
|---|------------|
| 1 | Multi-brand CSV ingestion to PostgreSQL |
| 2 | Missing value imputation and duplicate removal |
| 3 | Advanced feature engineering (CTR, CPL, cyclical time, channel flags) |
| 4 | Revenue regression (XGBoost) |
| 5 | Profit classification (XGBoost + SMOTETomek) |
| 6 | Leakage-safe train/test isolation |
| 7 | FastAPI `/forecast_revenue`, `/predict_profitability` |
| 8 | Streamlit 3-column forecasting UI |
| 9 | pytest API + inference tests |

### 2.2 Out of Scope

- Real-time ad platform API integration
- Automated budget optimization / reinforcement learning
- Multi-tenant SaaS deployment

---

## 3. Requirements

### 3.1 Functional Requirements

| ID | Requirement | Module | Status |
|----|-------------|--------|--------|
| FR-01 | Ingest brand CSVs | `src/data_ingestion.py` | ✅ |
| FR-02 | Clean and impute data | `src/preprocessing.py` | ✅ |
| FR-03 | Engineer features + targets | `src/feature_engineering.py` | ✅ |
| FR-04 | Train regression + classifier | `src/train_models.py` | ✅ |
| FR-05 | Export evaluation metrics | `scripts/export_evaluation.py` | ✅ |
| FR-06 | Inference API | `api/main.py` | ✅ |
| FR-07 | Streamlit dashboard | `app/app.py` | ✅ |

---

## 4. Architecture

```text
Nykaa / Purplle / Tira CSVs
        │
        ▼
PostgreSQL ◀── data_ingestion.py
        │
        ▼
preprocessing.py + feature_engineering.py
        │
        ├── XGBRegressor → revenue forecast
        └── XGBClassifier (+ SMOTETomek) → profit_flag
        │
        ▼
models/*.pkl ──▶ api/main.py + app/app.py
```

---

## 5. Data Specification

| Brand | Source | Key columns |
|-------|--------|-------------|
| Nykaa | `data/` CSV | impressions, clicks, spend, revenue, channel_used |
| Purplle | `data/` CSV | same schema family |
| Tira | `data/` CSV | same schema family |

**Derived features:** `ctr`, `conversion_rate`, `cpl`, `month_sin/cos`, binary `channel_*` flags.

**Targets:** `revenue` (regression), `profit_flag` (classification from ROI logic).

---

## 6. Models & Metrics

| Task | Model | Metric | Value |
|------|-------|--------|-------|
| Revenue forecast | XGBoost Regressor | R² | 0.7189 |
| Revenue forecast | XGBoost Regressor | RMSE | 252,360 |
| Profitability | XGBoost Classifier | Weighted F1 | 0.9689 |
| Profitability | XGBoost Classifier | Accuracy | 0.9686 |

**Interview note:** Cite weighted F1 and per-class recall — unprofitable class recall is lower.

Regenerate: `python src/train_models.py && python scripts/export_evaluation.py`

---

## 7. API Specification

### `GET /health`

Model artifact presence and service status.

### `POST /forecast_revenue`

**Input:** `CampaignInput` — impressions, clicks, spend, channel flags, date features.  
**Output:** Predicted revenue.

### `POST /predict_profitability`

**Input:** `CampaignInput` including forecasted revenue context.  
**Output:** Profit/loss prediction + probability.

---

## 8. Leakage Prevention

- Classification feature set excludes direct financial outcome columns used to define target
- Strict train/test split before SMOTETomek resampling (training only)
- Profit pipeline uses revenue forecast as feature only after regression stage (document in interviews)

---

## 9. Deployment

```powershell
pip install -r requirements.txt
python src/train_models.py
pytest -q
uvicorn api.main:app --port 8000
streamlit run app/app.py
```

CSV fallback path works without PostgreSQL for inference demos.

---

## 10. Testing

- `tests/test_api.py` — endpoint health and response schema
- `tests/test_inference.py` — model load and prediction shapes

---

## 11. Module Index

| Path | Purpose |
|------|---------|
| `src/data_ingestion.py` | Brand CSV → PostgreSQL |
| `src/preprocessing.py` | Cleaning and imputation |
| `src/train_models.py` | Train + serialize models |
| `api/main.py` | FastAPI service |
| `notebooks/Marketing Campaign Performance Analysis.ipynb` | EDA + training source of truth |

---

## 12. Future Improvements

- SHAP explainability dashboard
- Automated campaign alerting on predicted loss
- Cross-brand transfer learning experiments
