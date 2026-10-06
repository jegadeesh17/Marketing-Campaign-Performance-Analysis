# Multi-Brand Marketing Campaign Performance Analysis
---
### **Project Overview**

Marketing campaigns generate massive streams of performance indicators — impressions, clicks, conversions, and spend — across multiple brands and channels. This project builds an end-to-end machine learning and analytics platform to clean multi-brand datasets, engineer advanced features, train predictive models, and deploy a real-time forecasting dashboard.

The system forecasts campaign revenue (XGBoost regression, R² ≈ 0.72 on holdout) and predicts profitability (XGBoost classifier with class-weight balancing). **Interview framing:** cite weighted F1 and per-class recall from `reports/evaluation.md`, not headline accuracy alone.

**Repository:** [github.com/jegadeesh17/Marketing-Campaign-Performance-Analysis](https://github.com/jegadeesh17/Marketing-Campaign-Performance-Analysis)  
**Full specification:** [docs/PROJECT_SPEC.md](docs/PROJECT_SPEC.md)

---

### **Key Features**

* **Multi-Brand Data Ingestion:** Loads raw campaign CSVs from Nykaa, Purplle, and Tira into a central PostgreSQL database.
* **Advanced Feature Engineering:** Cyclical time encoding, CTR/conversion/CPL ratio generation, and multi-label channel parsing.
* **Revenue Regression:** XGBoost Regressor pipeline forecasting exact campaign revenue (R² ≈ 0.72).
* **Profit Classification:** XGBoost Classifier with `scale_pos_weight` for imbalanced profit/loss classes.
* **Interactive Glassmorphic Web App:** Real-time revenue and profit forecasts through a dark-mode glassmorphic interface (`/app`) with glowing accents, KPI cards, and live health status.
* **High-Throughput Batch Predictions:** Vectorized batch endpoints (`/forecast_revenue/batch` & `/predict_profitability/batch`) processing up to 500 records per request with Pydantic v2 invariant validation.
* **Production Observability & Probes:** Operational liveness (`/health`) and readiness (`/ready`) probes, structured JSON logging middleware, and automated smoke verification script.
* **Leakage-Safe Pipeline:** Strict train/test isolation and target leakage prevention throughout preprocessing.
* **Modular ML Architecture:** Separate ingestion, preprocessing, and training scripts for clean pipeline separation.

---

### **Dataset**

* **Source:** Multi-brand marketing campaign records (Nykaa, Purplle, Tira)
* **In repo:** `*_campaign_data_sample.csv` per brand
* **Full data:** Place `nykaa_campaign_data.csv`, `purplle_campaign_data.csv`, `tira_campaign_data.csv` in `data/` — see [data/DATA_SETUP.md](data/DATA_SETUP.md)
* **Coverage:** Multi-channel performance data with impressions, clicks, spend, revenue
* **Format:** Raw CSVs ingested into PostgreSQL

#### **Key Features**

* Impressions, clicks, and conversions
* Campaign spend and revenue
* Channel types (YouTube, Instagram, Email, etc.)
* Date and campaign month
* Brand and campaign identifiers

---

### **Project Structure**

```bash
MarketingCampaignAnalysis/
│
├── api/                          # FastAPI microservice & glassmorphic web app
│   ├── index.html                # Dark-mode glassmorphic single-page web app
│   └── main.py                   # Lifespan singletons, REST routes & logging middleware
├── data/                         # Project datasets
├── docs/                         # Documentation, living specs & EDA artifacts
├── models/                       # Saved trained models
├── notebooks/                    # Jupyter notebooks (Source of Truth)
├── scripts/                      # Operational utilities & smoke verification
│   └── smoke_test.py             # Live and offline --mock verification script
├── src/                          # Core Python logic, inference & configuration
├── tests/                        # Comprehensive test suite (131+ automated tests)
├── Dockerfile                    # Multi-stage OCI container definition
├── docker-compose.yml            # Container orchestration specification
├── requirements.txt              # Production Python dependencies
└── README.md
```

---

### **How It Works**

### **1. Data Ingestion & Preprocessing**

* Loads raw multi-brand CSVs using SQLAlchemy into PostgreSQL
* Handles missing values via mode imputation (categorical) and median imputation (numerical)
* Removes duplicates and standardizes column formats

| Step                | Operation                                    |
| ------------------- | -------------------------------------------- |
| Mode Imputation     | Fills categorical nulls with most-frequent   |
| Median Imputation   | Fills numerical nulls resistant to outliers  |
| Duplicate Removal   | Drops repeated records                       |
| Type Standardization| Enforces consistent dtypes across brands     |

#### **Exploratory Data Analysis (EDA)**

The project includes rich visual analysis to uncover trends in campaign performance:
* **ROI Distributions:** Histograms and KDE plots showing the spread of return on investment.
* **Correlation Heatmaps:** Uncovering relationships between impressions, clicks, conversions, and revenue.
* **ROI by Campaign Type:** Boxplots identifying typical performance and massive "viral" outliers for each channel.
* **Revenue vs. Acquisition Cost:** Scatterplots visualizing the efficiency of ad spend across campaigns.
* **Total Revenue by Brand:** Bar charts summarizing top-line performance.

---

### **2. Advanced Feature Engineering**

The system creates intelligent analytical features to improve model performance:

| Feature              | Purpose                                          |
| -------------------- | ------------------------------------------------ |
| `ctr`                | Click-Through Rate — measures audience engagement|
| `conversion_rate`    | Conversions per click — measures campaign quality|
| `cpl`                | Cost Per Lead — measures spend efficiency        |
| `month_sin/cos`      | Cyclical time encoding — preserves seasonality   |
| `channel_*` flags    | Binary features for YouTube, Instagram, Email    |

---

### **3. Machine Learning Pipelines**

#### Revenue Regression (XGBoost)
```python
from xgboost import XGBRegressor

regressor = XGBRegressor(random_state=42)
regressor.fit(X_train, y_revenue_train)
```

#### Profit Classification (XGBoost + SMOTETomek)
```python
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier

smt = SMOTETomek(random_state=42)
X_res, y_res = smt.fit_resample(X_train, y_profit_train)

classifier = XGBClassifier(random_state=42)
classifier.fit(X_res, y_res)
```

---

### **Model Performance**

See `reports/evaluation.md` (auto-generated after training). Latest holdout results:

| Metric                           | Score   |
| -------------------------------- | ------- |
| Revenue Regression R²            | 0.7189  |
| Revenue Regression RMSE          | 252,360 |
| Profit Classification Accuracy   | 0.9686  |
| Profit Classification Weighted F1| 0.9689  |

Unprofitable class (label 0) recall is lower than profitable class — discuss trade-offs in interviews.

### **REST API**

| Endpoint | Method | Description |
| -------- | ------ | ----------- |
| `/health` | GET | Operational liveness probe with process uptime and model status |
| `/ready` | GET | Kubernetes/Cloud Run readiness probe inspecting memory model state |
| `/app` | GET | Dark-mode glassmorphic single-page web application |
| `/forecast_revenue` | POST | Single campaign revenue forecast (INR) |
| `/predict_profitability` | POST | Single campaign profitability classification |
| `/forecast_revenue/batch` | POST | High-throughput batch revenue forecasting (up to 500 records) |
| `/predict_profitability/batch` | POST | High-throughput batch profitability classification |

---

### **Interactive Application Deployment**

The project features an interactive dark-mode **Glassmorphic Web Application** served natively by FastAPI at `/app`, enabling performance marketers and stakeholders to evaluate campaign parameters, inspect live probe statuses, and receive instant predictions via client-side REST calls.

#### **Run Locally (Uvicorn)**
```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Open [http://127.0.0.1:8000/app](http://127.0.0.1:8000/app) in your browser.

#### **Run with Docker Compose**
```bash
docker compose up --build
```
Access at [http://localhost:8000/app](http://localhost:8000/app).

Models are saved to `models/` after training (not committed). See `reports/evaluation.md` and `docs/DEMO.md`.

---

### **Technology Stack**

| Category             | Tools                                               |
| -------------------- | --------------------------------------------------- |
| Programming          | Python                                              |
| Data Processing      | Pandas, NumPy                                       |
| Database             | PostgreSQL, SQLAlchemy                              |
| Machine Learning     | Scikit-learn, XGBoost                               |
| Imbalanced Learning  | imbalanced-learn (SMOTETomek)                       |
| Web & Microservice   | FastAPI, Uvicorn, Vanilla HTML5/CSS3 (Glassmorphic)  |
| Data Validation      | Pydantic v2, Pydantic-Settings                      |
| Container & CI/CD    | Docker (Multi-stage), Docker Compose, GitHub Actions|
| Visualization        | Plotly                                              |

---

### **Getting Started**

### **1. Clone Repository**

```bash
git clone https://github.com/jegadeesh17/Marketing-Campaign-Performance-Analysis.git

cd MarketingCampaignAnalysis
```

---

### **2. Configure Database**

Ensure PostgreSQL is running with the `marketing_campaign` database. Update `.env` with your credentials:

```env
DB_HOST=localhost
DB_NAME=marketing_campaign
DB_USER=your_user
DB_PASSWORD=your_password
```

---

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

---

### **4. Launch Notebook**

```bash
jupyter notebook "notebooks/Marketing Campaign Performance Analysis.ipynb"
```

### **5. Run Tests & Smoke Verification**

```bash
# Fast automated test suite
python -m pytest -q -m "not slow"

# Full automated test suite
python -m pytest -q

# End-to-end operational smoke verification (in-process mock or live URL)
python scripts/smoke_test.py --mock
```

CSV fallback works without PostgreSQL. For DB ingestion: `python src/data_ingestion.py`

### **6. Launch Web Application**

```bash
uvicorn api.main:app --port 8000
```
Open [http://localhost:8000/app](http://localhost:8000/app) in your browser.

---

### **Example Use Case**

A marketing analytics team can use this platform to:

1. Forecast expected revenue for a proposed campaign before launch
2. Identify campaigns at risk of being unprofitable
3. Optimize channel mix based on CTR and conversion rate trends
4. Allocate budgets across brands based on model-driven profitability predictions

---

### **Future Improvements**

* SHAP-based model explainability dashboard
* Automated campaign performance alerting system

---

### **Contributors**

* **Jegadeesh D** — Data ingestion, feature engineering, XGBoost modeling, imbalanced learning, and Streamlit dashboard development

---

### **License**

MIT License
