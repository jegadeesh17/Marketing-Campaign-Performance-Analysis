# Multi-Brand Marketing Campaign Performance Analysis

---

### **Project Overview**

Marketing campaigns generate massive streams of performance indicators — impressions, clicks, conversions, and spend — across multiple brands and channels. This project builds an end-to-end machine learning and analytics platform to clean multi-brand datasets, engineer advanced features, train predictive models, and deploy a real-time forecasting dashboard.

The system achieves a **96.98% classification accuracy** for predicting campaign profitability and forecasts campaign revenue using XGBoost regression, enabling marketing managers to optimize budget allocation before capital is deployed.

---

### **Key Features**

* **Multi-Brand Data Ingestion:** Loads raw campaign CSVs from Nykaa, Purplle, and Tira into a central PostgreSQL database.
* **Advanced Feature Engineering:** Cyclical time encoding, CTR/conversion/CPL ratio generation, and multi-label channel parsing.
* **Revenue Regression:** XGBoost Regressor pipeline forecasting exact campaign revenue (R² ≈ 0.72).
* **Profit Classification:** XGBoost Classifier with SMOTETomek oversampling achieving >96% accuracy.
* **Imbalanced Class Handling:** SMOTETomek balancing for perfectly resolving unprofitable campaign detection.
* **Interactive Streamlit Dashboard:** Real-time revenue and profit forecasts through a 3-column floating island UI.
* **Leakage-Safe Pipeline:** Strict train/test isolation and target leakage prevention throughout preprocessing.
* **Modular ML Architecture:** Separate ingestion, preprocessing, and training scripts for clean pipeline separation.

---

### **Dataset**

* **Source:** Multi-brand marketing campaign records (Nykaa, Purplle, Tira)
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
├── app/                          # Streamlit application files
│   └── app.py                    # Main Streamlit dashboard
├── data/                         # Project datasets
├── docs/                         # Documentation and visualizations
├── models/                       # Saved trained models
├── notebooks/                    # Jupyter notebooks (Source of Truth)
├── src/                          # Core Python logic and scripts
├── requirements.txt              # Python dependencies
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

| Metric                           | Score   |
| -------------------------------- | ------- |
| Profit Classification Accuracy   | 96.98%  |
| Profit Classification Recall     | 96%     |
| Revenue Regression R² Score      | ~0.72   |

---

### **Interactive Application Deployment**

The project features an interactive **Streamlit Web Application** with a clean 3-column floating island UI, enabling marketing managers to input campaign parameters and receive real-time revenue and profitability forecasts.

#### **To Launch the Platform Locally:**
```powershell
streamlit run app/app.py
```

---

### **Technology Stack**

| Category             | Tools                          |
| -------------------- | ------------------------------ |
| Programming          | Python                         |
| Data Processing      | Pandas, NumPy                  |
| Database             | PostgreSQL, SQLAlchemy         |
| Machine Learning     | Scikit-learn, XGBoost          |
| Imbalanced Learning  | imbalanced-learn (SMOTETomek)  |
| Visualization        | Plotly                         |
| Web Framework        | Streamlit                      |

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

### **4. Run Ingestion & Training**

```bash
python src/data_ingestion.py
python src/train_models.py
```

---

### **5. Launch Dashboard**

```bash
streamlit run app/app.py
```

---

### **Example Use Case**

A marketing analytics team can use this platform to:

1. Forecast expected revenue for a proposed campaign before launch
2. Identify campaigns at risk of being unprofitable
3. Optimize channel mix based on CTR and conversion rate trends
4. Allocate budgets across brands based on model-driven profitability predictions

---

### **Future Improvements**

* Real-time campaign performance API integration
* Multi-objective optimization for spend vs. revenue tradeoffs
* SHAP-based model explainability dashboard
* Automated campaign performance alerting system

---

### **Contributors**

* **Jegadeesh D** — Data ingestion, feature engineering, XGBoost modeling, imbalanced learning, and Streamlit dashboard development

---

### **License**

MIT License
