# Marketing Campaign Performance Analysis

---

### **Project Overview**

Marketing campaigns generate vast streams of transactional, behavioral, and performance data. Optimizing marketing spend requires analyzing historical performance, cleaning inconsistent records, and building predictive analytics models. This project builds a **Marketing Campaign Performance Analysis** platform to inspect, clean, store, and analyze marketing campaign results.

The system ingests multi-source campaign data (e.g. Nykaa, Purplle, Tira), saves it to a central PostgreSQL database, runs predictive models to assess campaign success, and provides interactive trends via a Streamlit analytics dashboard.

---

### **Key Features**

* **Multi-Source Ingestion Pipeline:** Streams raw CSV records from different cosmetics platforms into a PostgreSQL database.
* **Intelligent Data Preprocessing:** Cleans null values, standardizes acquisition columns, and formats campaign dates.
* **Predictive Performance Modeling:** Trains models to forecast campaign success and revenue metrics.
* **Interactive Streamlit Dashboard:** Visualizes ROI, customer acquisition cost, conversion rates, and channel effectiveness.
* **PostgreSQL Schema Optimization:** Employs relational mapping to track impressions, clicks, conversions, and spend.

---

### **Dataset**

* **Source:** Multi-Platform Cosmetics Marketing Campaigns
* **Coverage:** Nykaa, Purplle, and Tira performance logs
* **Data Type:** High-dimensional relational campaign tables

#### **Included Files**

* `nykaa_campaign_data_with_nulls.csv`
* `purplle_campaign_data_with_nulls.csv`
* `tira_campaign_data_with_nulls.csv`

---

### **Project Structure**

```bash
Marketing-Campaign-Performance/
│
├── data/                         # Multi-source campaign CSV files
│
├── src/
│   ├── data_ingestion.py         # PostgreSQL ingestion pipeline
│   ├── data_preprocessing.py     # Preprocessing and standardizing script
│   └── train_models.py           # ML training pipeline for campaign ROI
│
├── app/
│   └── app.py                    # Streamlit analytics dashboard
│
├── models/                       # Saved trained model artifacts
│
├── .gitignore
└── README.md
```

---

### **How It Works**

### **1. Data Ingestion & Preprocessing**

* **`data_ingestion.py`**: Reads raw CSV datasets, standardizes headers, and loads records into target database tables.
* **`data_preprocessing.py`**: Handles missing values (e.g. imputing average click-through rates), cleans outliers, and formats column datatypes.

---

### **2. Predictive ML Pipeline**

Runs regression models to predict campaign performance indicators:

```python
# From src/train_models.py
from sklearn.ensemble import RandomForestRegressor

# Example training script logic
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

---

### **Database Configuration**

Ensure database connection configurations are configured appropriately to point to your PostgreSQL setup:
* **Database Name:** `marketing_campaigns`
* **Username:** `postgres`
* **Password:** `jaundice`
* **Port:** `5432`

---

### **Interactive Application Deployment**

The project features a **Streamlit Web Application** displaying real-time metrics, conversion funnel trends, and channel performance comparisons.

#### **To Launch the Platform Locally:**
```powershell
python -m streamlit run ".\Marketing Campaign Performance Analysis\app\app.py"
```

---

### **Technology Stack**

| Category             | Tools                                         |
| -------------------- | --------------------------------------------- |
| Programming          | Python                                        |
| Database Engine      | PostgreSQL                                    |
| Database Connection  | SQLAlchemy, Psycopg2                          |
| Data Processing      | Pandas, NumPy                                 |
| Machine Learning     | Scikit-learn                                  |
| Web Framework        | Streamlit                                     |
| Visualization        | Matplotlib, Seaborn, Plotly                   |

---

### **Getting Started**

### **1. Setup Database**

Create a PostgreSQL database named `marketing_campaigns` and ensure your database server is running.

---

### **2. Install Dependencies**

```bash
pip install pandas numpy streamlit psycopg2 sqlalchemy scikit-learn plotly matplotlib seaborn python-dotenv
```

---

### **3. Configure Environment Variables**

Create a `.env` file in the root of the project folder:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=marketing_campaigns
DB_USER=postgres
DB_PASSWORD=your_postgres_password
```

---

### **3. Run Processing & Training Pipelines**

```bash
# Ingest data to PostgreSQL
python src/data_ingestion.py

# Clean data
python src/data_preprocessing.py

# Train ML models
python src/train_models.py
```

---

### **4. Launch the Dashboard**

Start the Streamlit application server:

```bash
streamlit run app/app.py
```

---

### **Example Use Case**

Marketing managers can use the system to:
1. Compare performance trends across Nykaa, Purplle, and Tira channels.
2. View predicted conversion numbers before starting a campaign.
3. Optimize marketing budgets by allocating spend to channels with higher historical conversion rates.

---

### **Future Improvements**

* Integration with real-time marketing APIs (Google Ads, Meta Ads APIs).
* Advanced NLP sentiment analysis on campaign feedback comments.
* Multi-touch attribution modeling.

---

### **Contributors**

* **Jegadeesh D** — Database pipeline construction, machine learning model development, preprocessing scripts, and dashboard visualization

---

### **License**

MIT License
