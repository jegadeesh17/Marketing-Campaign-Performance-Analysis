# 🎯 Multi-Brand Marketing Campaign Performance Analysis

An end-to-end machine learning and analytics platform designed to clean multi-brand marketing campaign datasets, train predictive models, and deploy them to forecast campaign revenue and profitability before capital is deployed.

## 📖 Project Overview
Marketing campaigns generate massive streams of performance indicators (impressions, clicks, conversions, spend). Modeling this data often suffers from multi-source inconsistencies, complex categorical features (like multi-channel tags), and risks of target leakage.

This project builds a complete pipeline from raw ingestion (PostgreSQL) to an active web prediction dashboard (Streamlit). We successfully achieved a **96.98% classification accuracy** for predicting campaign profitability.

## 🏗️ Architecture & Pipeline

1. **Data Ingestion & Preprocessing**: An ingestion pipeline using SQLAlchemy to load raw CSVs from Nykaa, Purplle, and Tira into a central PostgreSQL database. Missing values are handled via mode imputation (categorical) and median imputation (numerical) to resist outliers.
2. **Advanced Feature Engineering**: 
   - **Cyclical Time Encoding**: Sine/Cosine transformations on the month feature to preserve seasonality.
   - **Advanced Ratios**: Calculation of Click-Through Rate (CTR), Conversion Rate, and Cost Per Lead (CPL).
   - **Multi-Label Parsing**: Binary feature extraction for multi-channel arrays (YouTube, Instagram, Email, etc.).
3. **Machine Learning Pipelines**: 
   - **Revenue Regression**: An XGBoost Regressor pipeline to forecast exact campaign revenue (R² Score: ~0.72).
   - **Profit Classification**: An XGBoost Classifier utilizing SMOTETomek to perfectly balance classes, achieving >96% accuracy and 96% recall in identifying unprofitable campaigns.
4. **Dashboard Deployment**: An interactive Streamlit application that provides real-time revenue and profit forecasts for marketing managers using a clean, 3-column "floating island" UI.

## 🚀 How to Run

1. **Verify Database Configuration**: Ensure PostgreSQL contains the `marketing_campaign` database.
2. **Execute Ingestion & ML Pipeline**:
   ```bash
   python src/data_ingestion.py
   python src/train_models.py
   ```
3. **Run the Streamlit Dashboard**:
   ```bash
   streamlit run app/app.py
   ```
