import os
import math
import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from data_preprocessing import load_and_clean_data

def train_pipelines():
    df = load_and_clean_data()
    
    # Define features based on types
    cat_features = ['campaign_type', 'target_audience', 'language', 'customer_segment', 'brand']
    num_features = ['duration', 'impressions', 'clicks', 'leads', 'conversions', 'engagement_score', 'month']
    channel_features = [col for col in df.columns if col.startswith('channel_') and col != 'channel_used']
    
    # Preprocessor for categorical variables
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ], remainder='passthrough'
    )
    
    # --- MODEL 1: REGRESSION (Predicting Revenue) ---
    print("\n--- Training Revenue Regressor ---")
    X_reg = df[cat_features + num_features + channel_features + ['acquisition_cost']]
    y_reg = df['revenue']
    
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    reg_pipeline = Pipeline(steps=[
        ('preprocessor', clone(preprocessor)),
        ('regressor', RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1))
    ])
    
    reg_pipeline.fit(X_train_r, y_train_r)
    preds_r = reg_pipeline.predict(X_test_r)
    print(f"Regression R² Score: {r2_score(y_test_r, preds_r):.4f}")
    print(f"Regression RMSE: {math.sqrt(mean_squared_error(y_test_r, preds_r)):.2f}")
    
    # --- MODEL 2: CLASSIFICATION (Predicting Profit Flag - No Data Leakage) ---
    print("\n--- Training Profit Classifier ---")
    # CRITICAL: Excluded 'revenue', 'roi', and 'acquisition_cost' to prevent target leakage
    X_cls = df[cat_features + num_features + channel_features]
    y_cls = df['profit_flag']
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
    
    cls_pipeline = Pipeline(steps=[
        ('preprocessor', clone(preprocessor)),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
    ])
    
    cls_pipeline.fit(X_train_c, y_train_c)
    preds_c = cls_pipeline.predict(X_test_c)
    print(f"Classification Accuracy: {accuracy_score(y_test_c, preds_c):.4f}")
    print(classification_report(y_test_c, preds_c))
    
    # Save artifacts safely
    os.makedirs('models', exist_ok=True)
    joblib.dump(reg_pipeline, 'models/revenue_regressor.joblib')
    joblib.dump(cls_pipeline, 'models/profit_classifier.joblib')
    print("Saved pipeline models to disk!")

if __name__ == "__main__":
    train_pipelines()