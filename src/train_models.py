import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import math
import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from src.data_preprocessing import load_and_clean_data

def train_pipelines():
    df = load_and_clean_data()
    
    # Define features based on types
    cat_features = ['campaign_type', 'target_audience', 'language', 'customer_segment', 'brand']
    num_features_base = ['impressions', 'clicks', 'leads', 'conversions', 'engagement_score', 'month_sin', 'month_cos', 'ctr', 'conversion_rate', 'cpl']
    channel_features = [col for col in df.columns if col.startswith('channel_') and col != 'channel_used']
    
    # Preprocessor for categorical and numerical variables
    preprocessor_reg = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features_base + ['acquisition_cost']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ], remainder='passthrough'
    )
    
    preprocessor_cls = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features_base + ['revenue', 'acquisition_cost']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ], remainder='passthrough'
    )
    
    # --- MODEL 1: REGRESSION (Predicting Revenue) ---
    print("\n--- Training Revenue Regressor ---")
    X_reg = df[cat_features + num_features_base + channel_features + ['acquisition_cost']]
    y_reg = df['revenue']
    
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    reg_pipeline = SklearnPipeline(steps=[
        ('preprocessor', clone(preprocessor_reg)),
        ('regressor', XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1))
    ])
    
    param_grid = {
        'regressor__max_depth': [3, 6],
        'regressor__learning_rate': [0.05, 0.1]
    }
    print("Starting Grid Search for Regression Model...")
    grid_search = GridSearchCV(reg_pipeline, param_grid, cv=2, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train_r, y_train_r)
    
    best_reg_pipeline = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")
    
    preds_r = best_reg_pipeline.predict(X_test_r)
    print(f"Regression R² Score: {r2_score(y_test_r, preds_r):.4f}") 
    print(f"Regression RMSE: {math.sqrt(mean_squared_error(y_test_r, preds_r)):.2f}") # heavily penalizes outliers
    
    # --- MODEL 2: CLASSIFICATION (Predicting Profit Flag - No Data Leakage) ---
    print("\n--- Training Profit Classifier ---")
    # Added 'revenue' and 'acquisition_cost' based on project guidelines
    X_cls = df[cat_features + num_features_base + channel_features + ['revenue', 'acquisition_cost']]
    y_cls = df['profit_flag']
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
    
    # Calculate class imbalance ratio dynamically to pass to XGBoost
    positive_cases = sum(y_train_c == 1)
    negative_cases = sum(y_train_c == 0)
    scale_pos_weight = negative_cases / max(1, positive_cases)
    
    cls_pipeline = SklearnPipeline(steps=[
        ('preprocessor', clone(preprocessor_cls)),
        ('classifier', XGBClassifier(n_estimators=500, max_depth=15, learning_rate=0.05, random_state=42, n_jobs=-1, scale_pos_weight=scale_pos_weight))
    ])
    
    cls_pipeline.fit(X_train_c, y_train_c)
    preds_c = cls_pipeline.predict(X_test_c)
    print(f"Classification Accuracy: {accuracy_score(y_test_c, preds_c):.4f}")
    print(classification_report(y_test_c, preds_c))
    
    # Save artifacts safely
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_reg_pipeline, 'models/revenue_regressor.joblib')
    joblib.dump(cls_pipeline, 'models/profit_classifier.joblib')
    print("Saved pipeline models to disk!")

if __name__ == "__main__":
    train_pipelines()