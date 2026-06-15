import pandas as pd
import numpy as np
from src.db_config import get_engine

def load_and_clean_data():
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM raw_campaign_data", engine)
    
    # 1. Deduplication
    df = df.drop_duplicates()
    
    # 2. Impute Categorical Columns
    categorical_cols = ['campaign_type', 'target_audience', 'language', 'customer_segment', 'brand']
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
    # 3. Impute Numerical Columns
    numerical_cols = ['impressions', 'clicks', 'leads', 'conversions', 'revenue', 'acquisition_cost', 'engagement_score', 'roi']
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
        
    # 4. Multi-Label Encoding for channel_used
    df['channel_used'] = df['channel_used'].fillna('Unknown')
    channels = ['YouTube', 'Instagram', 'Google', 'WhatsApp', 'Email', 'Facebook']
    for channel in channels:
        df[f'channel_{channel.lower()}'] = df['channel_used'].apply(lambda x: 1 if channel in str(x) else 0)
        
    # 5. Create Target Variable for Classification
    df['profit_flag'] = (df['roi'] > 0).astype(int)
    
    # 6. Parse Date column
    df['date_parsed'] = pd.to_datetime(df['date_str'], format='%d-%m-%Y', errors='coerce') 
    df['month'] = df['date_parsed'].dt.month.fillna(1)
    
    # Cyclical Encoding for month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # 7. Feature Engineering (Derived Metrics)
    df['ctr'] = np.where(df['impressions'] > 0, df['clicks'] / df['impressions'], 0)
    df['conversion_rate'] = np.where(df['clicks'] > 0, df['conversions'] / df['clicks'], 0)
    df['cpl'] = np.where(df['leads'] > 0, df['acquisition_cost'] / df['leads'], 0)
    
    return df

if __name__ == "__main__":
    cleaned_df = load_and_clean_data()
    print("Data preprocessed successfully. Columns:", cleaned_df.columns.tolist())