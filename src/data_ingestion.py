import os
import pandas as pd
from sqlalchemy import create_engine, text
from db_config import get_engine, get_db_url

def create_db_if_not_exists():
    target_db = 'marketing_campaign'
    # Try connecting to the target database directly first
    try:
        engine = get_engine(target_db)
        with engine.connect() as conn:
            # If this succeeds, the database already exists
            return
    except Exception:
        pass

    # List of possible default databases to connect to for creation
    default_dbs = ['Quandao', 'postgres', 'template1']
    for db in default_dbs:
        try:
            temp_db_url = get_db_url(db)
            temp_engine = create_engine(temp_db_url)
            with temp_engine.connect() as conn:
                conn = conn.execution_options(isolation_level="AUTOCOMMIT")
                result = conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname='{target_db}'"))
                if not result.fetchone():
                    print(f"Database '{target_db}' does not exist. Creating it...")
                    conn.execute(text(f"CREATE DATABASE {target_db}"))
                    print(f"Database '{target_db}' created successfully.")
            temp_engine.dispose()
            break
        except Exception:
            continue

def ingest_data():
    # Ensure the database exists
    create_db_if_not_exists()
    
    # 1. Database Connection URL (Adjust with your PostgreSQL credentials)
    # Format: postgresql://username:password@localhost:5432/database_name
    engine = get_engine()
    
    # 2. File maps with respective brand names
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data")
    
    files = {
        'nykaa': os.path.join(data_dir, 'nykaa_campaign_data_with_nulls.csv'),
        'purplle': os.path.join(data_dir, 'purplle_campaign_data_with_nulls.csv'),
        'tira': os.path.join(data_dir, 'tira_campaign_data_with_nulls.csv')
    }
    
    for brand, file_path in files.items():
        print(f"Ingesting data for brand: {brand}...")
        df = pd.read_csv(file_path)
        
        # Standardize column naming to lowercase matching SQL
        df.columns = [col.lower() for col in df.columns]
        if 'date' in df.columns:
            df = df.rename(columns={'date': 'date_str'})
            
        # Insert the metadata column
        df['brand'] = brand
        
        # Stream data directly into PostgreSQL
        df.to_sql('raw_campaign_data', engine, if_exists='append', index=False)
        print(f"Successfully loaded {len(df)} rows for {brand}.")

if __name__ == "__main__":
    ingest_data()