import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_db_url(dbname="marketing_campaign"):
    from urllib.parse import quote_plus
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD")
    db_password_encoded = f":{quote_plus(db_password)}" if db_password else ""
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", dbname)
    
    return f"postgresql://{db_user}{db_password_encoded}@{db_host}:{db_port}/{db_name}"

def get_engine(dbname="marketing_campaign"):
    return create_engine(get_db_url(dbname))
