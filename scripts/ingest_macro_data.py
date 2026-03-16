import os
import requests
import logging
import argparse
from datetime import datetime, timedelta
from sqlalchemy.orm import sessionmaker
from db import engine, MacroIndicator
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FRED_API_KEY = os.getenv("FRED_API_KEY")
BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

MACRO_INDICATORS = {
    'FEDFUNDS': 'Effective Federal Funds Rate',
    'CPIAUCSL': 'Consumer Price Index',
    'DGS10': '10-Year Treasury CMR'
}

def fetch_macro_data(session, series_id, indicator_name, start_date, end_date):
    if not FRED_API_KEY or FRED_API_KEY == 'your_fred_api_key':
        logger.warning("FRED_API_KEY is not set or invalid.")
        return 0
        
    params = {
        'series_id': series_id,
        'api_key': FRED_API_KEY,
        'file_type': 'json',
        'observation_start': start_date,
        'observation_end': end_date
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        observations = data.get('observations', [])
        records_added = 0
        for obs in observations:
            date_str = obs.get('date')
            value_str = obs.get('value')
            if date_str and value_str and value_str != '.':
                date_val = datetime.strptime(date_str, '%Y-%m-%d').date()
                val = float(value_str)
                
                # Check if exists
                indicator = session.query(MacroIndicator).filter_by(indicator_name=indicator_name, date=date_val).first()
                if not indicator:
                    indicator = MacroIndicator(indicator_name=indicator_name, date=date_val)
                    session.add(indicator)
                    records_added += 1
                
                indicator.value = val
                
        session.commit()
        return records_added
    except Exception as e:
        logger.error(f"Error fetching data for {indicator_name}: {e}")
        session.rollback()
        return 0

def run_ingestion(mode='daily'):
    if not engine:
        logger.error("Database connection not established. Export DATABASE_URL.")
        return

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    
    end_date = datetime.today()
    if mode == 'historical':
        start_date = end_date - timedelta(days=3650) # 10 years
        logger.info(f"Running historical macro data ingestion from {start_date.date()} to {end_date.date()}")
    else:
        start_date = end_date - timedelta(days=30) # get at least a month due to reporting delays
        logger.info(f"Running daily macro data ingestion back to {start_date.date()}")
        
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    total_added = 0
    tqdm_indicators = tqdm(MACRO_INDICATORS.items(), desc="Ingesting Macro Data")
    for series_id, name in tqdm_indicators:
        added = fetch_macro_data(session, series_id, name, start_date_str, end_date_str)
        tqdm_indicators.set_description(f"Ingesting {name}")
        total_added += added
        
    logger.info(f"Macro data ingestion complete. Added/Updated {total_added} records.")
    session.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['daily', 'historical'], default='daily')
    args = parser.parse_args()
    run_ingestion(args.mode)
