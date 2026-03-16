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

# Expanded indicators to cover employment, inflation, economic output, and credit
MACRO_INDICATORS = {
    # Interest Rates & Yields
    'FEDFUNDS': 'Effective Federal Funds Rate',
    'DGS10': '10-Year Treasury CMR',
    'DGS2': '2-Year Treasury CMR',
    'T10Y2Y': '10-Year Minus 2-Year Treasury Yield Spread',
    'MORTGAGE30US': '30-Year Fixed Rate Mortgage Average',
    
    # Inflation & Employment
    'CPIAUCSL': 'Consumer Price Index (Inflation)',
    'CPILFESL': 'Core CPI (Excluding Food and Energy)',
    'UNRATE': 'Unemployment Rate',
    'PAYEMS': 'Total Nonfarm Payrolls (Employment)',
    'ICSA': 'Initial Claims (Unemployment)',
    
    # Growth & Production
    'GDPC1': 'Real Gross Domestic Product',
    'INDPRO': 'Industrial Production Index',
    'RSAFS': 'Advance Retail Sales',
    'HOUST': 'New Privately-Owned Housing Units Started',
    
    # Credit & Sentiment
    'M2SL': 'M2 Money Supply',
    'PSAVERT': 'Personal Saving Rate',
    'UMCSENT': 'University of Michigan: Consumer Sentiment'
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
        
        start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
        existing_records = session.query(MacroIndicator).filter(
            MacroIndicator.indicator_name == indicator_name,
            MacroIndicator.date >= start_date_obj
        ).all()
        existing_map = {r.date: r for r in existing_records}
        
        records_added = 0
        for obs in observations:
            date_str = obs.get('date')
            value_str = obs.get('value')
            if date_str and value_str and value_str != '.':
                date_val = datetime.strptime(date_str, '%Y-%m-%d').date()
                val = float(value_str)
                
                if date_val in existing_map:
                    indicator = existing_map[date_val]
                    indicator.value = val
                else:
                    indicator = MacroIndicator(indicator_name=indicator_name, date=date_val, value=val)
                    session.add(indicator)
                    records_added += 1
                
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
        start_date = end_date - timedelta(days=14610) # 40 years
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
