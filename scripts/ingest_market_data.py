import argparse
import logging
from datetime import datetime, timedelta
import yfinance as yf
from sqlalchemy.orm import sessionmaker
from db import engine, DailyPrice
from tqdm import tqdm
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Shifted focus to broad market indexes, sector ETFs, and bonds instead of individual stocks
TICKERS = [
    # Major US Indexes
    'SPY',   # S&P 500 ETF
    'QQQ',   # Nasdaq 100 ETF
    'DIA',   # Dow Jones ETF
    'IWM',   # Russell 2000 ETF (Small Cap)
    'MDY',   # S&P MidCap 400 ETF
    '^VIX',  # Volatility Index
    
    # International & Emerging Markets
    'EFA',   # MSCI EAFE ETF (Developed Markets ex-US/Canada)
    'EEM',   # MSCI Emerging Markets ETF
    
    # Commodities & Real Estate
    'GLD',   # Gold
    'SLV',   # Silver
    'USO',   # US Oil Fund
    'VNQ',   # Real Estate (REITs)
    
    # Bonds
    'TLT',   # 20+ Year Treasury Bond
    'IEF',   # 7-10 Year Treasury Bond
    'SHY',   # 1-3 Year Treasury Bond
    'LQD',   # Investment Grade Corporate Bond
    'HYG',   # High Yield Corporate Bond
    
    # US Sector ETFs
    'XLE',   # Energy Sector
    'XLF',   # Financial Sector
    'XLV',   # Healthcare Sector
    'XRT',   # Retail Sector
    'XLK',   # Technology Sector
    'XLI',   # Industrial Sector
    'XLB',   # Materials Sector
    'XLP',   # Consumer Staples Sector
    'XLY',   # Consumer Discretionary Sector
    'XLU'    # Utilities Sector
]

def fetch_and_store_data(session, ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            logger.warning(f"No data found for {ticker} from {start_date} to {end_date}")
            return 0
        
        # Load all existing records for this ticker in a single query
        start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
        existing_records = session.query(DailyPrice).filter(
            DailyPrice.ticker == ticker,
            DailyPrice.date >= start_date_obj
        ).all()
        existing_map = {r.date: r for r in existing_records}
        
        new_objs = []
        records_added = 0
        for date, row in data.iterrows():
            date_val = date.date()
            
            if date_val in existing_map:
                continue # Skip overwriting historical data to save massive network time
            
            open_val = row['Open'].item() if hasattr(row['Open'], 'item') else row['Open']
            high_val = row['High'].item() if hasattr(row['High'], 'item') else row['High']
            low_val = row['Low'].item() if hasattr(row['Low'], 'item') else row['Low']
            close_val = row['Close'].item() if hasattr(row['Close'], 'item') else row['Close']
            
            adj_close_val = row['Adj Close'].item() if 'Adj Close' in row else close_val
            if not 'Adj Close' in row and hasattr(adj_close_val, 'item'): 
                adj_close_val = adj_close_val.item()
                
            volume_val = row['Volume'].item() if hasattr(row['Volume'], 'item') else row['Volume']
            
            price = DailyPrice(
                ticker=ticker, date=date_val,
                open=float(open_val), high=float(high_val), low=float(low_val),
                close=float(close_val), adj_close=float(adj_close_val), volume=float(volume_val)
            )
            new_objs.append(price)
            records_added += 1
            
        if new_objs:
            session.bulk_save_objects(new_objs)
        session.commit()
        time.sleep(1) # sleep to prevent rate limits
        return records_added
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
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
        start_date = end_date - timedelta(days=14610) # 40 years back
        logger.info(f"Running historical market data ingestion for 40 years from {start_date.date()} to {end_date.date()}")
    else:
        start_date = end_date - timedelta(days=7) # just fetch last week to be safe, db insert uses upsert logic
        logger.info(f"Running daily market data ingestion from {start_date.date()} to {end_date.date()}")
        
    total_added = 0
    tqdm_tickers = tqdm(TICKERS, desc="Ingesting Market Data")
    for ticker in tqdm_tickers:
        tqdm_tickers.set_description(f"Ingesting {ticker}")
        added = fetch_and_store_data(session, ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        total_added += added
        
    logger.info(f"Market data ingestion complete. Added/Updated {total_added} records.")
    session.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['daily', 'historical'], default='daily')
    args = parser.parse_args()
    run_ingestion(args.mode)
