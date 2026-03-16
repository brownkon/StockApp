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

TICKERS = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GLD']

def fetch_and_store_data(session, ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            logger.warning(f"No data found for {ticker} from {start_date} to {end_date}")
            return 0
        
        records_added = 0
        for date, row in data.iterrows():
            date_val = date.date()
            
            # Extract scalar values from the yfinance DataFrame
            # Sometimes yfinance returns a MultiIndex column DataFrame if multiple tickers are retrieved,
            # but here we download one by one, so it should be simple.
            open_val = row['Open'].item() if hasattr(row['Open'], 'item') else row['Open']
            high_val = row['High'].item() if hasattr(row['High'], 'item') else row['High']
            low_val = row['Low'].item() if hasattr(row['Low'], 'item') else row['Low']
            close_val = row['Close'].item() if hasattr(row['Close'], 'item') else row['Close']
            
            # Backwards compatibility: newer yfinance uses "Adj Close", older might be different
            adj_close_val = row['Adj Close'].item() if 'Adj Close' in row else close_val
            if not 'Adj Close' in row and hasattr(adj_close_val, 'item'): 
                adj_close_val = adj_close_val.item()
                
            volume_val = row['Volume'].item() if hasattr(row['Volume'], 'item') else row['Volume']
            
            # Upsert logic - using merge for basic SQLAlchemy
            price = session.query(DailyPrice).filter_by(ticker=ticker, date=date_val).first()
            if not price:
                price = DailyPrice(ticker=ticker, date=date_val)
                session.add(price)
                records_added += 1
                
            price.open = float(open_val)
            price.high = float(high_val)
            price.low = float(low_val)
            price.close = float(close_val)
            price.adj_close = float(adj_close_val)
            price.volume = float(volume_val)
            
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
        start_date = end_date - timedelta(days=3650) # 10 years back
        logger.info(f"Running historical market data ingestion for 10 years from {start_date.date()} to {end_date.date()}")
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
