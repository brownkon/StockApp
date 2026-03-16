import argparse
import logging
from datetime import datetime, date
import yfinance as yf
from sqlalchemy.orm import sessionmaker
from db import engine, DailyOptionsData
from ingest_market_data import TICKERS
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_and_store_options(session, ticker, target_date):
    try:
        tkr = yf.Ticker(ticker)
        options_dates = tkr.options
        if not options_dates:
            logger.debug(f"No options dates found for {ticker}")
            return 0
            
        total_call_vol = 0
        total_put_vol = 0
        
        # We'll just look at the nearest a few expirations to get a proxy for sentiment
        for opt_date in options_dates[:3]:
            opt_chain = tkr.option_chain(opt_date)
            # Sum volume, dropping NaNs
            calls_vol = opt_chain.calls['volume'].sum()
            puts_vol = opt_chain.puts['volume'].sum()
            
            total_call_vol += calls_vol if not pd.isna(calls_vol) else 0
            total_put_vol += puts_vol if not pd.isna(puts_vol) else 0
            
        if total_call_vol == 0 and total_put_vol == 0:
            return 0
            
        put_call_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else 0
        
        # Check existing
        existing = session.query(DailyOptionsData).filter(
            DailyOptionsData.ticker == ticker,
            DailyOptionsData.date == target_date
        ).first()
        
        if existing:
            existing.put_volume = float(total_put_vol)
            existing.call_volume = float(total_call_vol)
            existing.put_call_ratio = float(put_call_ratio)
            # Yfinance doesn't cleanly give aggregate IV easily without a specific strike, we'll leave it 0 or log standard VIX instead for broad market
            existing.implied_volatility = 0.0 
        else:
            opt_data = DailyOptionsData(
                ticker=ticker,
                date=target_date,
                put_volume=float(total_put_vol),
                call_volume=float(total_call_vol),
                put_call_ratio=float(put_call_ratio),
                implied_volatility=0.0
            )
            session.add(opt_data)
            
        session.commit()
        return 1
        
    except Exception as e:
        logger.error(f"Error fetching options for {ticker}: {e}")
        session.rollback()
        return 0

import pandas as pd

def run_ingestion(mode='daily'):
    if not engine:
        logger.error("Database connection not established. Export DATABASE_URL.")
        return

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    
    today = date.today()
    if mode == 'historical':
        logger.warning("Historical options ingestion not supported effectively by free yfinance. Doing today only.")
        
    logger.info(f"Running daily options data ingestion for {today}")
        
    total_added = 0
    tqdm_tickers = tqdm(TICKERS, desc="Ingesting Options Data")
    for ticker in tqdm_tickers:
        tqdm_tickers.set_description(f"Options for {ticker}")
        added = fetch_and_store_options(session, ticker, today)
        total_added += added
        
    logger.info(f"Options data ingestion complete. Added/Updated {total_added} records.")
    session.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['daily', 'historical'], default='daily')
    args = parser.parse_args()
    run_ingestion(args.mode)
