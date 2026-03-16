import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sqlalchemy.orm import sessionmaker
from db import engine, DailyPrice, TechnicalIndicator
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_rsi(series, period=14):
    delta = series.diff()
    up = delta.where(delta > 0, 0)
    down = -delta.where(delta < 0, 0)
    avg_up = up.rolling(window=period, min_periods=period).mean()
    avg_down = down.rolling(window=period, min_periods=period).mean()
    rs = avg_up / avg_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, fast_period=12, slow_period=26, signal_period=9):
    ema_fast = series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def calculate_bb(series, period=20, std_dev=2):
    bb_mid = series.rolling(window=period).mean()
    bb_std = series.rolling(window=period).std()
    bb_upper = bb_mid + (bb_std * std_dev)
    bb_lower = bb_mid - (bb_std * std_dev)
    return bb_upper, bb_lower, bb_mid

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def calculate_adx(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    tr = np.max(ranges, axis=1)

    up_move = df['high'] - df['high'].shift()
    down_move = df['low'].shift() - df['low']
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_smooth = tr.rolling(window=period).mean()
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=period).mean()
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=period).mean()
    
    # Avoid division by zero
    tr_smooth = tr_smooth.replace(0, np.nan)
    
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return plus_di, minus_di, adx

def calculate_obv(df):
    direction = np.where(df['close'] > df['close'].shift(1), 1, -1)
    direction[0] = 0
    obv = (direction * df['volume']).cumsum()
    return obv

def process_ticker(session, ticker):
    # Retrieve all prices for ticker ordered by date
    prices = session.query(DailyPrice).filter(DailyPrice.ticker == ticker).order_by(DailyPrice.date.asc()).all()
    if not prices:
        return 0
    
    # Convert to dataframe
    df = pd.DataFrame([{
        'date': p.date,
        'open': p.open,
        'high': p.high,
        'low': p.low,
        'close': p.close,
        'adj_close': p.adj_close,
        'volume': p.volume
    } for p in prices])
    
    # Calculate indicators
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    df['rsi_14'] = calculate_rsi(df['close'], period=14)
    
    macd, signal, hist = calculate_macd(df['close'])
    df['macd'] = macd
    df['macd_signal'] = signal
    df['macd_hist'] = hist
    
    bb_upper, bb_lower, bb_mid = calculate_bb(df['close'])
    df['bb_upper'] = bb_upper
    df['bb_lower'] = bb_lower
    df['bb_mid'] = bb_mid
    
    df['atr_14'] = calculate_atr(df)
    
    plus_di, minus_di, adx = calculate_adx(df)
    df['plus_di_14'] = plus_di
    df['minus_di_14'] = minus_di
    df['adx_14'] = adx
    
    df['obv'] = calculate_obv(df)
    
    # Existing technical indicators for update logic
    existing_records = session.query(TechnicalIndicator).filter(TechnicalIndicator.ticker == ticker).all()
    existing_map = {r.date: r for r in existing_records}
    
    records_added = 0
    
    # Iterate and save
    new_objs = []
    
    for i, row in df.iterrows():
        row_dict = {}
        for k, v in row.items():
            if pd.isna(v):
                row_dict[k] = None
            else:
                row_dict[k] = v
                
        if row_dict['date'] in existing_map:
            ti = existing_map[row_dict['date']]
            
            # If the record already has the newly added adx_14 indicator, it's fully up to date.
            if getattr(ti, 'adx_14', None) is not None:
                continue
                
            ti.sma_50 = row_dict['sma_50']
            ti.sma_200 = row_dict['sma_200']
            ti.ema_20 = row_dict['ema_20']
            ti.rsi_14 = row_dict['rsi_14']
            ti.macd = row_dict['macd']
            ti.macd_signal = row_dict['macd_signal']
            ti.macd_hist = row_dict['macd_hist']
            ti.bb_upper = row_dict['bb_upper']
            ti.bb_lower = row_dict['bb_lower']
            ti.bb_mid = row_dict['bb_mid']
            ti.atr_14 = row_dict['atr_14']
            ti.adx_14 = row_dict['adx_14']
            ti.plus_di_14 = row_dict['plus_di_14']
            ti.minus_di_14 = row_dict['minus_di_14']
            ti.obv = row_dict['obv']
        else:
            ti = TechnicalIndicator(
                ticker=ticker,
                date=row_dict['date'],
                sma_50=row_dict['sma_50'],
                sma_200=row_dict['sma_200'],
                ema_20=row_dict['ema_20'],
                rsi_14=row_dict['rsi_14'],
                macd=row_dict['macd'],
                macd_signal=row_dict['macd_signal'],
                macd_hist=row_dict['macd_hist'],
                bb_upper=row_dict['bb_upper'],
                bb_lower=row_dict['bb_lower'],
                bb_mid=row_dict['bb_mid'],
                atr_14=row_dict['atr_14'],
                adx_14=row_dict['adx_14'],
                plus_di_14=row_dict['plus_di_14'],
                minus_di_14=row_dict['minus_di_14'],
                obv=row_dict['obv']
            )
            new_objs.append(ti)
            records_added += 1
            
    if new_objs:
        # bulk insert new objects
        session.bulk_save_objects(new_objs)
        
    session.commit()
    return records_added

def run_calculations():
    if not engine:
        logger.error("Database connection not established. Export DATABASE_URL.")
        return

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    
    # Get all distinct tickers
    tickers = session.query(DailyPrice.ticker).distinct().all()
    tickers = [t[0] for t in tickers]
    
    logger.info(f"Calculating technical indicators for {len(tickers)} tickers...")
    
    total_added = 0
    tqdm_tickers = tqdm(tickers, desc="Calculating Indicators")
    for ticker in tqdm_tickers:
        tqdm_tickers.set_description(f"Calculating {ticker}")
        try:
            added = process_ticker(session, ticker)
            total_added += added
        except Exception as e:
            logger.error(f"Error calculating indicators for {ticker}: {e}")
            session.rollback()
            
    logger.info(f"Technical indicators calculation complete. Added {total_added} new records.")
    session.close()

if __name__ == "__main__":
    run_calculations()
