import os
import logging
from sqlalchemy import create_engine, Column, String, Date, Float, Integer, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Connection URI from Supabase
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL) if DATABASE_URL else None
if engine is None:
    logger.warning("DATABASE_URL not found in environment variables.")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class DailyPrice(Base):
    __tablename__ = 'daily_prices'
    ticker = Column(String(10), primary_key=True)
    date = Column(Date, primary_key=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adj_close = Column(Float)
    volume = Column(Float)

class MacroIndicator(Base):
    __tablename__ = 'macro_indicators'
    indicator_name = Column(String(50), primary_key=True)
    date = Column(Date, primary_key=True)
    value = Column(Float)

class RawSentimentText(Base):
    __tablename__ = 'raw_sentiment_text'
    id = Column(Integer, primary_key=True, autoincrement=True)
    external_id = Column(String(255), unique=True, nullable=True)
    source = Column(String(50))
    timestamp = Column(DateTime)
    text_content = Column(Text)
    ticker_mentioned = Column(String(10), nullable=True)

class TechnicalIndicator(Base):
    __tablename__ = 'technical_indicators'
    ticker = Column(String(10), primary_key=True)
    date = Column(Date, primary_key=True)
    
    # Moving Averages (Trend)
    sma_50 = Column(Float)   # Intermediate trend
    sma_200 = Column(Float)  # Long term trend
    ema_20 = Column(Float)   # Short term momentum

    # Oscillators (Overbought / Oversold / Momentum)
    rsi_14 = Column(Float)   # Relative Strength Index
    
    # MACD (Trend & Momentum)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_hist = Column(Float)
    
    # Bollinger Bands (Volatility)
    bb_upper = Column(Float)
    bb_lower = Column(Float)
    bb_mid = Column(Float)
    
    # Average True Range (Volatility / Stop loss placement)
    atr_14 = Column(Float)
    
    # Directional Movement (Trend Strength)
    adx_14 = Column(Float)
    plus_di_14 = Column(Float)
    minus_di_14 = Column(Float)
    
    # Volume
    obv = Column(Float)      # On-Balance Volume

class DailyOptionsData(Base):
    __tablename__ = 'daily_options_data'
    ticker = Column(String(10), primary_key=True)
    date = Column(Date, primary_key=True)
    put_volume = Column(Float)
    call_volume = Column(Float)
    put_call_ratio = Column(Float)
    implied_volatility = Column(Float)

def init_db():
    if engine:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully.")
    else:
        logger.error("Cannot initialize database without DATABASE_URL.")

if __name__ == "__main__":
    init_db()
