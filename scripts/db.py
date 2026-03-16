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
    source = Column(String(50))
    timestamp = Column(DateTime)
    text_content = Column(Text)
    ticker_mentioned = Column(String(10), nullable=True)

def init_db():
    if engine:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully.")
    else:
        logger.error("Cannot initialize database without DATABASE_URL.")

if __name__ == "__main__":
    init_db()
