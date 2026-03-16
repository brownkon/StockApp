import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import date, datetime

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

from db import Base, DailyPrice, MacroIndicator, RawSentimentText

@pytest.fixture(scope="module")
def engine():
    return create_engine('sqlite:///:memory:')

@pytest.fixture(scope="module")
def setup_database(engine):
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def session(engine, setup_database):
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

def test_daily_price_model(session):
    price = DailyPrice(
        ticker="SPY",
        date=date.today(),
        open=100.0,
        high=105.0,
        low=95.0,
        close=101.0,
        adj_close=101.0,
        volume=1000000
    )
    session.add(price)
    session.commit()
    
    saved_price = session.query(DailyPrice).filter_by(ticker="SPY").first()
    assert saved_price is not None
    assert saved_price.open == 100.0
    assert saved_price.volume == 1000000.0

def test_macro_indicator_model(session):
    indicator = MacroIndicator(
        indicator_name="FEDFUNDS",
        date=date.today(),
        value=5.33
    )
    session.add(indicator)
    session.commit()
    
    saved_ind = session.query(MacroIndicator).filter_by(indicator_name="FEDFUNDS").first()
    assert saved_ind is not None
    assert saved_ind.value == 5.33

def test_raw_sentiment_text_model(session):
    sentiment = RawSentimentText(
        source="Reddit",
        timestamp=datetime.now(),
        text_content="Bulls are back!"
    )
    session.add(sentiment)
    session.commit()
    
    saved_sen = session.query(RawSentimentText).filter_by(source="Reddit").first()
    assert saved_sen is not None
    assert "Bulls" in saved_sen.text_content
