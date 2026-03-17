import pytest
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))
from calculate_indicators import calculate_rsi, calculate_macd, calculate_bb, calculate_atr, calculate_adx

def test_calculate_rsi():
    # Simple test for RSI
    close_prices = pd.Series([100, 102, 104, 103, 105, 106, 108, 110, 109, 111, 112, 115, 114, 116, 118])
    rsi = calculate_rsi(close_prices, period=14)
    # The last value shouldn't be NaN
    assert not pd.isna(rsi.iloc[-1])
    assert rsi.iloc[-1] > 50  # Since it's generally an uptrend

def test_calculate_macd():
    close_prices = pd.Series(np.random.normal(100, 5, 50))
    macd, signal, hist = calculate_macd(close_prices)
    assert not pd.isna(macd.iloc[-1])
    assert not pd.isna(signal.iloc[-1])
    assert not pd.isna(hist.iloc[-1])
    assert len(macd) == 50

def test_calculate_bb():
    close_prices = pd.Series(np.random.normal(100, 5, 50))
    bb_upper, bb_lower, bb_mid = calculate_bb(close_prices, period=20)
    assert pd.isna(bb_upper.iloc[0])  # Should be NaN initially due to rolling window
    assert not pd.isna(bb_upper.iloc[-1])
    assert bb_upper.iloc[-1] > bb_lower.iloc[-1]
    assert bb_lower.iloc[-1] < bb_mid.iloc[-1]

def test_calculate_atr():
    df = pd.DataFrame({
        'high': np.random.normal(105, 2, 50),
        'low': np.random.normal(95, 2, 50),
        'close': np.random.normal(100, 2, 50)
    })
    atr = calculate_atr(df, period=14)
    assert not pd.isna(atr.iloc[-1])
    assert atr.iloc[-1] > 0
    
def test_calculate_adx():
    df = pd.DataFrame({
        'high': np.linspace(100, 150, 50),
        'low': np.linspace(90, 140, 50),
        'close': np.linspace(95, 145, 50)
    })
    plus_di, minus_di, adx = calculate_adx(df, period=14)
    assert not pd.isna(adx.iloc[-1])
    assert plus_di.iloc[-1] > minus_di.iloc[-1]  # Uptrend, so +DI should be > -DI
