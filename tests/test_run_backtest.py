"""
Tests for run_backtest.py
----------------------------
Validates the backtesting engine components.
"""

import pytest
import sys
import os

import numpy as np
import pandas as pd
import backtrader as bt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../scripts")))

from run_backtest import MLSignalStrategy, ParquetData, _compute_benchmark_return


@pytest.fixture
def price_df():
    """Create sample price data for backtesting."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.normal(0.1, 1, len(dates)))
    volume = np.random.randint(1_000_000, 50_000_000, len(dates)).astype(float)
    
    df = pd.DataFrame({
        "close": close,
        "volume": volume,
        "ticker": "SPY",
    }, index=dates)
    return df


@pytest.fixture
def prediction_df(price_df):
    """Create sample predictions matching the price data."""
    np.random.seed(42)
    dates = price_df.index
    
    return pd.DataFrame({
        "ticker": "SPY",
        "date": dates,
        "predicted_signal": np.random.choice([0, 1], len(dates)),
        "predicted_probability": np.random.uniform(0.3, 0.9, len(dates)),
    })


class TestMLSignalStrategy:
    def test_strategy_runs_without_error(self, price_df, prediction_df):
        """Strategy should complete a backtest without crashing."""
        cerebro = bt.Cerebro()
        cerebro.addstrategy(
            MLSignalStrategy,
            predictions=prediction_df,
            prob_threshold=0.55,
            max_positions=5,
        )
        
        ticker_data = price_df[["close", "volume"]].copy()
        data = ParquetData(dataname=ticker_data, name="SPY")
        cerebro.adddata(data)
        cerebro.broker.setcash(10000)
        
        results = cerebro.run()
        assert results is not None
        assert len(results) == 1
    
    def test_strategy_respects_threshold(self, price_df):
        """With a threshold of 1.0, no trades should execute."""
        dates = price_df.index
        # All signals are buy but probability is below threshold
        predictions = pd.DataFrame({
            "ticker": "SPY",
            "date": dates,
            "predicted_signal": np.ones(len(dates), dtype=int),
            "predicted_probability": np.full(len(dates), 0.5),  # Below 1.0 threshold
        })
        
        cerebro = bt.Cerebro()
        cerebro.addstrategy(
            MLSignalStrategy,
            predictions=predictions,
            prob_threshold=1.0,  # Impossibly high
            max_positions=5,
        )
        
        ticker_data = price_df[["close", "volume"]].copy()
        data = ParquetData(dataname=ticker_data, name="SPY")
        cerebro.adddata(data)
        cerebro.broker.setcash(10000)
        
        results = cerebro.run()
        strategy = results[0]
        
        # No trades should have been made
        assert len(strategy.trade_log) == 0
    
    def test_strategy_generates_trades(self, price_df):
        """With low threshold, some trades should execute."""
        dates = price_df.index
        n = len(dates)
        signals = np.tile([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], n // 10 + 1)[:n]
        probs = np.tile([0.8, 0.8, 0.8, 0.8, 0.8, 0.3, 0.3, 0.3, 0.3, 0.3], n // 10 + 1)[:n]
        predictions = pd.DataFrame({
            "ticker": "SPY",
            "date": dates,
            "predicted_signal": signals,
            "predicted_probability": probs,
        })
        
        cerebro = bt.Cerebro()
        cerebro.addstrategy(
            MLSignalStrategy,
            predictions=predictions,
            prob_threshold=0.5,
            max_positions=5,
        )
        
        ticker_data = price_df[["close", "volume"]].copy()
        data = ParquetData(dataname=ticker_data, name="SPY")
        cerebro.adddata(data)
        cerebro.broker.setcash(10000)
        
        results = cerebro.run()
        strategy = results[0]
        
        # At least some trades should have been made
        assert len(strategy.trade_log) > 0
    
    def test_initial_capital_preserved_without_trades(self, price_df):
        """With no predictions, capital should remain unchanged."""
        cerebro = bt.Cerebro()
        cerebro.addstrategy(
            MLSignalStrategy,
            predictions=pd.DataFrame(),  # Empty predictions
            prob_threshold=0.55,
            max_positions=5,
        )
        
        ticker_data = price_df[["close", "volume"]].copy()
        data = ParquetData(dataname=ticker_data, name="SPY")
        cerebro.adddata(data)
        cerebro.broker.setcash(10000)
        
        cerebro.run()
        assert cerebro.broker.getvalue() == 10000.0


class TestBenchmark:
    def test_benchmark_return_calculation(self, price_df):
        """Benchmark return should match simple start-to-end calculation."""
        ret = _compute_benchmark_return(price_df, "2024-01-01", None)
        
        spy_data = price_df[price_df["ticker"] == "SPY"].sort_index()
        expected = (spy_data["close"].iloc[-1] - spy_data["close"].iloc[0]) / spy_data["close"].iloc[0] * 100
        
        assert abs(ret - expected) < 0.01
    
    def test_benchmark_empty_returns_zero(self):
        """Should return 0 if no SPY data is available."""
        empty_df = pd.DataFrame(columns=["ticker", "close"], dtype=float)
        empty_df["ticker"] = empty_df["ticker"].astype(str)
        ret = _compute_benchmark_return(empty_df, "2024-01-01", None)
        assert ret == 0.0


class TestTradeLog:
    def test_trade_log_format(self, price_df):
        """Trade log entries should have all required fields."""
        dates = price_df.index
        n = len(dates)
        signals = np.tile([1, 1, 1, 0, 0], n // 5 + 1)[:n]
        probs = np.tile([0.8, 0.8, 0.8, 0.3, 0.3], n // 5 + 1)[:n]
        predictions = pd.DataFrame({
            "ticker": "SPY",
            "date": dates,
            "predicted_signal": signals,
            "predicted_probability": probs,
        })
        
        cerebro = bt.Cerebro()
        cerebro.addstrategy(
            MLSignalStrategy,
            predictions=predictions,
            prob_threshold=0.5,
            max_positions=5,
        )
        
        ticker_data = price_df[["close", "volume"]].copy()
        data = ParquetData(dataname=ticker_data, name="SPY")
        cerebro.adddata(data)
        cerebro.broker.setcash(10000)
        
        results = cerebro.run()
        strategy = results[0]
        
        if strategy.trade_log:
            trade = strategy.trade_log[0]
            assert "ticker" in trade
            assert "entry_date" in trade
            assert "exit_date" in trade
            assert "entry_price" in trade
            assert "exit_price" in trade
            assert "pnl_pct" in trade
            assert "duration_days" in trade
