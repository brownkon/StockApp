"""
Tests for build_features.py
-----------------------------
Validates feature engineering logic including derived features,
target computation, and data leakage prevention.
"""

import pytest
import sys
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../scripts")))

from build_features import compute_derived_features, compute_targets, get_feature_columns


@pytest.fixture
def sample_feature_df():
    """Create a realistic sample DataFrame mimicking merged feature data."""
    dates = pd.date_range("2023-01-01", periods=60, freq="B")
    n = len(dates)
    np.random.seed(42)
    
    close = 100 + np.cumsum(np.random.normal(0, 1, n))
    volume = np.random.randint(5_000_000, 20_000_000, n).astype(float)
    
    df = pd.DataFrame({
        "close": close,
        "volume": volume,
        "sma_50": pd.Series(close).rolling(50).mean().values,
        "sma_200": np.full(n, close.mean()),  # approximate
        "ema_20": pd.Series(close).ewm(span=20).mean().values,
        "rsi_14": np.random.uniform(20, 80, n),
        "macd": np.random.normal(0, 1, n),
        "macd_signal": np.random.normal(0, 0.5, n),
        "macd_hist": np.random.normal(0, 0.5, n),
        "bb_upper": close + 5,
        "bb_lower": close - 5,
        "bb_mid": close,
        "atr_14": np.random.uniform(1, 3, n),
        "adx_14": np.random.uniform(15, 40, n),
        "plus_di_14": np.random.uniform(10, 30, n),
        "minus_di_14": np.random.uniform(10, 30, n),
        "obv": np.cumsum(np.random.choice([-1, 1], n) * volume),
    }, index=dates)
    
    return df


class TestComputeDerivedFeatures:
    def test_returns_dataframe_with_expected_columns(self, sample_feature_df):
        result = compute_derived_features(sample_feature_df.copy())
        
        expected_derived = [
            "price_vs_sma50", "price_vs_sma200", "bb_position",
            "volatility_ratio", "volume_sma_ratio",
            "return_1d", "return_5d", "return_20d",
            "day_of_week", "month",
        ]
        for col in expected_derived:
            assert col in result.columns, f"Missing derived feature: {col}"
    
    def test_price_vs_sma_calculation(self, sample_feature_df):
        """Verify the relative distance from SMA is correctly computed."""
        df = sample_feature_df.copy()
        result = compute_derived_features(df)
        
        # Where sma_50 is not NaN, verify the formula
        valid = result["sma_50"].notna() & (result["sma_50"] != 0)
        if valid.any():
            idx = result[valid].index[0]
            expected = (result.loc[idx, "close"] - result.loc[idx, "sma_50"]) / result.loc[idx, "sma_50"]
            assert abs(result.loc[idx, "price_vs_sma50"] - expected) < 1e-10
    
    def test_bb_position_between_zero_and_one(self, sample_feature_df):
        """Bollinger position should be near 0-1 when price is within bands."""
        df = sample_feature_df.copy()
        # Force close to be between bands
        df["close"] = (df["bb_upper"] + df["bb_lower"]) / 2
        result = compute_derived_features(df)
        
        valid = result["bb_position"].notna()
        values = result.loc[valid, "bb_position"]
        assert (values >= -0.01).all() and (values <= 1.01).all()
    
    def test_day_of_week_range(self, sample_feature_df):
        result = compute_derived_features(sample_feature_df.copy())
        assert result["day_of_week"].min() >= 0
        assert result["day_of_week"].max() <= 4  # Business days
    
    def test_month_range(self, sample_feature_df):
        result = compute_derived_features(sample_feature_df.copy())
        assert result["month"].min() >= 1
        assert result["month"].max() <= 12
    
    def test_returns_not_all_nan(self, sample_feature_df):
        result = compute_derived_features(sample_feature_df.copy())
        assert result["return_1d"].notna().sum() > 0
        assert result["return_5d"].notna().sum() > 0


class TestComputeTargets:
    def test_target_columns_present(self, sample_feature_df):
        result = compute_targets(sample_feature_df.copy())
        assert "fwd_return_1d" in result.columns
        assert "fwd_return_5d" in result.columns
        assert "signal" in result.columns
    
    def test_fwd_return_uses_future_data(self, sample_feature_df):
        """Verify that forward returns use FUTURE close prices (shift negative)."""
        df = sample_feature_df.copy()
        result = compute_targets(df)
        
        # fwd_return_1d at index i should be: close[i+1]/close[i] - 1
        i = 10  # pick a row in the middle
        expected = df["close"].iloc[i + 1] / df["close"].iloc[i] - 1
        assert abs(result["fwd_return_1d"].iloc[i] - expected) < 1e-10
    
    def test_fwd_return_last_rows_nan(self, sample_feature_df):
        """The last row(s) should have NaN for forward returns (no future data)."""
        result = compute_targets(sample_feature_df.copy())
        assert pd.isna(result["fwd_return_1d"].iloc[-1])
        assert pd.isna(result["fwd_return_5d"].iloc[-1])
    
    def test_signal_is_binary(self, sample_feature_df):
        result = compute_targets(sample_feature_df.copy())
        valid_signals = result["signal"].dropna()
        assert set(valid_signals.unique()).issubset({0, 1})
    
    def test_signal_matches_forward_return(self, sample_feature_df):
        """Signal should be 1 when 5-day forward return > 0."""
        result = compute_targets(sample_feature_df.copy())
        valid = result["fwd_return_5d"].notna()
        pos_fwd = result.loc[valid, "fwd_return_5d"] > 0
        assert (result.loc[valid, "signal"] == pos_fwd.astype(int)).all()


class TestNoDataLeakage:
    def test_feature_columns_exclude_targets(self, sample_feature_df):
        """get_feature_columns must not include target columns."""
        df = sample_feature_df.copy()
        df = compute_derived_features(df)
        df = compute_targets(df)
        df["ticker"] = "SPY"
        
        feature_cols = get_feature_columns(df)
        
        assert "fwd_return_1d" not in feature_cols
        assert "fwd_return_5d" not in feature_cols
        assert "signal" not in feature_cols
        assert "ticker" not in feature_cols
    
    def test_derived_features_are_point_in_time(self, sample_feature_df):
        """Derived features should only use current or past data, never future."""
        df = sample_feature_df.copy()
        
        # Modify the last row's close and check that earlier rows are unaffected
        df_original = compute_derived_features(df.copy())
        
        df_modified = df.copy()
        df_modified.iloc[-1, df_modified.columns.get_loc("close")] = 999999
        df_modified_features = compute_derived_features(df_modified)
        
        # All rows except the last should be identical
        for col in ["price_vs_sma50", "volatility_ratio", "bb_position"]:
            if col in df_original.columns and col in df_modified_features.columns:
                orig_vals = df_original[col].iloc[:-1]
                mod_vals = df_modified_features[col].iloc[:-1]
                # Only compare non-NaN values
                mask = orig_vals.notna() & mod_vals.notna()
                if mask.any():
                    assert np.allclose(
                        orig_vals[mask].values,
                        mod_vals[mask].values,
                        equal_nan=True,
                    ), f"Feature {col} changed for historical rows when future data changed"
