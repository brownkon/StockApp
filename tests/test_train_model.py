"""
Tests for train_model.py
--------------------------
Validates the model training pipeline including temporal splitting,
feature column selection, and model serialization.
"""

import pytest
import sys
import os
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../scripts")))

from train_model import get_feature_columns, temporal_split, train_model, save_model


@pytest.fixture
def training_df():
    """Create a realistic training DataFrame with features and targets."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2025-01-01", freq="B")
    n = len(dates)
    
    df = pd.DataFrame({
        "close": 100 + np.cumsum(np.random.normal(0.01, 1, n)),
        "volume": np.random.randint(1_000_000, 50_000_000, n).astype(float),
        "sma_50": np.random.normal(100, 10, n),
        "sma_200": np.random.normal(100, 5, n),
        "rsi_14": np.random.uniform(20, 80, n),
        "macd": np.random.normal(0, 1, n),
        "atr_14": np.random.uniform(1, 3, n),
        "adx_14": np.random.uniform(15, 40, n),
        "sentiment_score": np.random.uniform(-0.5, 0.5, n),
        "macro_fed_funds": np.random.uniform(0, 5, n),
        "day_of_week": np.tile([0, 1, 2, 3, 4], n // 5 + 1)[:n],
        # Targets
        "fwd_return_1d": np.random.normal(0, 0.02, n),
        "fwd_return_5d": np.random.normal(0, 0.03, n),
        "signal": np.random.choice([0, 1], n),
        "ticker": "SPY",
    }, index=dates)
    
    return df


class TestGetFeatureColumns:
    def test_excludes_targets(self, training_df):
        feature_cols = get_feature_columns(training_df)
        assert "fwd_return_1d" not in feature_cols
        assert "fwd_return_5d" not in feature_cols
        assert "signal" not in feature_cols
    
    def test_excludes_metadata(self, training_df):
        feature_cols = get_feature_columns(training_df)
        assert "ticker" not in feature_cols
    
    def test_includes_actual_features(self, training_df):
        feature_cols = get_feature_columns(training_df)
        assert "close" in feature_cols
        assert "rsi_14" in feature_cols
        assert "sentiment_score" in feature_cols


class TestTemporalSplit:
    def test_no_data_leakage_across_split(self, training_df):
        """Training data must be strictly before the cutoff date."""
        train_df, val_df, test_df = temporal_split(training_df, "2024-01-01")
        
        cutoff = pd.Timestamp("2024-01-01")
        assert train_df.index.max() < cutoff
        if not val_df.empty:
            assert val_df.index.max() < cutoff
        assert test_df.index.min() >= cutoff
    
    def test_split_preserves_all_data(self, training_df):
        """All rows should end up in exactly one split."""
        train_df, val_df, test_df = temporal_split(training_df, "2024-01-01")
        total = len(train_df) + len(val_df) + len(test_df)
        assert total == len(training_df)
    
    def test_splits_are_not_empty(self, training_df):
        train_df, val_df, test_df = temporal_split(training_df, "2024-01-01")
        assert len(train_df) > 0
        assert len(test_df) > 0
    
    def test_temporal_order_maintained(self, training_df):
        """Within each split, data should remain in chronological order."""
        train_df, val_df, test_df = temporal_split(training_df, "2024-01-01")
        
        assert (train_df.index == train_df.index.sort_values()).all()
        if not val_df.empty:
            assert (val_df.index == val_df.index.sort_values()).all()
        assert (test_df.index == test_df.index.sort_values()).all()


class TestTrainModel:
    def test_model_trains_successfully(self, training_df):
        """Model should train without errors on valid data."""
        model, metrics, importances = train_model(
            training_df,
            train_end_date="2024-01-01",
            model_params={
                "n_estimators": 10,  # Small for speed
                "max_depth": 3,
                "learning_rate": 0.1,
                "eval_metric": "logloss",
                "random_state": 42,
            },
            early_stopping_rounds=None,
        )
        assert model is not None
        assert "accuracy" in metrics
        assert "roc_auc" in metrics
    
    def test_metrics_are_valid(self, training_df):
        model, metrics, _ = train_model(
            training_df,
            train_end_date="2024-01-01",
            model_params={
                "n_estimators": 10,
                "max_depth": 3,
                "learning_rate": 0.1,
                "eval_metric": "logloss",
                "random_state": 42,
            },
            early_stopping_rounds=None,
        )
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
    
    def test_feature_importances_match_features(self, training_df):
        model, _, importances = train_model(
            training_df,
            train_end_date="2024-01-01",
            model_params={
                "n_estimators": 10,
                "max_depth": 3,
                "learning_rate": 0.1,
                "eval_metric": "logloss",
                "random_state": 42,
            },
            early_stopping_rounds=None,
        )
        feature_cols = get_feature_columns(training_df)
        assert set(importances.keys()) == set(feature_cols)


class TestSaveModel:
    def test_model_serialization(self, training_df):
        """Model should be saveable and loadable via joblib."""
        import joblib
        
        model, metrics, importances = train_model(
            training_df,
            train_end_date="2024-01-01",
            model_params={
                "n_estimators": 10,
                "max_depth": 3,
                "learning_rate": 0.1,
                "eval_metric": "logloss",
                "random_state": 42,
            },
            early_stopping_rounds=None,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Temporarily override MODELS_DIR
            import train_model as tm
            original_dir = tm.MODELS_DIR
            tm.MODELS_DIR = tmpdir
            
            try:
                version = save_model(model, metrics, importances)
                assert version is not None
                
                # Verify files exist
                assert os.path.exists(os.path.join(tmpdir, "xgb_model.pkl"))
                assert os.path.exists(os.path.join(tmpdir, "model_metadata.json"))
                
                # Verify model can be loaded
                loaded = joblib.load(os.path.join(tmpdir, "xgb_model.pkl"))
                assert loaded is not None
            finally:
                tm.MODELS_DIR = original_dir
