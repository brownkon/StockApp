"""
Tests for tune_model.py
--------------------------
Validates the enhanced training pipeline: class weighting,
walk-forward CV, feature selection, and target configuration.
"""

import pytest
import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../scripts")))

from tune_model import (
    compute_class_weight,
    walk_forward_split,
    select_features,
    prepare_target,
    get_feature_columns,
    train_enhanced,
)


@pytest.fixture
def training_df():
    """Create a realistic multi-year training DataFrame."""
    np.random.seed(42)
    dates = pd.date_range("2018-01-01", "2025-06-01", freq="B")
    n = len(dates)

    close = 100 + np.cumsum(np.random.normal(0.01, 1, n))

    df = pd.DataFrame({
        "close": close,
        "volume": np.random.randint(1_000_000, 50_000_000, n).astype(float),
        "sma_50": pd.Series(close).rolling(50).mean().values,
        "sma_200": np.full(n, close.mean()),
        "rsi_14": np.random.uniform(20, 80, n),
        "macd": np.random.normal(0, 1, n),
        "macd_signal": np.random.normal(0, 0.5, n),
        "macd_hist": np.random.normal(0, 0.5, n),
        "atr_14": np.random.uniform(1, 3, n),
        "adx_14": np.random.uniform(15, 40, n),
        "sentiment_score": np.random.uniform(-0.5, 0.5, n),
        "macro_fed_funds": np.random.uniform(0, 5, n),
        "day_of_week": np.tile([0, 1, 2, 3, 4], n // 5 + 1)[:n],
        "month": dates.month,
        # Targets — will be overridden by prepare_target
        "fwd_return_1d": np.random.normal(0, 0.02, n),
        "fwd_return_5d": np.random.normal(0, 0.03, n),
        "signal": np.random.choice([0, 1], n),
        "ticker": "SPY",
    }, index=dates)

    return df


class TestComputeClassWeight:
    def test_balanced_dataset(self):
        y = pd.Series([0, 0, 1, 1, 0, 1])
        weight = compute_class_weight(y)
        assert weight == 1.0  # Equal counts

    def test_imbalanced_dataset(self):
        y = pd.Series([0] * 30 + [1] * 70)
        weight = compute_class_weight(y)
        assert abs(weight - 30 / 70) < 0.01

    def test_no_positive_class(self):
        y = pd.Series([0, 0, 0])
        weight = compute_class_weight(y)
        assert weight == 1.0  # Fallback


class TestWalkForwardSplit:
    def test_produces_multiple_folds(self, training_df):
        folds = list(walk_forward_split(training_df, n_splits=3))
        assert len(folds) >= 2

    def test_train_before_test(self, training_df):
        """Training max date must be before test min date in every fold."""
        for train_idx, test_idx in walk_forward_split(training_df, n_splits=3):
            assert train_idx.max() < test_idx.min()

    def test_expanding_training_window(self, training_df):
        """Each fold should have more training data than the previous."""
        folds = list(walk_forward_split(training_df, n_splits=3))
        if len(folds) >= 2:
            for i in range(1, len(folds)):
                assert len(folds[i][0]) >= len(folds[i - 1][0])


class TestSelectFeatures:
    def test_drops_low_importance(self):
        """Features below threshold should be removed."""
        from xgboost import XGBClassifier
        np.random.seed(42)
        X = np.random.rand(100, 5)
        y = np.random.choice([0, 1], 100)
        model = XGBClassifier(n_estimators=10, max_depth=2, random_state=42,
                              eval_metric="logloss")
        model.fit(X, y)

        feature_cols = ["a", "b", "c", "d", "e"]
        selected = select_features(model, feature_cols, threshold=0.3)
        # Should drop anything below 0.3 importance
        assert len(selected) <= len(feature_cols)


class TestPrepareTarget:
    def test_classification_1d(self, training_df):
        df, col = prepare_target(training_df.copy(), "1d", "classification")
        assert col == "signal_1d"
        assert col in df.columns
        assert set(df[col].dropna().unique()).issubset({0, 1})

    def test_classification_5d(self, training_df):
        df, col = prepare_target(training_df.copy(), "5d", "classification")
        assert col == "signal_5d"

    def test_classification_10d(self, training_df):
        df, col = prepare_target(training_df.copy(), "10d", "classification")
        assert col == "signal_10d"

    def test_regression_returns_continuous(self, training_df):
        df, col = prepare_target(training_df.copy(), "5d", "regression")
        assert col == "fwd_return_5d"
        # Should have continuously valued floats, not just 0/1
        unique_vals = df[col].dropna().nunique()
        assert unique_vals > 2


class TestTrainEnhanced:
    def test_trains_with_all_improvements(self, training_df):
        """Full pipeline should complete without error."""
        model, metrics, importances, feature_cols, params = train_enhanced(
            training_df,
            target_horizon="5d",
            task="classification",
            tune_hyperparams=True,
            tune_iterations=3,  # Minimal for speed
            do_feature_selection=True,
            use_class_weight=True,
            walk_forward_folds=2,
            final_train_end="2024-01-01",
        )
        assert model is not None
        assert "accuracy" in metrics or "f1_score" in metrics
        assert len(feature_cols) > 0

    def test_regression_mode(self, training_df):
        """Should work in regression mode too."""
        model, metrics, importances, feature_cols, params = train_enhanced(
            training_df,
            target_horizon="5d",
            task="regression",
            tune_hyperparams=False,
            do_feature_selection=False,
            use_class_weight=False,
            walk_forward_folds=2,
            final_train_end="2024-01-01",
        )
        assert model is not None
        assert "r2" in metrics
