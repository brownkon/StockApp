"""
ML Model Training Module
-------------------------
Trains an XGBoost binary classifier to predict whether
the 5-day forward return is positive (signal=1) or negative (signal=0).

Uses walk-forward temporal split — no data shuffling.
"""

import argparse
import json
import logging
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

# Columns that must never be used as features
TARGET_COLS = ["fwd_return_1d", "fwd_return_5d", "signal"]
META_COLS = ["ticker"]

# Default model hyperparameters
DEFAULT_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}


def get_feature_columns(df):
    """Return feature columns by excluding targets and metadata."""
    exclude = set(TARGET_COLS + META_COLS)
    return [c for c in df.columns if c not in exclude]


def load_feature_data(parquet_path=None):
    """Load the combined feature DataFrame from parquet."""
    if parquet_path is None:
        parquet_path = os.path.join(DATA_DIR, "all_features.parquet")

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Feature file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df)} rows from {parquet_path}")
    return df


def temporal_split(df, train_end_date="2024-01-01", val_fraction=0.2):
    """Split data by date. No shuffling — respects temporal ordering.
    
    Returns:
        train_df, val_df, test_df
    """
    train_end = pd.Timestamp(train_end_date)
    df.index = pd.to_datetime(df.index)

    train_full = df[df.index < train_end]
    test_df = df[df.index >= train_end]

    # Carve validation from end of training window
    val_size = int(len(train_full) * val_fraction)
    train_df = train_full.iloc[:-val_size] if val_size > 0 else train_full
    val_df = train_full.iloc[-val_size:] if val_size > 0 else pd.DataFrame()

    logger.info(
        f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    return train_df, val_df, test_df


def train_model(
    df,
    train_end_date="2024-01-01",
    model_params=None,
    early_stopping_rounds=50,
):
    """Train XGBoost classifier and evaluate on test set.
    
    Returns:
        model, metrics_dict, feature_importances
    """
    if model_params is None:
        model_params = DEFAULT_PARAMS.copy()

    feature_cols = get_feature_columns(df)
    logger.info(f"Using {len(feature_cols)} features: {feature_cols}")

    # Drop rows missing the target
    df_clean = df.dropna(subset=["signal"])

    # Drop rows missing all features (keep rows with some NaN — XGBoost handles them)
    df_clean = df_clean.dropna(subset=feature_cols, how="all")

    train_df, val_df, test_df = temporal_split(df_clean, train_end_date)

    if train_df.empty:
        raise ValueError("Training set is empty after split")
    if test_df.empty:
        raise ValueError("Test set is empty after split")

    X_train = train_df[feature_cols]
    y_train = train_df["signal"]
    X_test = test_df[feature_cols]
    y_test = test_df["signal"]

    # Build model
    model = XGBClassifier(**model_params)

    # Fit with early stopping if we have a validation set
    fit_params = {}
    if not val_df.empty:
        X_val = val_df[feature_cols]
        y_val = val_df["signal"]
        fit_params["eval_set"] = [(X_val, y_val)]
        if early_stopping_rounds:
            model.set_params(early_stopping_rounds=early_stopping_rounds)

    logger.info("Training XGBoost classifier...")
    model.fit(X_train, y_train, verbose=50, **fit_params)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        roc = roc_auc_score(y_test, y_proba)
    except ValueError:
        roc = float("nan")
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc, 4),
        "confusion_matrix": cm.tolist(),
        "train_size": len(train_df),
        "test_size": len(test_df),
        "train_end_date": train_end_date,
        "n_features": len(feature_cols),
    }

    logger.info(f"Test Accuracy:  {acc:.4f}")
    logger.info(f"Test Precision: {prec:.4f}")
    logger.info(f"Test Recall:    {rec:.4f}")
    logger.info(f"Test F1:        {f1:.4f}")
    logger.info(f"Test ROC-AUC:   {roc:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(classification_report(y_test, y_pred, zero_division=0))

    # Feature importances
    importances = dict(zip(feature_cols, model.feature_importances_.tolist()))
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    logger.info("Top 15 feature importances:")
    for name, imp in sorted_imp[:15]:
        logger.info(f"  {name}: {imp:.4f}")

    return model, metrics, importances


def save_model(model, metrics, importances, model_version=None):
    """Serialize the model and metadata to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    if model_version is None:
        model_version = datetime.now().strftime("v%Y%m%d_%H%M%S")

    model_path = os.path.join(MODELS_DIR, "xgb_model.pkl")
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    # Save metrics
    meta = {
        "model_version": model_version,
        "trained_at": datetime.now().isoformat(),
        "metrics": metrics,
        "feature_importances": importances,
    }
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Metadata saved to {meta_path}")

    return model_version


def main(train_end_date="2024-01-01", parquet_path=None):
    """Main training entry point."""
    df = load_feature_data(parquet_path)
    model, metrics, importances = train_model(df, train_end_date=train_end_date)
    version = save_model(model, metrics, importances)
    logger.info(f"Training complete. Model version: {version}")
    return model, metrics, importances


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost trading signal model")
    parser.add_argument(
        "--train-end-date",
        default="2024-01-01",
        help="Cutoff date for training (test starts after this)",
    )
    parser.add_argument(
        "--parquet",
        default=None,
        help="Path to feature parquet file (default: data/all_features.parquet)",
    )
    args = parser.parse_args()
    main(train_end_date=args.train_end_date, parquet_path=args.parquet)
