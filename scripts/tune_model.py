"""
Advanced Model Tuning & Training
----------------------------------
Implements 6 key improvements over the baseline trainer:

1. Hyperparameter tuning (RandomizedSearch)
2. Feature selection (drop low-importance features)
3. Class weighting (scale_pos_weight)
4. Walk-forward cross-validation (rolling temporal windows)
5. Configurable target variable (1d, 5d, 10d, regression)
6. Ticker-specific models (per-ticker or pooled)
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
    mean_squared_error,
    r2_score,
)
from xgboost import XGBClassifier, XGBRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

TARGET_COLS = ["fwd_return_1d", "fwd_return_5d", "fwd_return_10d", "signal",
               "signal_1d", "signal_5d", "signal_10d"]
META_COLS = ["ticker"]


# --------------------------------------------------------------------------- #
#  1. Hyperparameter search space
# --------------------------------------------------------------------------- #
PARAM_GRID = {
    "n_estimators": [200, 500, 800, 1200],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.6, 0.7, 0.8, 0.9],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0, 0.1, 0.3, 0.5],
    "reg_alpha": [0, 0.01, 0.1, 1.0],
    "reg_lambda": [1.0, 1.5, 2.0, 5.0],
}


def get_feature_columns(df):
    """Return feature columns by excluding targets and metadata."""
    exclude = set(TARGET_COLS + META_COLS)
    return [c for c in df.columns if c not in exclude]


# --------------------------------------------------------------------------- #
#  5. Configurable target variable
# --------------------------------------------------------------------------- #
def prepare_target(df, target_horizon="5d", task="classification"):
    """Prepare target variable based on horizon and task type.

    Args:
        df: DataFrame with 'close' column.
        target_horizon: '1d', '5d', or '10d'.
        task: 'classification' or 'regression'.

    Returns:
        DataFrame with the appropriate target column added.
    """
    shift_map = {"1d": 1, "5d": 5, "10d": 10}
    shift = shift_map.get(target_horizon, 5)

    col_name = f"fwd_return_{target_horizon}"
    if col_name not in df.columns:
        df[col_name] = df["close"].shift(-shift) / df["close"] - 1

    if task == "classification":
        target_col = f"signal_{target_horizon}"
        df[target_col] = (df[col_name] > 0).astype(int)
        return df, target_col
    else:
        return df, col_name


# --------------------------------------------------------------------------- #
#  3. Class weighting
# --------------------------------------------------------------------------- #
def compute_class_weight(y):
    """Compute scale_pos_weight for imbalanced binary classification.

    XGBoost's scale_pos_weight = count(negative) / count(positive).
    """
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    if n_pos == 0:
        return 1.0
    weight = n_neg / n_pos
    logger.info(f"Class balance: {n_neg} neg / {n_pos} pos -> scale_pos_weight={weight:.3f}")
    return weight


# --------------------------------------------------------------------------- #
#  4. Walk-forward cross-validation
# --------------------------------------------------------------------------- #
def walk_forward_split(df, n_splits=5, min_train_years=3, test_months=6):
    """Generate walk-forward temporal splits.

    Yields (train_idx, test_idx) index arrays that respect temporal ordering.
    Each fold trains on expanding history and tests on the next `test_months`.
    """
    df = df.sort_index()
    dates = df.index.unique().sort_values()
    total_days = len(dates)

    min_train_days = min_train_years * 252  # ~trading days per year
    test_days = test_months * 21            # ~trading days per month

    if total_days < min_train_days + test_days:
        logger.warning("Not enough data for walk-forward CV, using single split")
        cutoff = dates[int(len(dates) * 0.8)]
        train_mask = df.index <= cutoff
        test_mask = df.index > cutoff
        yield df.index[train_mask], df.index[test_mask]
        return

    # Calculate available space for test folds
    available = total_days - min_train_days
    step = max(available // n_splits, test_days)

    for i in range(n_splits):
        train_end_idx = min_train_days + i * step
        test_end_idx = min(train_end_idx + test_days, total_days)

        if train_end_idx >= total_days or test_end_idx <= train_end_idx:
            break

        train_end_date = dates[train_end_idx - 1]
        test_start_date = dates[train_end_idx]
        test_end_date = dates[test_end_idx - 1]

        train_mask = df.index <= train_end_date
        test_mask = (df.index >= test_start_date) & (df.index <= test_end_date)

        train_count = train_mask.sum()
        test_count = test_mask.sum()

        if train_count > 0 and test_count > 0:
            logger.info(
                f"Fold {i + 1}: train up to {train_end_date.date()} "
                f"({train_count} rows), test {test_start_date.date()} to "
                f"{test_end_date.date()} ({test_count} rows)"
            )
            yield df.index[train_mask], df.index[test_mask]


# --------------------------------------------------------------------------- #
#  2. Feature selection
# --------------------------------------------------------------------------- #
def select_features(model, feature_cols, threshold=0.01):
    """Remove features with importance below threshold.

    Returns the list of selected feature names.
    """
    importances = dict(zip(feature_cols, model.feature_importances_))
    selected = [f for f, imp in importances.items() if imp >= threshold]
    dropped = [f for f, imp in importances.items() if imp < threshold]

    if dropped:
        logger.info(f"Feature selection: dropping {len(dropped)} low-importance features: {dropped}")
    logger.info(f"Feature selection: keeping {len(selected)} features")

    return selected


# --------------------------------------------------------------------------- #
#  1. Hyperparameter tuning
# --------------------------------------------------------------------------- #
def random_search(X_train, y_train, X_val, y_val, n_iter=30,
                  task="classification", class_weight=1.0):
    """Random search over hyperparameter space.

    Uses validation set performance (not CV) for speed.
    """
    rng = np.random.RandomState(42)
    best_score = -np.inf
    best_params = None
    results = []

    for i in range(n_iter):
        params = {k: rng.choice(v) for k, v in PARAM_GRID.items()}
        params["random_state"] = 42
        params["n_jobs"] = -1
        params["eval_metric"] = "logloss" if task == "classification" else "rmse"

        if task == "classification":
            params["scale_pos_weight"] = class_weight
            model = XGBClassifier(**params)
        else:
            model = XGBRegressor(**params)

        model.set_params(early_stopping_rounds=30)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        if task == "classification":
            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred, zero_division=0)
        else:
            y_pred = model.predict(X_val)
            score = -mean_squared_error(y_val, y_pred)  # Negative so higher=better

        results.append({"params": params.copy(), "score": score})

        if score > best_score:
            best_score = score
            best_params = params.copy()
            logger.info(f"  Trial {i + 1}/{n_iter}: score={score:.4f} (new best)")

    logger.info(f"Best hyperparameters (score={best_score:.4f}): {best_params}")
    return best_params, results


# --------------------------------------------------------------------------- #
#  Core training with all improvements
# --------------------------------------------------------------------------- #
def train_enhanced(
    df,
    target_horizon="5d",
    task="classification",
    tune_hyperparams=True,
    tune_iterations=30,
    do_feature_selection=True,
    feature_importance_threshold=0.01,
    use_class_weight=True,
    walk_forward_folds=5,
    final_train_end="2024-01-01",
):
    """Train with all 6 improvements applied.

    Returns:
        model, metrics, feature_cols, best_params
    """
    # --- 5. Prepare target ---
    df, target_col = prepare_target(df.copy(), target_horizon, task)
    df_clean = df.dropna(subset=[target_col])

    feature_cols = get_feature_columns(df_clean)
    logger.info(f"Initial feature count: {len(feature_cols)}")
    logger.info(f"Target: {target_col} ({task}), horizon: {target_horizon}")

    # --- 4. Walk-forward CV for robust evaluation ---
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 1: Walk-Forward Cross-Validation")
    logger.info(f"{'='*60}")

    cv_scores = []
    for fold_idx, (train_idx, test_idx) in enumerate(
        walk_forward_split(df_clean, n_splits=walk_forward_folds)
    ):
        X_tr = df_clean.loc[train_idx, feature_cols]
        y_tr = df_clean.loc[train_idx, target_col]
        X_te = df_clean.loc[test_idx, feature_cols]
        y_te = df_clean.loc[test_idx, target_col]

        if task == "classification":
            model = XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                eval_metric="logloss", random_state=42, n_jobs=-1,
            )
            model.fit(X_tr, y_tr, verbose=False)
            y_pred = model.predict(X_te)
            fold_score = f1_score(y_te, y_pred, zero_division=0)
            fold_acc = accuracy_score(y_te, y_pred)
            cv_scores.append({"fold": fold_idx + 1, "f1": fold_score, "accuracy": fold_acc})
            logger.info(f"  Fold {fold_idx + 1}: F1={fold_score:.4f}, Acc={fold_acc:.4f}")
        else:
            model = XGBRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                eval_metric="rmse", random_state=42, n_jobs=-1,
            )
            model.fit(X_tr, y_tr, verbose=False)
            y_pred = model.predict(X_te)
            fold_score = -mean_squared_error(y_te, y_pred)
            r2 = r2_score(y_te, y_pred)
            cv_scores.append({"fold": fold_idx + 1, "neg_mse": fold_score, "r2": r2})
            logger.info(f"  Fold {fold_idx + 1}: R²={r2:.4f}, MSE={-fold_score:.6f}")

    if cv_scores and task == "classification":
        avg_f1 = np.mean([s["f1"] for s in cv_scores])
        avg_acc = np.mean([s["accuracy"] for s in cv_scores])
        logger.info(f"Walk-Forward CV average: F1={avg_f1:.4f}, Acc={avg_acc:.4f}")

    # --- Final train/test split for model selection ---
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 2: Final Model Training & Tuning")
    logger.info(f"{'='*60}")

    cutoff = pd.Timestamp(final_train_end)
    train_full = df_clean[df_clean.index < cutoff]
    test_df = df_clean[df_clean.index >= cutoff]

    # Validation = last 20% of training
    val_size = int(len(train_full) * 0.2)
    train_df = train_full.iloc[:-val_size] if val_size > 0 else train_full
    val_df = train_full.iloc[-val_size:] if val_size > 0 else pd.DataFrame()

    logger.info(f"Final split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # --- 3. Class weighting ---
    class_weight = 1.0
    if use_class_weight and task == "classification":
        class_weight = compute_class_weight(y_train)

    # --- 1. Hyperparameter tuning ---
    if tune_hyperparams and not val_df.empty:
        logger.info(f"\nRunning hyperparameter search ({tune_iterations} trials)...")
        best_params, search_results = random_search(
            X_train, y_train, X_val, y_val,
            n_iter=tune_iterations, task=task, class_weight=class_weight,
        )
    else:
        best_params = {
            "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "eval_metric": "logloss" if task == "classification" else "rmse",
            "random_state": 42, "n_jobs": -1,
        }
        if task == "classification":
            best_params["scale_pos_weight"] = class_weight

    # --- Train initial model for feature selection ---
    if task == "classification":
        model = XGBClassifier(**best_params)
    else:
        model = XGBRegressor(**best_params)

    if not val_df.empty:
        model.set_params(early_stopping_rounds=50)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )
    else:
        model.fit(X_train, y_train, verbose=50)

    # --- 2. Feature selection ---
    if do_feature_selection:
        logger.info(f"\n{'='*60}")
        logger.info("PHASE 3: Feature Selection")
        logger.info(f"{'='*60}")

        selected_features = select_features(model, feature_cols, feature_importance_threshold)

        if len(selected_features) < len(feature_cols):
            feature_cols = selected_features
            X_train = train_df[feature_cols]
            X_val = val_df[feature_cols] if not val_df.empty else pd.DataFrame()
            X_test = test_df[feature_cols]

            # Retrain with selected features
            logger.info("Retraining with selected features...")
            if task == "classification":
                model = XGBClassifier(**best_params)
            else:
                model = XGBRegressor(**best_params)

            if not val_df.empty:
                model.set_params(early_stopping_rounds=50)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=50,
                )
            else:
                model.fit(X_train, y_train, verbose=50)

    # --- Evaluate on test set ---
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 4: Test Set Evaluation")
    logger.info(f"{'='*60}")

    if task == "classification":
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
            "cv_scores": cv_scores,
        }

        logger.info(f"Test Accuracy:  {acc:.4f}")
        logger.info(f"Test Precision: {prec:.4f}")
        logger.info(f"Test Recall:    {rec:.4f}")
        logger.info(f"Test F1:        {f1:.4f}")
        logger.info(f"Test ROC-AUC:   {roc:.4f}")
        logger.info(f"Confusion Matrix:\n{cm}")
        logger.info(classification_report(y_test, y_pred, zero_division=0))
    else:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            "mse": round(mse, 6),
            "rmse": round(np.sqrt(mse), 6),
            "r2": round(r2, 4),
            "cv_scores": cv_scores,
        }

        logger.info(f"Test MSE:  {mse:.6f}")
        logger.info(f"Test RMSE: {np.sqrt(mse):.6f}")
        logger.info(f"Test R²:   {r2:.4f}")

    # Feature importances
    importances = dict(zip(feature_cols, model.feature_importances_.tolist()))
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    logger.info("\nTop 15 feature importances:")
    for name, imp in sorted_imp[:15]:
        logger.info(f"  {name}: {imp:.4f}")

    metrics.update({
        "target_horizon": target_horizon,
        "task": task,
        "n_features": len(feature_cols),
        "train_size": len(train_df),
        "test_size": len(test_df),
        "best_params": {k: (int(v) if isinstance(v, (np.integer,)) else
                            float(v) if isinstance(v, (np.floating,)) else v)
                        for k, v in best_params.items()},
    })

    return model, metrics, importances, feature_cols, best_params


# --------------------------------------------------------------------------- #
#  6. Ticker-specific models
# --------------------------------------------------------------------------- #
def train_ticker_models(
    df,
    target_horizon="5d",
    task="classification",
    tune_iterations=15,
    final_train_end="2024-01-01",
):
    """Train a separate model per ticker.

    Returns dict of {ticker: (model, metrics)}.
    """
    tickers = sorted(df["ticker"].unique())
    logger.info(f"Training individual models for {len(tickers)} tickers")

    ticker_models = {}

    for ticker in tickers:
        logger.info(f"\n{'#'*60}")
        logger.info(f"  TICKER: {ticker}")
        logger.info(f"{'#'*60}")

        ticker_df = df[df["ticker"] == ticker].copy()
        if len(ticker_df) < 500:
            logger.warning(f"Skipping {ticker}: only {len(ticker_df)} rows (need 500+)")
            continue

        try:
            model, metrics, importances, feature_cols, params = train_enhanced(
                ticker_df,
                target_horizon=target_horizon,
                task=task,
                tune_hyperparams=True,
                tune_iterations=tune_iterations,
                do_feature_selection=True,
                use_class_weight=True,
                walk_forward_folds=3,  # Fewer folds per ticker
                final_train_end=final_train_end,
            )
            ticker_models[ticker] = {
                "model": model,
                "metrics": metrics,
                "feature_cols": feature_cols,
            }
            logger.info(f"{ticker} done: {metrics.get('f1_score', metrics.get('r2', 'N/A'))}")
        except Exception as e:
            logger.error(f"Failed to train {ticker}: {e}")

    return ticker_models


# --------------------------------------------------------------------------- #
#  Save helpers
# --------------------------------------------------------------------------- #
def save_enhanced_model(model, metrics, importances, feature_cols, mode="pooled",
                        ticker=None, model_version=None):
    """Serialize model and metadata."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    if model_version is None:
        model_version = datetime.now().strftime("v%Y%m%d_%H%M%S")

    if mode == "pooled":
        model_path = os.path.join(MODELS_DIR, "xgb_model.pkl")
        meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    else:
        ticker_dir = os.path.join(MODELS_DIR, "ticker_models")
        os.makedirs(ticker_dir, exist_ok=True)
        model_path = os.path.join(ticker_dir, f"xgb_{ticker}.pkl")
        meta_path = os.path.join(ticker_dir, f"meta_{ticker}.json")

    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    meta = {
        "model_version": model_version,
        "trained_at": datetime.now().isoformat(),
        "mode": mode,
        "ticker": ticker,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "feature_importances": importances,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info(f"Metadata saved to {meta_path}")

    return model_version


# --------------------------------------------------------------------------- #
#  Main entry point
# --------------------------------------------------------------------------- #
def main(
    target_horizon="5d",
    task="classification",
    mode="pooled",
    tune_iterations=30,
    final_train_end="2024-01-01",
    parquet_path=None,
):
    """Run the full enhanced training pipeline."""
    if parquet_path is None:
        parquet_path = os.path.join(DATA_DIR, "all_features.parquet")

    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df)} rows from {parquet_path}")

    if mode == "pooled":
        model, metrics, importances, feature_cols, params = train_enhanced(
            df,
            target_horizon=target_horizon,
            task=task,
            tune_hyperparams=True,
            tune_iterations=tune_iterations,
            final_train_end=final_train_end,
        )
        version = save_enhanced_model(model, metrics, importances, feature_cols, mode="pooled")
        logger.info(f"\nPooled model training complete. Version: {version}")

    elif mode == "per-ticker":
        ticker_models = train_ticker_models(
            df,
            target_horizon=target_horizon,
            task=task,
            tune_iterations=max(tune_iterations // 3, 10),
            final_train_end=final_train_end,
        )
        for ticker, data in ticker_models.items():
            save_enhanced_model(
                data["model"], data["metrics"], {},
                data["feature_cols"], mode="ticker", ticker=ticker,
            )
        logger.info(f"\nTicker-specific training complete for {len(ticker_models)} tickers.")

    elif mode == "both":
        # Train pooled first
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING POOLED MODEL")
        logger.info("=" * 60)
        model, metrics, importances, feature_cols, params = train_enhanced(
            df,
            target_horizon=target_horizon,
            task=task,
            tune_hyperparams=True,
            tune_iterations=tune_iterations,
            final_train_end=final_train_end,
        )
        save_enhanced_model(model, metrics, importances, feature_cols, mode="pooled")

        # Then per-ticker
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING PER-TICKER MODELS")
        logger.info("=" * 60)
        ticker_models = train_ticker_models(
            df,
            target_horizon=target_horizon,
            task=task,
            tune_iterations=max(tune_iterations // 3, 10),
            final_train_end=final_train_end,
        )
        for ticker, data in ticker_models.items():
            save_enhanced_model(
                data["model"], data["metrics"], {},
                data["feature_cols"], mode="ticker", ticker=ticker,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced XGBoost training pipeline")
    parser.add_argument(
        "--horizon", choices=["1d", "5d", "10d"], default="5d",
        help="Target prediction horizon (default: 5d)",
    )
    parser.add_argument(
        "--task", choices=["classification", "regression"], default="classification",
        help="Prediction task type (default: classification)",
    )
    parser.add_argument(
        "--mode", choices=["pooled", "per-ticker", "both"], default="pooled",
        help="Training mode (default: pooled)",
    )
    parser.add_argument(
        "--tune-iterations", type=int, default=30,
        help="Number of hyperparameter search trials (default: 30)",
    )
    parser.add_argument(
        "--train-end-date", default="2024-01-01",
        help="Cutoff date for final train/test split",
    )
    parser.add_argument(
        "--parquet", default=None,
        help="Path to feature parquet file",
    )
    args = parser.parse_args()
    main(
        target_horizon=args.horizon,
        task=args.task,
        mode=args.mode,
        tune_iterations=args.tune_iterations,
        final_train_end=args.train_end_date,
        parquet_path=args.parquet,
    )
