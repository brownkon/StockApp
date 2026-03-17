"""
Signal Prediction Module
-------------------------
Loads a trained XGBoost model and generates trading signals
for the latest available date for each tracked ticker.
"""

import argparse
import logging
import os
from datetime import datetime

import joblib
import json
import pandas as pd
from sqlalchemy.orm import sessionmaker

from db import engine, Prediction
from build_features import build_features_for_ticker, get_feature_columns, DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

# Columns that must never be used as features
TARGET_COLS = ["fwd_return_1d", "fwd_return_5d", "signal"]
META_COLS = ["ticker"]


def load_model(model_path=None):
    """Load serialized model and metadata."""
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "xgb_model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")

    # Load metadata for version info
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    model_version = "unknown"
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        model_version = meta.get("model_version", "unknown")

    return model, model_version


def predict_latest_signals(tickers=None, model_path=None, save_to_db=True):
    """Generate signals for the latest available date per ticker."""
    if not engine:
        logger.error("Database connection not established")
        return pd.DataFrame()

    model, model_version = load_model(model_path)

    Session = sessionmaker(bind=engine)
    session = Session()

    if tickers is None:
        from db import DailyPrice
        result = session.query(DailyPrice.ticker).distinct().all()
        tickers = sorted([r[0] for r in result])
        tickers = [t for t in tickers if not t.startswith("^")]

    predictions = []

    for ticker in tickers:
        try:
            df = build_features_for_ticker(session, ticker)
            if df.empty:
                logger.warning(f"No features for {ticker}, skipping")
                continue

            # Use the model's training feature order to ensure consistency
            if hasattr(model, "feature_names_in_"):
                feature_cols = list(model.feature_names_in_)
            else:
                feature_cols = get_feature_columns(df)
                feature_cols = [c for c in feature_cols if c not in TARGET_COLS and c not in META_COLS]

            # Get the latest row (last available date)
            latest = df.iloc[[-1]].copy()
            latest_date = latest.index[0]

            # Ensure all expected columns exist, fill missing with NaN
            for col in feature_cols:
                if col not in latest.columns:
                    latest[col] = float("nan")

            X = latest[feature_cols]
            pred_signal = model.predict(X)[0]
            pred_proba = model.predict_proba(X)[:, 1][0]

            predictions.append({
                "ticker": ticker,
                "date": latest_date,
                "predicted_signal": int(pred_signal),
                "predicted_probability": round(float(pred_proba), 4),
                "model_version": model_version,
            })

            direction = "BUY" if pred_signal == 1 else "HOLD/SELL"
            logger.info(
                f"{ticker} | {latest_date.date()} | {direction} "
                f"(prob={pred_proba:.4f})"
            )

        except Exception as e:
            logger.error(f"Prediction failed for {ticker}: {e}")

    if not predictions:
        logger.warning("No predictions generated")
        return pd.DataFrame()

    pred_df = pd.DataFrame(predictions)

    # Save to CSV
    csv_path = os.path.join(DATA_DIR, "predictions_latest.csv")
    pred_df.to_csv(csv_path, index=False)
    logger.info(f"Predictions saved to {csv_path}")

    # Save to database
    if save_to_db:
        _save_predictions_to_db(session, pred_df, model_version)

    session.close()
    return pred_df


def _save_predictions_to_db(session, pred_df, model_version):
    """Upsert predictions into the database."""
    now = datetime.utcnow()

    for _, row in pred_df.iterrows():
        existing = (
            session.query(Prediction)
            .filter_by(ticker=row["ticker"], date=row["date"])
            .first()
        )

        if existing:
            existing.predicted_signal = int(row["predicted_signal"])
            existing.predicted_probability = float(row["predicted_probability"])
            existing.model_version = model_version
            existing.created_at = now
        else:
            pred = Prediction(
                ticker=row["ticker"],
                date=row["date"],
                predicted_signal=int(row["predicted_signal"]),
                predicted_probability=float(row["predicted_probability"]),
                model_version=model_version,
                created_at=now,
            )
            session.add(pred)

    try:
        session.commit()
        logger.info(f"Saved {len(pred_df)} predictions to database")
    except Exception as e:
        logger.error(f"Failed to save predictions to DB: {e}")
        session.rollback()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate trading signals")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Specific tickers to predict (default: all)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to serialized model (default: models/xgb_model.pkl)",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Skip saving to database",
    )
    args = parser.parse_args()
    predict_latest_signals(
        tickers=args.tickers,
        model_path=args.model_path,
        save_to_db=not args.no_db,
    )
