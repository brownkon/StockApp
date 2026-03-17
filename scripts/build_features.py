"""
Feature Engineering Module
--------------------------
Assembles a unified, point-in-time feature matrix for ML model training
by joining daily prices, technical indicators, macro indicators, sentiment
scores, and options flow data.

No future data is used — all features are strictly lagged or same-day.
"""

import argparse
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy.orm import sessionmaker

from db import (
    engine,
    DailyPrice,
    TechnicalIndicator,
    MacroIndicator,
    DailySentiment,
    DailyOptionsData,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Key macro indicators to include as features
MACRO_SERIES = [
    "Effective Federal Funds Rate",
    "Consumer Price Index (Inflation)",
    "10-Year Treasury CMR",
    "Unemployment Rate",
    "University of Michigan: Consumer Sentiment",
]

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def _load_prices(session, ticker):
    """Load daily prices for a single ticker into a DataFrame."""
    rows = (
        session.query(DailyPrice)
        .filter(DailyPrice.ticker == ticker)
        .order_by(DailyPrice.date.asc())
        .all()
    )
    if not rows:
        return pd.DataFrame()

    records = [
        {
            "date": r.date,
            "close": r.close,
            "volume": r.volume,
        }
        for r in rows
    ]
    df = pd.DataFrame(records).set_index("date")
    return df


def _load_technicals(session, ticker):
    """Load technical indicators for a single ticker."""
    rows = (
        session.query(TechnicalIndicator)
        .filter(TechnicalIndicator.ticker == ticker)
        .order_by(TechnicalIndicator.date.asc())
        .all()
    )
    if not rows:
        return pd.DataFrame()

    records = [
        {
            "date": r.date,
            "sma_50": r.sma_50,
            "sma_200": r.sma_200,
            "ema_20": r.ema_20,
            "rsi_14": r.rsi_14,
            "macd": r.macd,
            "macd_signal": r.macd_signal,
            "macd_hist": r.macd_hist,
            "bb_upper": r.bb_upper,
            "bb_lower": r.bb_lower,
            "bb_mid": r.bb_mid,
            "atr_14": r.atr_14,
            "adx_14": r.adx_14,
            "plus_di_14": r.plus_di_14,
            "minus_di_14": r.minus_di_14,
            "obv": r.obv,
        }
        for r in rows
    ]
    return pd.DataFrame(records).set_index("date")


def _load_macro(session):
    """Load and pivot macro indicators into a date-indexed DataFrame."""
    rows = session.query(MacroIndicator).all()
    if not rows:
        return pd.DataFrame()

    records = [
        {"date": r.date, "indicator_name": r.indicator_name, "value": r.value}
        for r in rows
    ]
    df = pd.DataFrame(records)
    # Filter to only key series
    df = df[df["indicator_name"].isin(MACRO_SERIES)]
    # Pivot: each indicator becomes a column
    pivot = df.pivot_table(index="date", columns="indicator_name", values="value")
    # Rename columns for cleaner feature names
    rename_map = {
        "Effective Federal Funds Rate": "macro_fed_funds",
        "Consumer Price Index (Inflation)": "macro_cpi",
        "10-Year Treasury CMR": "macro_10y_treasury",
        "Unemployment Rate": "macro_unemployment",
        "University of Michigan: Consumer Sentiment": "macro_consumer_sentiment",
    }
    pivot = pivot.rename(columns=rename_map)
    pivot.index = pd.to_datetime(pivot.index)
    return pivot.sort_index()


def _load_sentiment(session, ticker):
    """Load daily sentiment for a specific ticker (or MARKET fallback)."""
    rows = (
        session.query(DailySentiment)
        .filter(DailySentiment.ticker.in_([ticker, "MARKET"]))
        .order_by(DailySentiment.date.asc())
        .all()
    )
    if not rows:
        return pd.DataFrame()

    records = [
        {
            "date": r.date,
            "ticker_sent": r.ticker,
            "unified_score": r.unified_score,
            "article_count": r.article_count,
        }
        for r in rows
    ]
    df = pd.DataFrame(records)

    # Prefer ticker-specific sentiment; fall back to MARKET
    ticker_df = df[df["ticker_sent"] == ticker]
    market_df = df[df["ticker_sent"] == "MARKET"]

    if not ticker_df.empty:
        result = ticker_df.set_index("date")[["unified_score", "article_count"]]
    else:
        result = market_df.set_index("date")[["unified_score", "article_count"]]

    result = result.rename(columns={
        "unified_score": "sentiment_score",
        "article_count": "sentiment_articles",
    })
    return result


def _load_options(session, ticker):
    """Load options flow data for a ticker."""
    rows = (
        session.query(DailyOptionsData)
        .filter(DailyOptionsData.ticker == ticker)
        .order_by(DailyOptionsData.date.asc())
        .all()
    )
    if not rows:
        return pd.DataFrame()

    records = [
        {
            "date": r.date,
            "put_call_ratio": r.put_call_ratio,
            "implied_volatility": r.implied_volatility,
        }
        for r in rows
    ]
    return pd.DataFrame(records).set_index("date")


def compute_derived_features(df):
    """Compute derived features from the merged DataFrame. All point-in-time."""

    # Distance from moving averages (relative)
    df["price_vs_sma50"] = np.where(
        df["sma_50"].notna() & (df["sma_50"] != 0),
        (df["close"] - df["sma_50"]) / df["sma_50"],
        np.nan,
    )
    df["price_vs_sma200"] = np.where(
        df["sma_200"].notna() & (df["sma_200"] != 0),
        (df["close"] - df["sma_200"]) / df["sma_200"],
        np.nan,
    )

    # Bollinger position (0 = at lower band, 1 = at upper band)
    bb_range = df["bb_upper"] - df["bb_lower"]
    df["bb_position"] = np.where(
        bb_range.notna() & (bb_range != 0),
        (df["close"] - df["bb_lower"]) / bb_range,
        np.nan,
    )

    # Normalised volatility
    df["volatility_ratio"] = np.where(
        df["close"].notna() & (df["close"] != 0),
        df["atr_14"] / df["close"],
        np.nan,
    )

    # Relative volume (vs 20-day average)
    vol_sma20 = df["volume"].rolling(window=20).mean()
    df["volume_sma_ratio"] = np.where(
        vol_sma20.notna() & (vol_sma20 != 0),
        df["volume"] / vol_sma20,
        np.nan,
    )

    # Returns
    df["return_1d"] = df["close"].pct_change(1)
    df["return_5d"] = df["close"].pct_change(5)
    df["return_20d"] = df["close"].pct_change(20)

    # Calendar features
    if hasattr(df.index, 'dayofweek'):
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month
    else:
        df["day_of_week"] = pd.to_datetime(df.index).dayofweek
        df["month"] = pd.to_datetime(df.index).month

    return df


def compute_targets(df):
    """Compute forward-looking target variables (labels). These use future data
    and must ONLY be used as labels — never as features."""

    # 1-day, 5-day, and 10-day forward returns
    df["fwd_return_1d"] = df["close"].shift(-1) / df["close"] - 1
    df["fwd_return_5d"] = df["close"].shift(-5) / df["close"] - 1
    df["fwd_return_10d"] = df["close"].shift(-10) / df["close"] - 1

    # Binary signal: 1 if 5-day forward return is positive
    df["signal"] = (df["fwd_return_5d"] > 0).astype(int)

    return df


def build_features_for_ticker(session, ticker):
    """Build the complete feature DataFrame for a single ticker."""
    logger.info(f"Building features for {ticker}")

    # Load all data sources
    prices_df = _load_prices(session, ticker)
    if prices_df.empty:
        logger.warning(f"No price data for {ticker}, skipping")
        return pd.DataFrame()

    tech_df = _load_technicals(session, ticker)
    macro_df = _load_macro(session)
    sentiment_df = _load_sentiment(session, ticker)
    options_df = _load_options(session, ticker)

    # Ensure date indices are datetime
    prices_df.index = pd.to_datetime(prices_df.index)
    if not tech_df.empty:
        tech_df.index = pd.to_datetime(tech_df.index)
    if not sentiment_df.empty:
        sentiment_df.index = pd.to_datetime(sentiment_df.index)
    if not options_df.empty:
        options_df.index = pd.to_datetime(options_df.index)

    # Start with prices, join technicals (1:1 match expected)
    df = prices_df.copy()
    if not tech_df.empty:
        df = df.join(tech_df, how="left")

    # Join macro (forward-fill since it's lower frequency)
    if not macro_df.empty:
        df = df.join(macro_df, how="left")
        macro_cols = [c for c in df.columns if c.startswith("macro_")]
        df[macro_cols] = df[macro_cols].ffill()

    # Join sentiment (forward-fill)
    if not sentiment_df.empty:
        df = df.join(sentiment_df, how="left")
        sent_cols = ["sentiment_score", "sentiment_articles"]
        existing_sent_cols = [c for c in sent_cols if c in df.columns]
        if existing_sent_cols:
            df[existing_sent_cols] = df[existing_sent_cols].ffill()

    # Join options (forward-fill)
    if not options_df.empty:
        df = df.join(options_df, how="left")
        opt_cols = ["put_call_ratio", "implied_volatility"]
        existing_opt_cols = [c for c in opt_cols if c in df.columns]
        if existing_opt_cols:
            df[existing_opt_cols] = df[existing_opt_cols].ffill()

    # Compute derived features
    df = compute_derived_features(df)

    # Compute targets
    df = compute_targets(df)

    # Add ticker column
    df["ticker"] = ticker

    return df


def get_feature_columns(df):
    """Return the list of columns that are features (not targets or metadata)."""
    target_cols = {"fwd_return_1d", "fwd_return_5d", "fwd_return_10d", "signal",
                   "signal_1d", "signal_5d", "signal_10d"}
    meta_cols = {"ticker"}
    return [c for c in df.columns if c not in target_cols and c not in meta_cols]


def build_all_features(tickers=None):
    """Build features for all tickers and save to parquet files."""
    if not engine:
        logger.error("Database connection not established")
        return

    Session = sessionmaker(bind=engine)
    session = Session()

    if tickers is None:
        # Get all tickers from daily_prices
        result = session.query(DailyPrice.ticker).distinct().all()
        tickers = sorted([r[0] for r in result])
        # Exclude non-tradeable indices
        tickers = [t for t in tickers if not t.startswith("^")]

    logger.info(f"Building features for {len(tickers)} tickers: {tickers}")

    os.makedirs(DATA_DIR, exist_ok=True)
    all_frames = []

    for ticker in tickers:
        try:
            df = build_features_for_ticker(session, ticker)
            if df.empty:
                continue

            # Save per-ticker file
            out_path = os.path.join(DATA_DIR, f"features_{ticker}.parquet")
            df.to_parquet(out_path, engine="pyarrow")
            logger.info(f"Saved {len(df)} rows for {ticker} -> {out_path}")

            all_frames.append(df)
        except Exception as e:
            logger.error(f"Error building features for {ticker}: {e}")

    if all_frames:
        all_df = pd.concat(all_frames, axis=0)
        all_path = os.path.join(DATA_DIR, "all_features.parquet")
        all_df.to_parquet(all_path, engine="pyarrow")
        logger.info(f"Saved combined features ({len(all_df)} rows) -> {all_path}")
    else:
        logger.warning("No feature data was generated")

    session.close()
    return all_frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ML feature datasets")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Specific tickers to build features for (default: all)",
    )
    args = parser.parse_args()
    build_all_features(tickers=args.tickers)
