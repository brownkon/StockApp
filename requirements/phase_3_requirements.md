# Phase 3 Requirements Document: ML Modeling & Backtesting

## 1. Overview
Phase 3 transforms the raw and engineered data (from Phases 1–2) into actionable trading signals using machine learning, and validates those signals through rigorous historical backtesting. By the end of this phase, the system will have a trained XGBoost model that predicts forward returns and a backtesting harness that proves (or disproves) the strategy's viability on out-of-sample data.

**Estimated Timeline**: 2 Weeks (Weeks 4–5)
**Primary Persona**: Single Developer / Student
**Prerequisite**: Phases 1 and 2 complete — database populated with daily prices, technical indicators, macro indicators, sentiment scores, and options flow data.

---

## 2. Feature Engineering & Dataset Assembly

### 2.1 Unified Feature DataFrame Builder (`scripts/build_features.py`)
Assemble a single, point-in-time feature matrix per ticker per date by joining the following sources:

| Source Table           | Features to Extract                                                                                                                                             |
|:-----------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `daily_prices`         | `close`, `volume`, daily returns (`pct_change`), 5-day return, 20-day return                                                                                    |
| `technical_indicators` | `sma_50`, `sma_200`, `ema_20`, `rsi_14`, `macd`, `macd_signal`, `macd_hist`, `bb_upper`, `bb_lower`, `bb_mid`, `atr_14`, `adx_14`, `plus_di_14`, `minus_di_14`, `obv` |
| `macro_indicators`     | Latest forward-filled values for: Fed Funds Rate, CPI, 10Y Treasury, Unemployment Rate, Consumer Sentiment                                                      |
| `daily_sentiment`      | `unified_score`, `article_count` (forward-filled when missing for a given ticker/date)                                                                           |
| `daily_options_data`   | `put_call_ratio`, `implied_volatility` (forward-filled when missing)                                                                                             |

**Derived features to compute:**
- `price_vs_sma50`: `(close - sma_50) / sma_50` — distance from 50-day SMA
- `price_vs_sma200`: `(close - sma_200) / sma_200` — distance from 200-day SMA
- `bb_position`: `(close - bb_lower) / (bb_upper - bb_lower)` — where price sits in Bollinger Band
- `volatility_ratio`: `atr_14 / close` — normalized volatility
- `volume_sma_ratio`: `volume / volume.rolling(20).mean()` — relative volume
- Day-of-week (0–4) as a categorical integer
- Month (1–12) as a categorical integer

**Target variable (label):**
- `fwd_return_1d`: The next trading day's percentage return `(close[t+1] - close[t]) / close[t]`
- `fwd_return_5d`: The 5-trading-day forward return `(close[t+5] - close[t]) / close[t]`
- `signal`: Binary classification — 1 if `fwd_return_5d > 0`, else 0

**Requirements:**
- All features must be strictly point-in-time: no future data leakage.
- Forward-fill macro and sentiment data where daily values are missing (these indicators report less frequently).
- Drop rows where any critical feature is NaN (after forward-fill).
- The builder must be runnable standalone or callable as a module.
- Output: a pandas DataFrame saved to `data/features_{ticker}.parquet` per ticker, and optionally an `all_features.parquet` combining all tickers.

### 2.2 Train / Test Split Strategy
- **Walk-Forward Validation**: Respect temporal ordering. Never shuffle data.
- **Split**: Use data up to a configurable cutoff date as training, and data after that date as testing (out-of-sample).
- **Default split**: Train on everything before `2024-01-01`, test on `2024-01-01` onward.
- A validation set can be carved from the last 20% of the training window for hyperparameter tuning.

---

## 3. ML Model Training

### 3.1 Model: XGBoost Classifier (`scripts/train_model.py`)
- **Library**: `xgboost` (via `pip install xgboost`)
- **Task**: Binary classification — predict whether the 5-day forward return is positive (`signal = 1`) or negative (`signal = 0`).
- **Features**: All numeric columns from the unified DataFrame (excluding target columns and date/ticker).

**Training requirements:**
1. Load feature files from `data/features_{ticker}.parquet`.
2. Combine all tickers into one large training set (multi-stock model).
3. Apply the temporal train/test split.
4. Train an `XGBClassifier` with the following baseline hyperparameters:
   - `n_estimators=500`
   - `max_depth=6`
   - `learning_rate=0.05`
   - `subsample=0.8`
   - `colsample_bytree=0.8`
   - `eval_metric='logloss'`
   - `early_stopping_rounds=50`
5. Use the validation set for early stopping.
6. Log feature importances.
7. Serialize the trained model to `models/xgb_model.pkl` (via `joblib`).
8. Print and log evaluation metrics on the **test set**:
   - Accuracy
   - Precision, Recall, F1 (for class 1 — our "buy" signal)
   - ROC-AUC
   - Confusion Matrix

### 3.2 Model Inference (`scripts/predict_signals.py`)
- Load the serialized model from `models/xgb_model.pkl`.
- Build the feature row for the latest available date for each ticker.
- Output a signal DataFrame: `ticker | date | predicted_signal | predicted_probability`.
- Save predictions to `data/predictions_latest.csv` and optionally write to a new `predictions` table in the database.

### 3.3 Database Schema Addition
```sql
CREATE TABLE predictions (
    ticker VARCHAR(10),
    date DATE,
    predicted_signal INTEGER,    -- 0 or 1
    predicted_probability FLOAT, -- probability of signal=1
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (ticker, date)
);
```
Add this model to `scripts/db.py`.

---

## 4. Backtesting

### 4.1 Backtesting Engine (`scripts/run_backtest.py`)
- **Library**: `backtrader` (via `pip install backtrader`)
- **Strategy**: Implement a custom `backtrader.Strategy` that:
  1. Reads the model's predictions for each trading day.
  2. If `predicted_signal == 1` and `predicted_probability >= threshold` (default 0.55), go **long** at market open the next day.
  3. If already in a position and the signal flips to 0, **close** the position.
  4. Implements a trailing stop-loss based on ATR (e.g., 2× ATR below entry).
  5. Limits concurrent positions to a configurable max (default 5).
  6. Uses equal-weight position sizing: `equity / max_positions`.

**Backtest parameters:**
- **Data range**: Out-of-sample test period (default `2024-01-01` to present).
- **Starting capital**: `$10,000` (paper).
- **Commission**: `0.0` (Alpaca zero-commission), but include `0.01%` slippage.
- **Benchmark**: Buy-and-hold SPY over the same period.

### 4.2 Performance Metrics
After the backtest, compute and display:
- **Total Return** (strategy vs. benchmark)
- **Annualized Return**
- **Sharpe Ratio** (risk-free rate = current Fed Funds / 252)
- **Max Drawdown** (peak-to-trough in %)
- **Win Rate** (% of trades that were profitable)
- **Profit Factor** (gross profit / gross loss)
- **Total Trades**
- **Average Trade Duration (days)**

### 4.3 Output
- Print a formatted performance summary to the console.
- Generate and save an equity curve plot to `data/backtest_equity_curve.png`.
- Save trade log to `data/backtest_trades.csv`.
- Save a markdown report to `data/backtest_report.md`.

---

## 5. New Database Models
Add to `scripts/db.py`:

```python
class Prediction(Base):
    __tablename__ = 'predictions'
    ticker = Column(String(10), primary_key=True)
    date = Column(Date, primary_key=True)
    predicted_signal = Column(Integer)
    predicted_probability = Column(Float)
    model_version = Column(String(50))
    created_at = Column(DateTime)
```

---

## 6. New Dependencies
Add to `requirements.txt`:
```
xgboost
scikit-learn
joblib
backtrader
matplotlib
pyarrow
```

---

## 7. File Deliverables

| File                           | Purpose                                      |
|:-------------------------------|:---------------------------------------------|
| `scripts/build_features.py`    | Assemble unified feature DataFrame           |
| `scripts/train_model.py`       | Train XGBoost model, evaluate, serialize     |
| `scripts/predict_signals.py`   | Run inference for latest data                |
| `scripts/run_backtest.py`      | Execute backtest, generate report            |
| `models/xgb_model.pkl`         | Serialized trained model                     |
| `data/features_*.parquet`      | Per-ticker feature files                     |
| `data/predictions_latest.csv`  | Latest model predictions                     |
| `data/backtest_report.md`      | Backtest performance summary                 |
| `data/backtest_equity_curve.png`| Equity curve visualization                  |
| `data/backtest_trades.csv`     | Full trade log from backtest                 |
| `tests/test_build_features.py` | Unit tests for feature engineering           |
| `tests/test_train_model.py`    | Unit tests for model training pipeline       |
| `tests/test_predict_signals.py`| Unit tests for inference pipeline            |
| `tests/test_run_backtest.py`   | Unit tests for backtesting engine            |

---

## 8. Acceptance Criteria
Phase 3 is complete when:
1. `build_features.py` produces a clean, leakage-free feature DataFrame for all tickers.
2. `train_model.py` trains an XGBoost model, logs metrics, and serializes to disk.
3. `predict_signals.py` generates a signal for the latest available date for each tracked ticker.
4. `run_backtest.py` simulates the strategy on out-of-sample data and produces a formatted performance report.
5. All unit tests pass.
6. No future data leakage exists in the pipeline (validated by tests).
