# Product Requirements Document (PRD): Zero-Cost AI Stock Trading System

## 1. Product Overview
**Purpose:** 
The objective is to build a fully autonomous, AI-driven stock and ETF trading system. The platform will ingest historical market data, macroeconomic indicators, and social sentiment, process this data through machine learning models to generate trading signals, and execute trades automatically via a broker API.

**Constraints & Scale:**
- **Zero Budget Operations:** The system must utilize exclusively free data sources, open-source software, free cloud tiers, and free broker APIs. The user has a maximum budget allowance of $10, which can be reserved for edge-case infrastructural needs (e.g., a cheap domain name or a very basic VPS if needed, but not required).
- **Scale:** The system is designed for a single developer or student. It will not compete with High-Frequency Trading (HFT) firms. Instead, it will focus on lower-frequency trading strategies, such as daily or hourly swing trading, where latency is not the critical factor.

## 2. Free Data Sources
The system will construct its dataset by polling the following free sources:
- **Price & Volume Data:** `yfinance` library for free historical and near real-time stock/ETF data from Yahoo Finance.
- **Macroeconomic Data:** **FRED (Federal Reserve Economic Data)** API for interest rates, inflation data, and employment metrics (free API key).
- **Social Sentiment (Reddit):** Web scraping via public Reddit JSON endpoints (using `requests`) to pull from financial subreddits (e.g., r/investing, r/stocks, r/pennystocks).
- **Financial News / RSS:** Free RSS feeds from Yahoo Finance, CNBC, or Google News for daily headlines.
- **Search Trends:** `pytrends` library to scrape Google Trends data for specific stock tickers or market terms.
- **Corporate Filings:** SEC EDGAR API for free access to fundamental data and quarterly reports (10-Q, 10-K).

## 3. System Architecture
The architecture is designed to be highly modular and decoupled so it can run either locally on a personal computer or via serverless cron jobs (like GitHub Actions).

```text
+-------------------+       +-----------------------+       +-------------------------+
|   Data Sources    | ----> |   Data Ingestion      | ----> |     Database            |
| (yfinance, FRED,  |       |   & Preprocessing     |       | (Supabase / local DB)   |
|  Reddit, RSS)     |       +-----------------------+       +-------------------------+
+-------------------+                                                   |
                                                                        v
+-------------------+       +-----------------------+       +-------------------------+
|  Model Training   | <---- | Feature Engineering   | <---- |   Sentiment Analysis    |
|  (Local Machine)  |       | (Technical, Macro)    |       |   (FinBERT Model)       |
+-------------------+       +-----------------------+       +-------------------------+
          |
          v
+-------------------+       +-----------------------+       +-------------------------+
| ML Inference      | ----> | Risk Management       | ----> | Trade Execution         |
| (Signal Engine)   |       | (Sizing, Stop Loss)   |       | (Alpaca Free API)       |
+-------------------+       +-----------------------+       +-------------------------+
```

## 4. Machine Learning Approach
We will utilize lightweight, robust tabular models for point-in-time predictions and open-source models for NLP.
- **Primary Signal Models:** **XGBoost**, **LightGBM**, or **Random Forest** (via `scikit-learn`). These models are computationally inexpensive, highly interpretable, and perfect for tabular feature sets (technical indicators, sentiment scores, macroeconomic variables).
- **Time-Series (Optional):** **LSTM** (Long Short-Term Memory) networks via `PyTorch` can be used to model sequential price action, but require more tuning and compute power.
- **Training Strategy:** Models will be trained using Walk-Forward Validation on comprehensive historical market data (e.g., train on historical data from years 1-3, test on year 4; train on years 2-4, test on year 5) to respect the temporal nature of market data and thoroughly evaluate performance.
- **Model Updates:** The system will re-train its ML models locally once a week (e.g., on weekends) and export the serialized model weights (e.g., `.pkl` or `.onnx` files) for the daily inference engine to use.

## 5. Sentiment Analysis System
Evaluating market psychology relies on interpreting unstructured text data.
- **Model:** **FinBERT** (available via Hugging Face `transformers`). FinBERT is a pre-trained NLP model fine-tuned specifically on financial text.
- **Execution:** 
  1. Scrape Reddit titles/comments and RSS news headlines daily.
  2. Pass texts through FinBERT locally.
  3. Aggregate the probability scores of `Positive`, `Negative`, and `Neutral` classifications into a unified daily "Sentiment Score" for each traded ticker.
- **Cost:** Running FinBERT inference on daily headlines on a standard laptop CPU takes only seconds to minutes and is 100% free.

## 6. Backtesting System
Robust backtesting is the only way to prove viability before paper trading.
- **Framework:** **Backtrader** or **VectorBT** (open-source Python libraries).
- **Simulation:** Must account for transaction costs (e.g., assumed bid-ask slippage, even if commissions are zero).
- **Preventing Overfitting:** Strict isolation of training and testing data. The backtester will simulate trading on "out-of-sample" data that the ML model has never seen. 

## 7. Trading Execution
The system will interface programmatically with a brokerage.
- **Broker:** **Alpaca** API. Alpaca offers a world-class, free REST API tailored for developers. It supports both a free paper trading environment (simulated money) and fully automated real-money live trading with zero commissions.
- **Order Types:** The system will predominantly use Market-on-Open (MOO) or Limit orders depending on the predictions, executing once per day or once per hour.

## 8. Infrastructure
To maintain a $0 budget, infrastructure will leverage generous free tiers and local compute:
- **Database:** **Supabase** (Free Tier PostgreSQL) to store historical data, predictions, and trade logs in the cloud. Alternatively, a local SQLite database that syncs via GitHub.
- **Daily Operations (Inference & Trading):** **GitHub Actions**. You can schedule a Python script to run via cron syntax in GitHub Actions every trading day at 9:00 AM. It pulls the latest data, runs inference using the pre-trained model stored in the repo, checks risk constraints, and POSTs orders to Alpaca.
- **Heavy Compute (Training):** Your personal laptop. Model training happens offline over the weekend. Once trained, you push the new model file to your GitHub repository.

## 9. Risk Management
A rigorous, hard-coded ruleset to prevent account ruin:
- **Position Sizing:** Aggressive allocation. The system will be configured for higher risk tolerance, allowing position sizes of up to 10% to 20% of total account equity on high-conviction trades to maximize potential returns.
- **Stop Losses:** Average True Range (ATR) based trailing stop losses to dynamically adjust to asset volatility.
- **Portfolio Limits:** Maximum of 5 concurrent open positions. If overall market sentiment (e.g., broad S&P 500 trend) is highly negative, a hard-coded "circuit breaker" halts all long trades.

## 10. MVP Scope (4-8 Weeks)
The Minimum Viable Product focuses purely on daily swing trading for a basket of 5-10 highly liquid stocks/ETFs (e.g., SPY, QQQ, AAPL).
- **In Scope:** Daily data fetching (yfinance), basic feature engineering (Moving Averages, RSI, MACD), one ML model (XGBoost), Alpaca paper trading integration.
- **Out of Scope for MVP:** Intraday (minute-level) trading, options trading, complex deep learning architectures.

## 11. Technology Stack
- **Language:** Python 3.10+
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn`, `xgboost`, `PyTorch`
- **NLP / Sentiment:** `transformers` (Hugging Face)
- **APIs/Connections:** `yfinance`, `alpaca-trade-api`, `requests` (Reddit JSON/RSS)
- **Backtesting:** `backtrader`
- **Database:** `sqlite3` or PostgreSQL (Supabase)
- **Automation/Hosting:** GitHub Actions (Cron), Docker (optional, for local consistency)

## 12. Development Roadmap
**Phase 1: Data Infrastructure (Weeks 1-2)**
- Set up GitHub repository and Supabase database.
- Write Python scripts to download historical price data via `yfinance` and macro data via FRED.
- Write scripts to scrape Reddit and RSS feeds.

**Phase 2: Pipelines & Sentiment (Week 3)**
- Implement Hugging Face `FinBERT` to parse scraped text into daily sentiment scores.
- Build feature engineering pipelines to calculate technical indicators (RSI, MACD, Bollinger Bands).

**Phase 3: ML Modeling & Backtesting (Weeks 4-5)**
- Combine price, macro, and sentiment data into a unified DataFrame.
- Train an XGBoost model locally to predict 1-day or 5-day forward returns.
- Integrate `Backtrader` to simulate strategy performance on the last 5 years of out-of-sample data.

**Phase 4: Trading Execution (Week 6)**
- Register for an Alpaca Developer account.
- Map ML signal outputs to Alpaca order execution functions (Buy/Sell/Hold).
- Implement Risk Management logic (Stop losses, position sizing).

**Phase 5: Automation & Paper Trading (Weeks 7-8)**
- Dockerize the application or set up `requirements.txt`.
- Create a GitHub Actions workflow to run the prediction and execution script daily before market open.
- Run the system in Alpaca's **Paper Trading** environment for at least 1 month to verify that execution matches backtested expectations before considering putting real capital at risk.
