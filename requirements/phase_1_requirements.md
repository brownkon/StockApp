# Phase 1 Requirements Document: Data Infrastructure

## 1. Overview
Phase 1 focuses exclusively on laying the foundation of the AI trading system. The core objective is to establish the data infrastructure capable of reliably collecting required datasets at virtually zero cost. By the end of this phase, the system should be able to fetch historical and daily updates for price data, macroeconomic indicators, and social sentiment data, storing them securely for subsequent ML training and inference.

**Estimated Timeline**: 1–2 Weeks
**Primary Persona**: Single Developer / Student

## 2. Infrastructure Setup
### 2.1 Code Repository
- **Tool**: GitHub (Free Tier)
- **Requirements**:
  - Initialize a new Git repository.
  - Implement a standard directory structure (e.g., `data/`, `scripts/`, `notebooks/`, `tests/`).
  - Create a robust `.gitignore` file to exclude environment variables, local databases, virtual environments, and intermediate data files.
  - Establish a `requirements.txt` or `Pipfile` to strictly manage dependencies for reproducibility.

### 2.2 Database Configuration
- **Tool**: Supabase (Free Tier PostgreSQL)
- **Objective**: Serve as the persistent storage layer for the trading system.
- **Requirements**:
  - Set up a new project on Supabase.
  - Define the initial database schemas (see Section 3.4 for details).
  - Securely store connection credentials (connection string, API keys) in a local `.env` file (never committed to source control).
  - Install and configure required Python client libraries (`supabase` or `psycopg2`/`SQLAlchemy`).

## 3. Data Collection Modules
The system must gracefully handle daily data ingestion. All scripts must be written in Python.

### 3.1 Market Data Ingestion
- **Source**: Yahoo Finance via the `yfinance` library.
- **Scope**: A predefined basket of 5–10 highly liquid stocks or ETFs (e.g., SPY, QQQ, AAPL, MSFT, GLD).
- **Requirements**:
  - **Historical Backfill**: Script to download the last 5–10 years of daily Open, High, Low, Close, Adjusted Close, and Volume data for the instrument basket.
  - **Daily Update**: Script to query the previous trading day's metrics and append them to the database.
  - **Error Handling**: Implement basic retries and logging in case the Yahoo Finance API rate-limits or returns malformed data.

### 3.2 Macroeconomic Data Ingestion
- **Source**: Federal Reserve Economic Data (FRED) API.
- **Requirements**:
  - Register for a free FRED API key.
  - Identify 3–5 key macroeconomic indicators (e.g., Effective Federal Funds Rate, Consumer Price Index, 10-Year Treasury Constant Maturity Rate).
  - Develop a script to retrieve historical monthly/quarterly data and the latest available figures.
  - Handle alignment issues (macro data is often monthly, while price data is daily) by implementing forward-fill strategies during the preprocessing stage (to be handled in Phase 2, but the raw data must be stored correctly now).

### 3.3 Sentiment Data Ingestion
- **Sources**: 
  - Reddit (via public JSON API endpoints using `requests`)
  - Financial RSS Feeds (e.g., Yahoo Finance RSS, CNBC)
- **Requirements**:
  - **Reddit**:
    - Script to scrape the top daily and historical posts/titles from targeted financial subreddits (e.g., `r/investing`, `r/stocks`, `r/pennystocks`) using Reddit's JSON structure.
    - Include a custom `User-Agent` to respect Reddit's API guidelines for unauthenticated requests.
  - **RSS Feeds**:
    - Use the `feedparser` library to extract headlines and publish dates from 2–3 major financial news feeds daily.
  - **Storage**: Store the raw text, timestamps, the source label, and an `external_id` to prevent duplicates. (Sentiment scoring via FinBERT is out of scope for Phase 1; this phase only concerns raw data collection).

### 3.4 Database Schema Design
Design initial tables in the PostgreSQL database:
1.  **`daily_prices`**: Ticker, Date, Open, High, Low, Close, Adj_Close, Volume. (Composite Primary Key: Ticker, Date).
2.  **`macro_indicators`**: Indicator_Name, Date, Value. (Composite Primary Key: Indicator_Name, Date).
3.  **`raw_sentiment_text`**: ID (Auto-increment), External_ID (Unique string for duplicate checking), Source (Reddit/RSS), Timestamp, Text_Content, Ticker_Mentioned.

## 4. Automation & Maintenance Tools
### 4.1 Logging & Progress Tracking
- Use Python's built-in `logging` module for system logs and `tqdm` for terminal progress bars.
- Requirements:
  - All ingestion scripts must log `INFO` messages upon successful completion and `ERROR` messages with stack traces if a failure occurs.
  - Implement `tqdm` progress bars for long-running tasks, such as historical data bulk downloads and (in later phases) model training epochs.
  - Include exact counts of rows ingested/updated in the final logs.

### 4.2 Utility Scripts
- Create a master runner script (e.g., `run_ingestion_pipeline.py`) that executes Market, Macro, and Sentiment scripts sequentially.
- Ensure the runner script can handle both historical bulk-loading (run once) and daily incremental updates.

## 5. Acceptance Criteria
Phase 1 is complete when:
1.  The GitHub repository is configured with correct `.gitignore` and dependency files.
2.  The Supabase database is online, accessible via Python, with the necessary tables created.
3.  A developer can run the historical ingestion scripts to populate the database with years of market and macro data.
4.  A developer can run the master daily ingestion script effortlessly, downloading the latest daily data across prices, economic indicators, and text sources, and successfully writing them to the database without duplicates.
