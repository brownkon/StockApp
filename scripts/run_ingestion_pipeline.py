import argparse
import logging
from ingest_market_data import run_ingestion as run_market
from ingest_macro_data import run_ingestion as run_macro
from ingest_sentiment_data import run_ingestion as run_sentiment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(mode):
    logger.info(f"--- Starting Data Ingestion Pipeline in {mode.upper()} mode ---")
    
    try:
        logger.info("Initializing Market Data Ingestion...")
        run_market(mode=mode)
    except Exception as e:
        logger.error(f"Market data ingestion failed: {e}")
        
    try:
        logger.info("Initializing Macro Data Ingestion...")
        run_macro(mode=mode)
    except Exception as e:
        logger.error(f"Macro data ingestion failed: {e}")
        
    try:
        logger.info("Initializing Sentiment Data Ingestion...")
        run_sentiment(mode=mode)
    except Exception as e:
        logger.error(f"Sentiment data ingestion failed: {e}")
        
    logger.info("--- Data Ingestion Pipeline Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['daily', 'historical'], default='daily')
    args = parser.parse_args()
    main(args.mode)
