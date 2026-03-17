import argparse
import logging
from ingest_market_data import run_ingestion as run_market
from ingest_macro_data import run_ingestion as run_macro
from ingest_sentiment_data import run_ingestion as run_sentiment
from process_sentiment import process_sentiments as run_sentiment_processing
from calculate_indicators import run_calculations as run_technical_indicators
from ingest_options_data import run_ingestion as run_options
from view_data import main as generate_report

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
        logger.info("Initializing Technical Indicators Calculation...")
        run_technical_indicators()
    except Exception as e:
        logger.error(f"Technical indicators calculation failed: {e}")
        
    try:
        logger.info("Initializing Macro Data Ingestion...")
        run_macro(mode=mode)
    except Exception as e:
        logger.error(f"Macro data ingestion failed: {e}")
        
    try:
        logger.info("Initializing Sentiment Data Ingestion...")
        run_sentiment(mode=mode)
        
        logger.info("Processing Daily Sentiments with FinBERT...")
        # For daily mode, checking last 7 days is enough. For historical, maybe more.
        days_back = 30 if mode == 'historical' else 7
        run_sentiment_processing(days_back=days_back)
    except Exception as e:
        logger.error(f"Sentiment data ingestion/processing failed: {e}")
        
    try:
        logger.info("Initializing Options Data Ingestion...")
        run_options(mode=mode)
    except Exception as e:
        logger.error(f"Options data ingestion failed: {e}")
        
    logger.info("--- Data Ingestion Pipeline Complete ---")
    
    logger.info("Generating pipeline summary report...")
    try:
        generate_report()
    except Exception as e:
        logger.error(f"Could not generate report: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['daily', 'historical'], default='daily')
    args = parser.parse_args()
    main(args.mode)
