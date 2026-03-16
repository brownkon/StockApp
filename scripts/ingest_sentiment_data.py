import time
import requests
import logging
import argparse
from datetime import datetime, timedelta, timezone
import feedparser
from sqlalchemy.orm import sessionmaker
from db import engine, RawSentimentText
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Reddit config
SUBREDDITS = ['investing', 'stocks', 'pennystocks']

# RSS config
RSS_FEEDS = {
    'Yahoo Finance': 'https://finance.yahoo.com/news/rssindex',
    'CNBC': 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664'
}

def clean_duplicate_sentiment(session, source, text):
    """Simple check for exact duplicates."""
    return session.query(RawSentimentText).filter_by(source=source, text_content=text).first() is not None

def fetch_reddit_sentiment(session, mode):
    limit = 100 # Unauthenticated json limits usually apply around 100
    time_filter = 'all' if mode == 'historical' else 'day'
    headers = {'User-Agent': 'python:stock.trading.app:v1.0 (by local dev)'}
    
    total_added = 0
    tqdm_subs = tqdm(SUBREDDITS, desc="Ingesting Reddit Sentiment (JSON Web Scrape)")
    
    for sub_name in tqdm_subs:
        try:
            url = f"https://www.reddit.com/r/{sub_name}/top.json?t={time_filter}&limit={limit}"
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            posts = data.get('data', {}).get('children', [])
            
            for post_data in posts:
                post = post_data.get('data', {})
                title = post.get('title', '')
                selftext = post.get('selftext', '')
                text_content = f"{title} {selftext}".strip()
                if not text_content: continue
                
                created_utc = post.get('created_utc')
                if not created_utc: continue
                
                dt = datetime.fromtimestamp(created_utc, tz=timezone.utc)
                source_label = f"Reddit: r/{sub_name}"
                
                if not clean_duplicate_sentiment(session, source_label, text_content):
                    record = RawSentimentText(
                        source=source_label,
                        timestamp=dt,
                        text_content=text_content
                    )
                    session.add(record)
                    total_added += 1
            session.commit()
            time.sleep(2) # Sleep to respect Reddit rate limits
            
        except Exception as e:
            logger.error(f"Error fetching from r/{sub_name}: {e}")
            session.rollback()

    return total_added

def fetch_rss_sentiment(session):
    total_added = 0
    tqdm_feeds = tqdm(RSS_FEEDS.items(), desc="Ingesting RSS Sentiment")
    
    for source_name, url in tqdm_feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                text_content = entry.title
                if not text_content: continue
                
                # Try to parse published date
                dt = datetime.now(timezone.utc)
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    import time
                    dt = datetime.fromtimestamp(time.mktime(entry.published_parsed), tz=timezone.utc)
                
                source_label = f"RSS: {source_name}"
                if not clean_duplicate_sentiment(session, source_label, text_content):
                    record = RawSentimentText(
                        source=source_label,
                        timestamp=dt,
                        text_content=text_content
                    )
                    session.add(record)
                    total_added += 1
            session.commit()
        except Exception as e:
            logger.error(f"Error fetching RSS for {source_name}: {e}")
            session.rollback()

    return total_added

def run_ingestion(mode='daily'):
    if not engine:
        logger.error("Database connection not established. Export DATABASE_URL.")
        return

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    
    logger.info(f"Running {mode} sentiment data ingestion")
    
    added_reddit = fetch_reddit_sentiment(session, mode)
    added_rss = fetch_rss_sentiment(session) # RSS is naturally point-in-time, we just get latest
    
    total = added_reddit + added_rss
    logger.info(f"Sentiment data ingestion complete. Added {total} records (Reddit: {added_reddit}, RSS: {added_rss}).")
    session.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['daily', 'historical'], default='daily')
    args = parser.parse_args()
    run_ingestion(args.mode)
