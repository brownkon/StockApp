import argparse
import logging
from datetime import datetime, timedelta, date
from collections import defaultdict
import torch
from sqlalchemy.orm import sessionmaker
from transformers import pipeline
from db import engine, RawSentimentText, DailySentiment
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_sentiment_pipeline():
    # Use GPU/MPS if available
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Avoid MPS for some transformer models if it causes issues, but typically fine
        device = "cpu" # Defaulting to CPU for stability on Mac unless requested otherwise, PyTorch MPS sometimes has embedding issues.
    
    logger.info(f"Loading FinBERT model on {device}...")
    # 'top_k=None' returns scores for all classes
    return pipeline("text-classification", model="ProsusAI/finbert", top_k=None, device=device)

def calculate_unified_score(pos, neg, neu):
    # Example formula for unified score: range from -1 to 1
    # positive * 1 + negative * (-1) + neutral * 0
    return pos - neg

def process_sentiments(days_back=7):
    if not engine:
        logger.error("Database connection not established. Export DATABASE_URL.")
        return

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    
    cutoff_date = datetime.utcnow() - timedelta(days=days_back)
    
    # Fetch texts to process
    texts = session.query(RawSentimentText).filter(
        RawSentimentText.timestamp >= cutoff_date
    ).all()
    
    if not texts:
        logger.info("No texts to process.")
        session.close()
        return
        
    logger.info(f"Processing {len(texts)} texts...")
    
    # Group by (date, ticker)
    grouped_texts = defaultdict(list)
    for t in texts:
        d = t.timestamp.date()
        ticker = t.ticker_mentioned if t.ticker_mentioned else 'MARKET'
        grouped_texts[(d, ticker)].append(t.text_content)
        
    try:
        nlp = get_sentiment_pipeline()
    except Exception as e:
        logger.error(f"Failed to load FinBERT: {e}")
        session.close()
        return

    new_objs = []
    
    # Existing sentiment records to update
    existing_records = session.query(DailySentiment).filter(
        DailySentiment.date >= cutoff_date.date()
    ).all()
    existing_map = {(r.date, r.ticker): r for r in existing_records}
    
    for (d, ticker), content_list in tqdm(grouped_texts.items(), desc="Calculating Daily Sentiments"):
        # Batch inference
        # To avoid sequence length issues, we can truncate or just let pipeline handle it if truncation=True
        results = nlp(content_list, truncation=True, max_length=512)
        
        # results is a list of lists, e.g., [[{'label': 'positive', 'score': 0.8}, ...], [...]]
        avg_pos = 0.0
        avg_neg = 0.0
        avg_neu = 0.0
        
        for res in results:
            # res is a list of dicts for each label
            for score_dict in res:
                if score_dict['label'] == 'positive':
                    avg_pos += score_dict['score']
                elif score_dict['label'] == 'negative':
                    avg_neg += score_dict['score']
                elif score_dict['label'] == 'neutral':
                    avg_neu += score_dict['score']
                    
        count = len(results)
        if count > 0:
            avg_pos /= count
            avg_neg /= count
            avg_neu /= count
            
        unified = calculate_unified_score(avg_pos, avg_neg, avg_neu)
        
        if (d, ticker) in existing_map:
            record = existing_map[(d, ticker)]
            record.positive_score = avg_pos
            record.negative_score = avg_neg
            record.neutral_score = avg_neu
            record.unified_score = unified
            record.article_count = count
        else:
            record = DailySentiment(
                date=d,
                ticker=ticker,
                positive_score=avg_pos,
                negative_score=avg_neg,
                neutral_score=avg_neu,
                unified_score=unified,
                article_count=count
            )
            new_objs.append(record)
            
    if new_objs:
        session.bulk_save_objects(new_objs)
        
    try:
        session.commit()
        logger.info(f"Sentiment processing complete. Updated {len(grouped_texts)} daily sentiment records.")
    except Exception as e:
        logger.error(f"Error saving sentiment scores: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7, help="Number of days to go back and process")
    args = parser.parse_args()
    process_sentiments(args.days)
