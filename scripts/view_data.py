import os
import pandas as pd
from db import engine
from datetime import datetime

def generate_report():
    if not engine:
        return "Database connection not established. Make sure DATABASE_URL is in your .env file."

    report = [f"# StockTradingApp Master Data Report"]
    report.append(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        # Check Overall Database Health
        report.append("## 🏥 Database Health")
        total_prices = pd.read_sql("SELECT COUNT(*) as c FROM daily_prices", engine).iloc[0]['c']
        total_macro = pd.read_sql("SELECT COUNT(*) as c FROM macro_indicators", engine).iloc[0]['c']
        total_tech = pd.read_sql("SELECT COUNT(*) as c FROM technical_indicators", engine).iloc[0]['c']
        total_sent = pd.read_sql("SELECT COUNT(*) as c FROM raw_sentiment_text", engine).iloc[0]['c']
        total_opt = pd.read_sql("SELECT COUNT(*) as c FROM daily_options_data", engine).iloc[0]['c']
        
        health_data = {
            "Table": ["daily_prices", "macro_indicators", "technical_indicators", "raw_sentiment_text", "daily_options_data"],
            "Total Rows": [total_prices, total_macro, total_tech, total_sent, total_opt]
        }
        report.append(pd.DataFrame(health_data).to_markdown(index=False) + "\n")

        # 1. Market Data Coverage
        df_prices = pd.read_sql("SELECT ticker, COUNT(*) as days, MIN(date) as first_date, MAX(date) as last_date FROM daily_prices GROUP BY ticker ORDER BY ticker", engine)
        report.append("## 📊 Pricing & Coverage")
        if df_prices.empty:
            report.append("> No market data found.\n")
        else:
            report.append(df_prices.to_markdown(index=False) + "\n")

        # 2. Latest Macro Conditions
        report.append("## 📈 Macroeconomic Environment (Latest Print)")
        df_latest_macro = pd.read_sql("""
            SELECT indicator_name, value, date 
            FROM (
                SELECT indicator_name, value, date,
                       ROW_NUMBER() OVER(PARTITION BY indicator_name ORDER BY date DESC) as rn
                FROM macro_indicators
            ) as sub
            WHERE rn = 1
        """, engine)
        if df_latest_macro.empty:
            report.append("> No macro data available.\n")
        else:
            report.append(df_latest_macro.to_markdown(index=False) + "\n")
            
        # 3. Present Technical Signals (Market Breadth & Extremes)
        report.append("## ⚙️ Technical Health & Breadth")
        # Let's get the absolute latest date for technicals
        latest_date_tech_df = pd.read_sql("SELECT MAX(date) as max_d FROM technical_indicators", engine)
        latest_date = latest_date_tech_df.iloc[0]['max_d'] if not latest_date_tech_df.empty else None
        
        if latest_date:
            report.append(f"*(As of {latest_date})*\n")
            # Calculate Breadth
            breadth_query = f"""
                SELECT 
                    SUM(CASE WHEN close > sma_50 THEN 1 ELSE 0 END) as above_50,
                    SUM(CASE WHEN close > sma_200 THEN 1 ELSE 0 END) as above_200,
                    COUNT(*) as total_tickers
                FROM technical_indicators t
                JOIN daily_prices p ON t.ticker = p.ticker AND t.date = p.date
                WHERE t.date = '{latest_date}'
            """
            
            try:
                breadth_df = pd.read_sql(breadth_query, engine)
                above_50 = breadth_df.iloc[0]['above_50']
                above_200 = breadth_df.iloc[0]['above_200']
                total_tickers = breadth_df.iloc[0]['total_tickers']
                
                pct_50 = (above_50 / total_tickers * 100) if total_tickers > 0 else 0
                pct_200 = (above_200 / total_tickers * 100) if total_tickers > 0 else 0
                
                report.append(f"- **Stocks > 50-day SMA:** {pct_50:.1f}%\n")
                report.append(f"- **Stocks > 200-day SMA:** {pct_200:.1f}%\n\n")
            except Exception as e:
                report.append(f"_Could not load breadth data: {e}_\n")
            
            # Show RSI Extremes
            df_rsi = pd.read_sql(f"SELECT ticker, rsi_14, adx_14 FROM technical_indicators WHERE date = '{latest_date}' ORDER BY rsi_14 DESC", engine)
            if not df_rsi.empty:
                overbought = df_rsi[df_rsi['rsi_14'] > 70]
                oversold = df_rsi[df_rsi['rsi_14'] < 30]
                
                report.append("### Overbought vs Oversold")
                report.append(f"- **Overbought Tracking (RSI > 70):** {', '.join(overbought['ticker'].tolist()) if not overbought.empty else 'None'}")
                report.append(f"- **Oversold Tracking (RSI < 30):** {', '.join(oversold['ticker'].tolist()) if not oversold.empty else 'None'}\n")
                
                report.append("### Highest Trend Strength (ADX > 25)")
                strong_trend = df_rsi[df_rsi['adx_14'] > 25].sort_values(by='adx_14', ascending=False).head(5)
                if not strong_trend.empty:
                    report.append(strong_trend.to_markdown(index=False) + "\n")
                else:
                    report.append("> Market is generally chopping sideways (No strong ADX).\n")
                    
        else:
            report.append("> No technical indicators found.\n")

        # 4. Sentiment Output
        df_sent = pd.read_sql("SELECT source, COUNT(*) as posts, MIN(timestamp) as oldest, MAX(timestamp) as newest FROM raw_sentiment_text GROUP BY source", engine)
        report.append("## 💬 Social & News Sentiment Pipeline")
        if df_sent.empty:
            report.append("> No sentiment data found.\n")
        else:
            report.append(df_sent.to_markdown(index=False) + "\n")

        # 5. Options Flow
        df_opt_latest = pd.read_sql("""
            SELECT ticker, put_volume, call_volume, put_call_ratio, date 
            FROM (
                SELECT ticker, put_volume, call_volume, put_call_ratio, date,
                       ROW_NUMBER() OVER(PARTITION BY ticker ORDER BY date DESC) as rn
                FROM daily_options_data
            ) as sub
            WHERE rn = 1
            ORDER BY put_call_ratio DESC
            LIMIT 10
        """, engine)
        report.append("## 📉 Latest Options Flow Extrema (Highest P/C Ratios)")
        report.append("*When Put/Call ratio is high (>1.0), traders are betting strongly against the ticker.*\n")
        if df_opt_latest.empty:
            report.append("> No options data found.\n")
        else:
            report.append(df_opt_latest.to_markdown(index=False) + "\n")

    except Exception as e:
        report.append(f"**Error querying the database:** {e}")

    return "\n".join(report)

def main():
    report_text = generate_report()
    print(report_text)
    
    # Save to data folder
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    report_path = os.path.join(data_dir, 'pipeline_report.md')
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"\nReport saved to: {report_path}")

if __name__ == "__main__":
    main()
