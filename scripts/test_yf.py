import yfinance as yf
from datetime import datetime
data = yf.download('SPY', start='2026-03-01', end='2026-03-10', progress=False)
for date, row in data.iterrows():
    print(repr(date), repr(date.date()))
    break
