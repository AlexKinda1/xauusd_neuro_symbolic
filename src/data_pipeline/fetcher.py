"""
Data fetcher module for retrieving historical OHLC data.
"""

import yfinance as yf
import pandas as pd

def fetch_yfinance_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch data from Yahoo Finance.
    """
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

# Add functions for OANDA, MT5 if needed