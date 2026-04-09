"""
Feature engineering module for technical indicators.
"""

import pandas_ta as ta

def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    # Add RSI
    data['RSI'] = ta.rsi(data['Close'])
    # Add MACD
    macd = ta.macd(data['Close'])
    data = pd.concat([data, macd], axis=1)
    # Add more indicators as needed
    return data