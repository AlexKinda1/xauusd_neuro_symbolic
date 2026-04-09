"""
Data preprocessing module for normalization and windowing.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def normalize_data(data: pd.DataFrame, features: list) -> pd.DataFrame:
    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])
    return data, scaler

def create_sequences(data: pd.DataFrame, seq_length: int, target_col: str):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i+seq_length].values
        target = data.iloc[i+seq_length][target_col]
        sequences.append(seq)
        targets.append(target)
    return sequences, targets