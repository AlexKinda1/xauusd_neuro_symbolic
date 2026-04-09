"""
Training script for the LSTM model.
"""

import numpy as np
from architecture import build_lstm_model
from src.data_pipeline.preprocessor import create_sequences

def train_model(X_train, y_train, params):
    model = build_lstm_model((params['seq_length'], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'])
    return model