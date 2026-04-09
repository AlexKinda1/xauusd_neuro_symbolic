"""
Neural network architecture for LSTM model.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm_model(input_shape: tuple, output_units: int = 1) -> Sequential:
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(output_units))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model