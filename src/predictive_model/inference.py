"""
Inference module for making predictions.
"""

import tensorflow as tf

def load_model(model_path: str):
    model = tf.keras.models.load_model(model_path)
    return model

def predict(model, input_data):
    prediction = model.predict(input_data)
    return prediction