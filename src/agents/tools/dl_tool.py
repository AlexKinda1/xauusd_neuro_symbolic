"""
Tool for deep learning inference.
"""

from src.predictive_model.inference import load_model, predict

class DLTool:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def get_prediction(self, data):
        return predict(self.model, data)