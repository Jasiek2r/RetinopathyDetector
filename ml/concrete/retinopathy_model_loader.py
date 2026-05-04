import torch

from ml.abstractions.ml_engine import MLEngine
from ml.abstractions.model_loader import ModelLoader
from ml.concrete.simple_cnn import SimpleCNN


class RetinopathyModelLoader(ModelLoader):
    def __init__(self, engine: MLEngine):
        self.engine = engine

    def load(self, path):
        model = SimpleCNN()
        weights = torch.load(path)
        model.load_state_dict(weights)
        model.to("cuda")
        self.engine.model = model
