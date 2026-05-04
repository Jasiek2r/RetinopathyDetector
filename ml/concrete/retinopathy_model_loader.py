import torch

from ml.abstractions.ml_engine import MLEngine
from ml.abstractions.model_loader import ModelLoader


class RetinopathyModelLoader(ModelLoader):
    def __init__(self, engine: MLEngine):
        self.engine = engine

    def load(self, path):
        model = torch.load(path)
        self.engine.model = model
