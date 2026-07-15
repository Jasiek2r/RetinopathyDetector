import torch

from ml.abstractions.ml_engine import MLEngine
from ml.abstractions.model_loader import ModelLoader
from ml.concrete.model_provider import ModelProvider


class RetinopathyModelLoader(ModelLoader):
    def __init__(self, engine: MLEngine, provider: ModelProvider):
        self.engine = engine
        self.provider = provider

    def load(self, path):
        model = self.provider.create_model()
        weights = torch.load(path)
        model.load_state_dict(weights)
        model.to("cuda")
        self.engine.model = model
