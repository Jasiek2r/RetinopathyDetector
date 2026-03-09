from abc import abstractmethod

import torch


class MLEngine:
    @abstractmethod
    def train(self, x: torch.Tensor, y: torch.Tensor):
        pass

    @abstractmethod
    def test(self, x: torch.Tensor, y: torch.Tensor):
        pass
