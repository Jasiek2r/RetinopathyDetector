from abc import abstractmethod

import torch
from torch.utils.data import Dataset


class MLEngine:
    @abstractmethod
    def train(self, train_dataset: Dataset, val_dataset: Dataset):
        pass

    @abstractmethod
    def test(self, dataset: Dataset):
        pass

    @abstractmethod
    def full_evaluation(self, train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset):
        pass

    @abstractmethod
    def evaluate_with_uncertainty(self, dataset, batch_size=32, num_tta_steps=5,
                                  coverage_steps=[1.0, 0.9, 0.8, 0.7, 0.6]):
        pass

