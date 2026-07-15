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

