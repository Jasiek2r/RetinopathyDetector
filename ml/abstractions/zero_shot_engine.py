from abc import abstractmethod


class ZeroShotEngine:
    @abstractmethod
    def evaluate(self, dataset):
        pass
