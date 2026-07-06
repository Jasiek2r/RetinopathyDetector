from abc import abstractmethod


class ZeroShotEngine:
    @abstractmethod
    def evaluate(self, dataset):
        pass
    @abstractmethod
    def build_prototypes(self, dataset):
        pass