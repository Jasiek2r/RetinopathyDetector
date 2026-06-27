from abc import abstractmethod


class ModelLoader:
    @abstractmethod
    def load(self, path):
        pass