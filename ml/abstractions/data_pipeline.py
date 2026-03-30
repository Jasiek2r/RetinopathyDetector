from abc import abstractmethod


class DataPipeline:
    @abstractmethod
    def run(self, data, path):
        pass
