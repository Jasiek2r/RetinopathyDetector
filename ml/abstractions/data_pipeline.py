from abc import abstractmethod


class DataPipeline:
    @abstractmethod
    def run(self, dataset, dir_path):
        pass
