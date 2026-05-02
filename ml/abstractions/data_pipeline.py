from abc import abstractmethod


class DataPipeline:
    @abstractmethod
    def run(self, dir_path):
        pass
