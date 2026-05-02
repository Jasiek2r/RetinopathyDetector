from abc import abstractmethod


class DataPipeline:
    @abstractmethod
    def run(self, dir_path, max_images):
        pass
