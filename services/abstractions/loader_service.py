from __future__ import annotations
from abc import abstractmethod


class LoaderService:

    @abstractmethod
    def load_data(self, dataset_dir: str, max_images: int | None):
        pass
