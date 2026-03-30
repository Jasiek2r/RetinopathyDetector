from ml.concrete.retinopathy_dataset import RetinopathyDataset
from services.abstractions.loader_service import LoaderService


class ResearchProjectLoaderService(LoaderService):

    def load_data(self, dataset_dir: str, max_images=None):
        return RetinopathyDataset(dataset_dir=dataset_dir, transform=None)
