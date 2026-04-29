from ml.concrete.retinopathy_dataset import RetinopathyDataset
from services.abstractions.loader_service import LoaderService


class ResearchProjectLoaderService(LoaderService):

    def load_data(self, dataset_dir: str, max_images=None, transform=None):
        train_ds = RetinopathyDataset(
            dataset_dir=dataset_dir,
            split="train",
            transform=transform,
            max_images=max_images
        )

        val_ds = RetinopathyDataset(
            dataset_dir=dataset_dir,
            split="val",
            transform=transform
        )

        test_ds = RetinopathyDataset(
            dataset_dir=dataset_dir,
            split="test",
            transform=transform
        )

        return train_ds, val_ds, test_ds
