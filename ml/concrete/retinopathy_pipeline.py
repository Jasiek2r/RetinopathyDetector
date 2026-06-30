import torch

from torchvision.transforms import v2 as T

from ml.abstractions.data_pipeline import DataPipeline
from ml.concrete.retinopathy_folder_dataset import RetinopathyFolderDataset

class RetinopathyPipeline(DataPipeline):

    def run(self, dir_path, max_images):

        train_tf = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.RandomRotation(degrees=5),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.1),
            T.RandomAffine(
                degrees=5,
                translate=(0.02, 0.02),  # 2% przesunięcia
                scale=(0.95, 1.05)
            )
        ])

        eval_tf = T.Compose([
            T.ToDtype(torch.float32, scale=True),
        ])

        train_ds = RetinopathyFolderDataset(dir_path, "train", train_tf, max_images=max_images)
        val_ds = RetinopathyFolderDataset(dir_path, "val", eval_tf, max_images=max_images)
        test_ds = RetinopathyFolderDataset(dir_path, "test", eval_tf, max_images=max_images)

        return train_ds, val_ds, test_ds
