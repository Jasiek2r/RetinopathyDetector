import torch

from torchvision.transforms import v2 as T

from ml.abstractions.data_pipeline import DataPipeline
from ml.concrete.retinopathy_folder_dataset import RetinopathyFolderDataset

class RetinopathyPipeline(DataPipeline):

    def run(self, dir_path, max_images):
        train_tf = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.GaussianBlur(kernel_size=3),
            T.RandomPerspective(distortion_scale=0.2, p=0.5),
            T.RandomErasing(p=0.25),
        ])

        eval_tf = T.Compose([
            T.ToDtype(torch.float32, scale=True),
        ])

        train_ds = RetinopathyFolderDataset(dir_path, "train", train_tf, max_images=max_images)
        val_ds = RetinopathyFolderDataset(dir_path, "val", eval_tf, max_images=max_images)
        test_ds = RetinopathyFolderDataset(dir_path, "test", eval_tf, max_images=max_images)

        return train_ds, val_ds, test_ds
