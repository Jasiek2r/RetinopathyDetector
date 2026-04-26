import torch

from ml.abstractions.data_pipeline import DataPipeline
from torch.utils.data import random_split
from torchvision import transforms
from ml.concrete.retinopathy_dataset import RetinopathyDataset


class RetinopathyPipeline(DataPipeline):

    def __init__(self):
        # --- AUGMENTACJE TRENINGOWE ---
        self.train_tf = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.RandomResizedCrop(384, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # --- AUGMENTACJE WALIDACYJNE / TESTOWE ---
        self.eval_tf = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def run(self, dataset, dir_path):
        # --- PODZIAŁ NA INDEKSY ---
        test_size = int(0.15 * len(dataset))
        val_size = int(0.15 * len(dataset))
        train_size = len(dataset) - test_size - val_size

        train_subset, val_subset, test_subset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        # --- TWORZENIE TRZECH OSOBNYCH DATASETÓW ---
        train_ds = RetinopathyDataset(
            dir_path,
            transform=self.train_tf
        )
        val_ds = RetinopathyDataset(
            dir_path,
            transform=self.eval_tf
        )
        test_ds = RetinopathyDataset(
            dir_path,
            transform=self.eval_tf
        )

        # --- PODMIANA DF NA ODPOWIEDNIE PODZBIORY ---
        train_ds.df = dataset.df.iloc[train_subset.indices].reset_index(drop=True)
        val_ds.df = dataset.df.iloc[val_subset.indices].reset_index(drop=True)
        test_ds.df = dataset.df.iloc[test_subset.indices].reset_index(drop=True)

        return train_ds, val_ds, test_ds
