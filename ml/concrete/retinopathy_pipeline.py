from ml.abstractions.data_pipeline import DataPipeline
from torch.utils.data import random_split
from torchvision import transforms

from ml.abstractions.data_pipeline import DataPipeline
from torch.utils.data import random_split
from torchvision import transforms


class RetinopathyPipeline(DataPipeline):

    def __init__(self):
        # --- AUGMENTACJE TRENINGOWE (SOTA-lite, ale lekkie) ---
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

        # --- AUGMENTACJE WALIDACYJNE / TESTOWE (spójne z train) ---
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
        # --- PODZIAŁ NA TRAIN/VAL/TEST ---
        test_size = int(0.15 * len(dataset))
        val_size = int(0.15 * len(dataset))
        train_size = len(dataset) - test_size - val_size

        train_subset, val_subset, test_subset = random_split(
            dataset,
            [train_size, val_size, test_size]
        )

        # --- PODMIANA TRANSFORMACJI W SUBSETACH ---
        train_subset.dataset.transform = self.train_tf
        val_subset.dataset.transform = self.eval_tf
        test_subset.dataset.transform = self.eval_tf

        return train_subset, val_subset, test_subset
