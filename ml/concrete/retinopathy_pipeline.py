from ml.abstractions.data_pipeline import DataPipeline
from torch.utils.data import random_split
from torchvision import transforms


class RetinopathyPipeline(DataPipeline):

    def __init__(self):
        # --- AUGMENTACJE TRENINGOWE ---
        self.train_tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # --- AUGMENTACJE WALIDACYJNE / TESTOWE ---
        self.eval_tf = transforms.Compose([
            transforms.Resize((224, 224)),
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
