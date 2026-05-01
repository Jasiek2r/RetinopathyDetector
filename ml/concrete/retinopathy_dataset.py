import torch
from PIL import Image
from torch.utils.data import Dataset
import os
import pandas as pd


class RetinopathyDataset(Dataset):
    def __init__(self, dataset_dir: str, transform=None, max_images=None, balanced_subset_per_class=None):
        csv_path = os.path.join(dataset_dir, "train.csv")
        images_dir = os.path.join(dataset_dir, "train_images")

        df = pd.read_csv(csv_path)
        df["diagnosis"] = df["diagnosis"].astype(int)

        # remove missing files first
        def exists(row):
            file_id = str(row["id_code"])
            for ext in [".png", ".jpg", ".jpeg"]:
                if os.path.exists(os.path.join(images_dir, file_id + ext)):
                    return True
            return False

        df = df[df.apply(exists, axis=1)].reset_index(drop=True)

        # balancing
        if balanced_subset_per_class is not None:
            df = df.groupby("diagnosis").apply(
                lambda x: x.sample(
                    n=min(len(x), balanced_subset_per_class),
                    random_state=42
                )
            ).reset_index(drop=True)

        # max images
        if max_images is not None:
            df = df.sample(n=max_images, random_state=42).reset_index(drop=True)

        # cache paths
        df["img_path"] = df["id_code"].apply(lambda x: self._resolve_path(images_dir, x))

        df = df.dropna(subset=["img_path"]).reset_index(drop=True)

        self.df = df
        self.images_dir = images_dir

    def _resolve_path(self, file_id):
        file_id = str(file_id)

        for ext in [".png", ".jpg", ".jpeg"]:
            path = os.path.join(self.images_dir, file_id + ext)
            if os.path.exists(path):
                return path
        return None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(row["img_path"]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(row["diagnosis"], dtype=torch.long)

        return img, label