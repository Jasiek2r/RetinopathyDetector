import torch
from PIL import Image
from torch.utils.data import Dataset
import os
import pandas as pd
from pathlib import Path


class RetinopathyDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        transform=None,
        max_images=None,
        balanced_subset_per_class=None
    ):
        self.transform = transform

        csv_path = os.path.join(dataset_dir, "train.csv")
        images_dir = os.path.join(dataset_dir, "train_images")
        self.images_dir = images_dir

        # =========================
        # 1. LOAD CSV
        # =========================
        df = pd.read_csv(csv_path)
        df["diagnosis"] = df["diagnosis"].astype(int)

        # =========================
        # 2. FAST IMAGE INDEXING (CRITICAL FIX)
        # =========================
        print("Indexing images...")

        file_index = {
            p.stem: str(p)
            for p in Path(images_dir).glob("*")
            if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
        }

        df["img_path"] = df["id_code"].map(file_index)

        # drop missing
        df = df.dropna(subset=["img_path"]).reset_index(drop=True)

        # =========================
        # 3. BALANCING (SAFE)
        # =========================
        if balanced_subset_per_class is not None:
            df = (
                df.groupby("diagnosis", group_keys=False)
                .apply(lambda x: x.sample(
                    n=min(len(x), balanced_subset_per_class),
                    random_state=42
                ))
                .reset_index(drop=True)
            )

        # =========================
        # 4. LIMIT DATASET SIZE
        # =========================
        if max_images is not None:
            df = df.sample(n=max_images, random_state=42).reset_index(drop=True)

        self.df = df

        print(f"Dataset ready: {len(self.df)} images")

    # =========================
    # BASIC METHODS
    # =========================
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(row["img_path"]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(row["diagnosis"], dtype=torch.long)

        return img, label