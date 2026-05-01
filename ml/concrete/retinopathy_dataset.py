import torch
from PIL import Image
from torch.utils.data import Dataset
import os
import pandas as pd


class RetinopathyDataset(Dataset):
    def __init__(self, dataset_dir: str, transform=None,
                 max_images=None, balanced_subset_per_class=None):

        self.transform = transform

        csv_path = os.path.join(dataset_dir, "train.csv")
        images_dir = os.path.join(dataset_dir, "train_images")
        self.images_dir = images_dir

        df = pd.read_csv(csv_path)
        df["diagnosis"] = df["diagnosis"].astype(int)

        # 🔥 FAST INDEX (NO os.path.exists loop)
        file_index = {
            fname.split(".")[0]: os.path.join(images_dir, fname)
            for fname in os.listdir(images_dir)
            if fname.lower().endswith((".png", ".jpg", ".jpeg"))
        }

        df["img_path"] = df["id_code"].map(file_index)
        df = df.dropna(subset=["img_path"]).reset_index(drop=True)

        # optional balancing
        if balanced_subset_per_class is not None:
            df = df.groupby("diagnosis", group_keys=False).apply(
                lambda x: x.sample(
                    n=min(len(x), balanced_subset_per_class),
                    random_state=42
                )
            ).reset_index(drop=True)

        # optional limit
        if max_images is not None:
            df = df.sample(n=max_images, random_state=42).reset_index(drop=True)

        self.df = df

        print(f"Dataset ready: {len(self.df)} images")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(row["img_path"]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(row["diagnosis"], dtype=torch.long)

        return img, label