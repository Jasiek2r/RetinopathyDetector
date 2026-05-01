import torch
from PIL import Image
from torch.utils.data import Dataset
import os
import pandas as pd


class RetinopathyDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: str,
        transform=None
    ):
        self.transform = transform
        self.images_dir = images_dir

        df = df.copy()
        df["diagnosis"] = df["diagnosis"].astype(int)

        # 🔥 PRECOMPUTE IMAGE PATHS (CRITICAL FIX)
        df["img_path"] = df["id_code"].apply(self._resolve_path)

        # drop missing files
        df = df.dropna(subset=["img_path"]).reset_index(drop=True)

        self.df = df
        print(f"Loaded dataset: {len(self.df)} images")

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