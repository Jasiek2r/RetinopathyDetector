import torch
from PIL import Image
from torch.utils.data import Dataset
import os
import pandas as pd

class RetinopathyDataset(Dataset):
    def __init__(self, dataset_dir: str, split: str, transform=None,
                 max_images=None):

        assert split in ["train", "val", "test"], "split must be train/val/test"

        if transform is None:
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.ToTensor()
            ])

        self.transform = transform
        self.max_images = max_images
        self.balanced_subset_per_class = balanced_subset_per_class

        csv_path = os.path.join(dataset_dir, "train.csv")
        images_dir = os.path.join(dataset_dir, split)

        df = pd.read_csv(csv_path)

        # Test set może nie mieć etykiet — obsługujemy oba przypadki
        if "diagnosis" in df.columns:
            df["diagnosis"] = df["diagnosis"].astype(int)


        self.images_dir = images_dir

        # Ograniczenie liczby obrazów
        if max_images is not None:
            df = df.sample(n=max_images, random_state=42).reset_index(drop=True)

        # Sprawdzamy, które obrazy faktycznie istnieją
        valid_rows = []
        for _, row in df.iterrows():
            file_id = str(row["id_code"])
            img_path = self.build_image_path(images_dir, file_id)
            if img_path is not None:
                valid_rows.append(row)

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)
        print(f"[{split}] Załadowano {len(self.df)} poprawnych obrazów.")

    def __len__(self):
        return len(self.df)

    def build_image_path(self, images_dir, file_id):
        for ext in [".jpg", ".jpeg", ".png"]:
            path = os.path.join(images_dir, file_id + ext)
            if os.path.exists(path):
                return path
        return None

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_id = str(row["id_code"])
        img_path = self.build_image_path(self.images_dir, file_id)

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        # Test set może nie mieć etykiet
        if "diagnosis" in row:
            label = torch.tensor(row["diagnosis"], dtype=torch.long)
        else:
            label = -1  # placeholder

        return img, label
