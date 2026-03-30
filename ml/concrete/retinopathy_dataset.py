import torch
from PIL import Image
from torch.utils.data import Dataset
import os
import pandas as pd
from torchvision import transforms


class RetinopathyDataset(Dataset):
    def __init__(self, dataset_dir: str, transform, max_images=None, balanced_subset_per_class=None):

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        csv_path = os.path.join(dataset_dir, "train.csv")
        images_dir = os.path.join(dataset_dir, "train_images")

        df = pd.read_csv(csv_path)
        df["diagnosis"] = df["diagnosis"].astype(int)

        if balanced_subset_per_class is not None:
            df = df.groupby("diagnosis").apply(
                lambda x: x.sample(
                    n=min(len(x), balanced_subset_per_class),
                    replace=False,  # NIE duplikujemy!
                    random_state=42
                )
            ).reset_index(drop=True)

        self.images_dir = images_dir

        if max_images is not None:
            df = df.sample(n=max_images, random_state=42).reset_index(drop=True)

        # Filtr istniejących plików
        valid_rows = []
        for _, row in df.iterrows():
            file_id = str(row["id_code"])

            img_path = self.build_image_path(images_dir, file_id)
            if img_path is not None:
                valid_rows.append(row)

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)
        print(f"Załadowano {len(self.df)} poprawnych obrazów.")

    def __len__(self):
        return len(self.df)

    def build_image_path(self, images_dir, file_id):
        # jeśli ma rozszerzenie → użyj jak jest
        if "." in file_id:
            return os.path.join(images_dir, file_id)

        # jeśli nie ma → sprawdź różne rozszerzenia
        for ext in [".png", ".jpeg", ".jpg"]:
            path = os.path.join(images_dir, file_id + ext)
            if os.path.exists(path):
                return path

        return None

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_id = str(row["id_code"])
        img_path = self.build_image_path(self.images_dir, file_id)
        if img_path is None:
            raise FileNotFoundError(file_id)

        img = Image.open(img_path)
        img = self.transform(img)


        label = torch.tensor(row["diagnosis"], dtype=torch.long)
        return img, label
