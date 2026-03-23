import torch
from PIL import Image
from torch.utils.data import Dataset
import os
import pandas as pd
from torchvision import transforms
import numpy as np

class ResearchProjectDataset(Dataset):
    def __init__(self, dataset_dir: str, max_images=None, balanced_subset_per_class=None):
        csv_path = os.path.join(dataset_dir, "train.csv")
        images_dir = os.path.join(dataset_dir, "train_images")

        df = pd.read_csv(csv_path)
        # Balanced subset: po X obrazów z każdej klasy
        if balanced_subset_per_class is not None:
            df = df.groupby("diagnosis").apply(
                lambda x: x.sample(
                    n=min(len(x), balanced_subset_per_class),
                    replace=True,  # jeśli klasa ma mniej niż X, duplikujemy
                    random_state=42
                )
            ).reset_index(drop=True)
        self.images_dir = images_dir

        if max_images is not None:
            df = df.sample(n=max_images, random_state=42).reset_index(drop=True)

        valid_rows = []
        for _, row in df.iterrows():
            file_id = str(row["id_code"])

            # Jeśli nazwa ma rozszerzenie → używamy jej bez zmian
            if file_id.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(images_dir, file_id)
                if os.path.exists(img_path):
                    valid_rows.append(row)
            else:
                # Jeśli nie ma rozszerzenia → zawsze .png
                img_path = os.path.join(images_dir, file_id + ".png")
                if os.path.exists(img_path):
                    valid_rows.append(row)

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

        print(f"Załadowano {len(self.df)} poprawnych obrazów (odfiltrowano brakujące).")

        self.transform = transforms.Compose([
            CLAHETransform(),
            CenterCropCircle(),
            transforms.Resize((224, 224)),

            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2
            ),
            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.25, 0.25, 0.25]
            )
        ])



    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_id = str(row["id_code"])

        # Jeśli nazwa ma rozszerzenie → używamy jej bez zmian
        if file_id.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(self.images_dir, file_id)
        else:
            # Jeśli nie ma rozszerzenia → zawsze .png
            img_path = os.path.join(self.images_dir, file_id + ".png")

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Plik nie istnieje: {img_path}")

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        label = torch.tensor(row["diagnosis"], dtype=torch.long)
        return img, label



import numpy as np
from PIL import Image, ImageOps

class CLAHETransform:
    def __call__(self, img):
        # PIL → numpy
        np_img = np.array(img)

        # Konwersja do YCbCr
        ycbcr = Image.fromarray(np_img).convert("YCbCr")
        y, cb, cr = ycbcr.split()

        # CLAHE na kanale Y (jasność)
        y = ImageOps.equalize(y)

        # Połączenie z powrotem
        ycbcr = Image.merge("YCbCr", (y, cb, cr))
        rgb = ycbcr.convert("RGB")

        return rgb



class CenterCropCircle:
    def __call__(self, img):
        np_img = np.array(img)
        h, w, _ = np_img.shape
        r = min(h, w) // 2
        y, x = h // 2, w // 2

        Y, X = np.ogrid[:h, :w]
        mask = (X - x)**2 + (Y - y)**2 <= r*r

        out = np.zeros_like(np_img)
        out[mask] = np_img[mask]

        return Image.fromarray(out)


