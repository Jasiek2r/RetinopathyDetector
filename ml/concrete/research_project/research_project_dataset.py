import torch
from PIL import Image
from torch.utils.data import Dataset
import os
import pandas as pd
from torchvision import transforms
import numpy as np
import cv2


class CLAHETransform:
    def __call__(self, img):
        np_img = np.array(img)
        lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return Image.fromarray(final)

class CenterCropCircle:
    def __call__(self, img):
        np_img = np.array(img)
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        gray = cv2.medianBlur(gray, 5)

        # Wykrywanie okręgu metodą Hougha
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=gray.shape[0]//2,
            param1=50,
            param2=30,
            minRadius=gray.shape[0]//4,
            maxRadius=gray.shape[0]//2
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            x, y, r = circles[0]  # bierzemy największy/centralny okrąg

            # Bezpieczny crop (zabezpieczenie przed wyjściem poza obraz)
            y1 = max(0, y - r)
            y2 = min(np_img.shape[0], y + r)
            x1 = max(0, x - r)
            x2 = min(np_img.shape[1], x + r)

            crop = np_img[y1:y2, x1:x2]
            return Image.fromarray(crop)

        # fallback: jeśli Hough nic nie znalazł zwykły center crop
        h, w, _ = np_img.shape
        r = min(h, w) // 2
        cy, cx = h // 2, w // 2
        crop = np_img[cy - r:cy + r, cx - r:cx + r]
        return Image.fromarray(crop)


class ResearchProjectDataset(Dataset):
    def __init__(self, dataset_dir: str, max_images=None, balanced_subset_per_class=None):
        csv_path = os.path.join(dataset_dir, "train.csv")
        images_dir = os.path.join(dataset_dir, "train_images")

        df = pd.read_csv(csv_path)

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

            # Obsługa rozszerzeń
            if "." in file_id:
                img_path = os.path.join(images_dir, file_id)
            else:
                img_path = os.path.join(images_dir, file_id + ".png")
            if os.path.exists(img_path):
                valid_rows.append(row)

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)
        print(f"Załadowano {len(self.df)} poprawnych obrazów.")

        self.transform = transforms.Compose([
            CLAHETransform(),
            CenterCropCircle(),
            transforms.Resize((380, 380)),

            # delikatne augmentacje
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1
            ),
            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),

            # ImageNet mean/std (dla pretrained)
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_id = str(row["id_code"])
        if "." in file_id:
            img_path = os.path.join(self.images_dir, file_id)
        else:
            img_path = os.path.join(self.images_dir, file_id + ".png")

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        label = torch.tensor(row["diagnosis"], dtype=torch.long)
        return img, label
