import torch
from PIL import Image
from torchvision import transforms
import os
import pandas as pd

from services.abstractions.loader_service import LoaderService


class ResearchProjectLoaderService(LoaderService):

    def load_data(self, dataset_dir: str, max_images=None):
        csv_path = os.path.join(dataset_dir, "train.csv")
        images_dir = os.path.join(dataset_dir, "train_images")

        df = pd.read_csv(csv_path)

        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

        xs = []
        ys = []

        for i, row in df.iterrows():

            if max_images is not None and i >= max_images:
                break

            if i % 10 == 0:
                total = max_images if max_images is not None else len(df)
                print(f"Loaded {i}/{total} images")

            img_path = os.path.join(images_dir, row["id_code"] + ".png")

            if not os.path.exists(img_path):
                print(f"Warning: file not found: {img_path}")
                continue

            img = Image.open(img_path).convert("RGB")
            xs.append(transform(img))
            ys.append(row["diagnosis"])

        return torch.stack(xs), torch.tensor(ys)
