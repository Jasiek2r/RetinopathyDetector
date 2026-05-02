import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image  # <-- kluczowe: wczytuje TENSOR, nie PIL


class RetinopathyFolderDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None,
                 max_images=None, balanced_subset_per_class=None):

        self.transform = transform
        self.root = os.path.join(root_dir, split)

        if not os.path.isdir(self.root):
            raise ValueError(f"Split folder not found: {self.root}")

        data = []

        for label_str in sorted(os.listdir(self.root)):
            class_dir = os.path.join(self.root, label_str)
            if not os.path.isdir(class_dir):
                continue

            try:
                label = int(label_str)
            except ValueError:
                continue

            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    data.append({
                        "img_path": os.path.join(class_dir, fname),
                        "diagnosis": label
                    })

        df = pd.DataFrame(data)

        if balanced_subset_per_class is not None:
            df = df.groupby("diagnosis", group_keys=False).apply(
                lambda x: x.sample(
                    n=min(len(x), balanced_subset_per_class),
                    random_state=42
                )
            ).reset_index(drop=True)

        if max_images is not None:
            df = df.sample(n=max_images, random_state=42).reset_index(drop=True)

        self.df = df
        print(f"[{split}] Dataset ready: {len(self.df)} images")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- KLUCZOWA ZMIANA: wczytujemy TENSOR, nie PIL ---
        img = read_image(row["img_path"]).float() / 255.0  # [C,H,W] tensor

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(row["diagnosis"], dtype=torch.long)
        return img, label
