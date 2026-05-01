import os

from torch.utils.data import random_split
import torch
import pandas as pd


class RetinopathyPipeline:
    def run(self, dataset_dir, dataset_class, train_tf, eval_tf):

        csv_path = f"{dataset_dir}/train.csv"
        images_dir = f"{dataset_dir}/train_images"

        df = pd.read_csv(csv_path)
        df["diagnosis"] = df["diagnosis"].astype(int)

        # 🔥 STEP 1: REMOVE INVALID FILES FIRST
        def exists(row):
            file_id = str(row["id_code"])
            for ext in [".png", ".jpg", ".jpeg"]:
                if os.path.exists(f"{images_dir}/{file_id}{ext}"):
                    return True
            return False

        df = df[df.apply(exists, axis=1)].reset_index(drop=True)

        # 🔥 STEP 2: STRATIFIED SPLIT (IMPORTANT FIX)
        train_df, val_df, test_df = self._split(df)

        # 🔥 STEP 3: CREATE DATASETS (CLEAN)
        train_ds = dataset_class(train_df, images_dir, transform=train_tf)
        val_ds = dataset_class(val_df, images_dir, transform=eval_tf)
        test_ds = dataset_class(test_df, images_dir, transform=eval_tf)

        return train_ds, val_ds, test_ds

    def _split(self, df):
        from sklearn.model_selection import train_test_split

        train_df, temp_df = train_test_split(
            df,
            test_size=0.3,
            stratify=df["diagnosis"],
            random_state=42
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            stratify=temp_df["diagnosis"],
            random_state=42
        )

        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True)
        )