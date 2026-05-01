import os

from sklearn.model_selection import train_test_split
from torch.utils.data import random_split
import torch
import pandas as pd

from ml.abstractions.data_pipeline import DataPipeline


class RetinopathyPipeline(DataPipeline):

    def run(self, dataset, dir_path):

        df = dataset.df if hasattr(dataset, "df") else None

        if df is None:
            raise ValueError("Dataset must expose df attribute")

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

        train_ds = dataset.__class__(dir_path, transform=dataset.transform)
        val_ds = dataset.__class__(dir_path, transform=dataset.transform)
        test_ds = dataset.__class__(dir_path, transform=dataset.transform)

        train_ds.df = train_df.reset_index(drop=True)
        val_ds.df = val_df.reset_index(drop=True)
        test_ds.df = test_df.reset_index(drop=True)

        return train_ds, val_ds, test_ds

