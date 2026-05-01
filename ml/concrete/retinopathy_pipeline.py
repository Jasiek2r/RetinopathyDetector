from sklearn.model_selection import train_test_split
from torchvision import transforms

from ml.concrete.retinopathy_dataset import RetinopathyDataset


class RetinopathyPipeline:

    def run(self, dataset, dir_path):
        df = dataset.df

        # --------------------
        # SPLIT (tylko indeksy/DF)
        # --------------------
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

        # --------------------
        # TRANSFORMY
        # --------------------
        train_tf = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(384, scale=(0.9, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        eval_tf = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # --------------------
        # DATASETS (3 sztuki)
        # --------------------
        train_ds = RetinopathyDataset.from_df(train_df, dir_path, train_tf)
        val_ds = RetinopathyDataset.from_df(val_df, dir_path, eval_tf)
        test_ds = RetinopathyDataset.from_df(test_df, dir_path, eval_tf)

        return train_ds, val_ds, test_ds
