from sklearn.model_selection import train_test_split
from torchvision import transforms

from ml.abstractions.data_pipeline import DataPipeline


class RetinopathyPipeline(DataPipeline):

    def run(self, dataset, dir_path):

        df = dataset.df

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

        DatasetClass = dataset.__class__

        train_ds = DatasetClass.from_df(train_df, dataset)
        val_ds = DatasetClass.from_df(val_df, dataset)
        test_ds = DatasetClass.from_df(test_df, dataset)

        train_tf = transforms.Compose([
            transforms.Resize((384, 384)),

            # augmentacje (realistyczne dla medycznych obrazów)
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),

            # lekka zmienność kadru (OK dla fundus images)
            transforms.RandomResizedCrop(
                384,
                scale=(0.9, 1.0),
                ratio=(0.95, 1.05)
            ),

            transforms.ToTensor(),

            # ImageNet normalization (dla pretrained CNN)
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

        train_ds.transform = train_tf
        val_ds.transform = eval_tf
        test_ds.transform = eval_tf

        return train_ds, val_ds, test_ds