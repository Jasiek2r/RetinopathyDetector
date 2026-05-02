from sklearn.model_selection import train_test_split
from torchvision import transforms

from ml.abstractions.data_pipeline import DataPipeline
from ml.concrete.retinopathy_folder_dataset import RetinopathyFolderDataset

class RetinopathyPipeline(DataPipeline):

    def run(self, dataset, dir_path):
        df = dataset.df

        # --------------------
        # TRANSFORMY
        # --------------------

        tf = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # --------------------
        # DATASETS (3 sztuki)
        # --------------------
        train_ds = RetinopathyFolderDataset(dir_path, split="train", transform=tf)
        val_ds = RetinopathyFolderDataset(dir_path, split="val", transform=tf)
        test_ds = RetinopathyFolderDataset(dir_path, split="test", transform=tf)

        return train_ds, val_ds, test_ds
