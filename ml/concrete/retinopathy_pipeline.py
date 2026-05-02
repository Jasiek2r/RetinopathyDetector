from sklearn.model_selection import train_test_split
from torchvision import transforms

from ml.abstractions.data_pipeline import DataPipeline
from ml.concrete.retinopathy_folder_dataset import RetinopathyFolderDataset

class RetinopathyPipeline(DataPipeline):

    def run(self, dir_path, max_images):

        tf = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        train_ds = RetinopathyFolderDataset(dir_path, "train", tf, max_images=max_images)
        val_ds = RetinopathyFolderDataset(dir_path, "val", tf, max_images=max_images)
        test_ds = RetinopathyFolderDataset(dir_path, "test", tf, max_images=max_images)

        return train_ds, val_ds, test_ds
