from ml.abstractions.data_pipeline import DataPipeline
from torch.utils.data import random_split
from torchvision import transforms

class RetinopathyPipeline(DataPipeline):
    def __augument__(self, dataset, dir_path):

        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        dataset.transform = transform

        test_size = int(0.15 * len(dataset))
        val_size = int(0.15 * len(dataset))
        train_size = len(dataset) - test_size - val_size

        return random_split(dataset, [train_size, val_size, test_size])

    def run(self, dataset, dir_path):
        return self.__augument__(dataset, dir_path)
