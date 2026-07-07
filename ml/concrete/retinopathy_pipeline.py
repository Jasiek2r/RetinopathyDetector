import torch

from torchvision.transforms import v2 as T

from ml.abstractions.data_pipeline import DataPipeline
from ml.concrete.retinopathy_folder_dataset import RetinopathyFolderDataset

class RetinopathyPipeline(DataPipeline):

    def run(self, dir_path, max_images):
        train_tf = T.Compose([
            T.ToImage(),  # jeśli wejście to tensor, zamień na tv_tensors.Image
            T.ToDtype(torch.float32, scale=True),

            T.CenterCrop(224),
            # fotometria: symulacja różnych kamer / ekspozycji
            T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.02
            ),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),

            # delikatna zmiana gamma (ciemniej/jaśniej)
            T.RandomApply([
                T.Lambda(lambda x: x ** torch.empty(1).uniform_(0.9, 1.1).item())
            ], p=0.3),
        ])

        eval_tf = T.Compose([
            T.ToDtype(torch.float32, scale=True),
        ])

        train_ds = RetinopathyFolderDataset(dir_path, "train", train_tf, max_images=max_images)
        val_ds = RetinopathyFolderDataset(dir_path, "val", eval_tf, max_images=max_images)
        test_ds = RetinopathyFolderDataset(dir_path, "test", eval_tf, max_images=max_images)

        return train_ds, val_ds, test_ds
