from ml.abstractions.data_pipeline import DataPipeline
from torch.utils.data import random_split
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image


class CLAHETransform:
    def __call__(self, img):
        np_img = np.array(img)
        lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return Image.fromarray(final)


class RetinopathyPipeline(DataPipeline):
    def __augument__(self, dataset, dir_path):

        train_transform = transforms.Compose([
            CLAHETransform(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        val_test_transform = transforms.Compose([
            CLAHETransform(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        test_size = int(0.15 * len(dataset))
        val_size = int(0.15 * len(dataset))
        train_size = len(dataset) - test_size - val_size

        train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

        train_ds.dataset.transform = train_transform
        val_ds.dataset.transform = val_test_transform
        test_ds.dataset.transform = val_test_transform

        return train_ds, val_ds, test_ds

