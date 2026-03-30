from ml.abstractions.data_pipeline import DataPipeline
from torch.utils.data import random_split, Subset
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image

from ml.concrete.retinopathy_dataset import RetinopathyDataset


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


class CenterCropCircle:
    def __call__(self, img):
        np_img = np.array(img)
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        gray = cv2.medianBlur(gray, 5)

        # Wykrywanie okręgu metodą Hougha
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=gray.shape[0] // 2,
            param1=50,
            param2=30,
            minRadius=gray.shape[0] // 4,
            maxRadius=gray.shape[0] // 2
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            x, y, r = circles[0]  # bierzemy największy/centralny okrąg

            # Bezpieczny crop (zabezpieczenie przed wyjściem poza obraz)
            y1 = max(0, y - r)
            y2 = min(np_img.shape[0], y + r)
            x1 = max(0, x - r)
            x2 = min(np_img.shape[1], x + r)

            crop = np_img[y1:y2, x1:x2]
            return Image.fromarray(crop)

        # fallback: jeśli Hough nic nie znalazł zwykły center crop
        h, w, _ = np_img.shape
        r = min(h, w) // 2
        cy, cx = h // 2, w // 2
        crop = np_img[cy - r:cy + r, cx - r:cx + r]
        return Image.fromarray(crop)


class RetinopathyPipeline(DataPipeline):
    def __augument__(self, dataset, dir_path):
        # --- PODZIAŁ NA TRAIN/VAL/TEST ---
        test_size = int(0.15 * len(dataset))
        val_size = int(0.15 * len(dataset))
        train_size = len(dataset) - test_size - val_size

        train_subset, val_subset, test_subset = random_split(
            dataset,
            [train_size, val_size, test_size]
        )


        return train_subset, val_subset, test_subset

    def run(self, dataset, dir_path):
        return self.__augument__(dataset, dir_path)
