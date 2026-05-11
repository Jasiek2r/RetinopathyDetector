import traceback

import timm
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

from ml.abstractions.ml_engine import MLEngine

from ml.concrete.FocalLoss import FocalLoss
from tqdm import tqdm

from sklearn.metrics import cohen_kappa_score, confusion_matrix

class RetinopathyMLEngine(MLEngine):
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.create_model().to(self.device)

    def train(self, train_dataset, val_dataset, batch_size=8, epochs=30):

        # --- Przygotowanie wag klas ---
        labels = train_dataset.df["diagnosis"].to_numpy()
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum()
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        criterion = FocalLoss(gamma=2.0)
        optimizer = optim.AdamW(self.model.parameters(), lr=3e-5, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        self.model.train()

        # --- Walidacja ---
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                prefetch_factor=4,
                persistent_workers=True
            )

        # --- WeightedRandomSampler ---
        if hasattr(train_dataset, "indices"):
            labels = train_dataset.dataset.df.iloc[train_dataset.indices]["diagnosis"].values
        else:
            labels = train_dataset.df["diagnosis"].values

        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        # --- TRAIN LOADER ---
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=12,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True
        )

        # --- PĘTLA TRENINGOWA ---
        for epoch in range(epochs):
            print(f"\n===== EPOKA {epoch + 1} / {epochs} =====")

            running_loss = 0.0

            # tqdm z procentami i ETA
            progress_bar = tqdm(
                train_loader,
                desc=f"Epoka {epoch+1}/{epochs}",
                ncols=120,
                unit="batch"
            )

            for batch_idx, (images, labels) in enumerate(progress_bar):
                try:
                    # --- MOVE TO GPU ---
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    # --- GPU RESIZE ---
                    images = torch.nn.functional.interpolate(
                        images,
                        size=(256, 256),
                        mode="bilinear",
                        align_corners=False
                    )

                    # --- GPU NORMALIZE ---
                    mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[None, :, None, None]
                    std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[None, :, None, None]
                    images = (images - mean) / std

                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    running_loss += loss.item()

                    # aktualizacja opisu progress bara
                    avg_loss = running_loss / (batch_idx + 1)
                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}"
                    })

                except Exception as e:
                    print(f"\n*** ERROR in batch {batch_idx} ***")
                    print("Exception:", e)
                    traceback.print_exc()
                    raise

            print(f"✓ Epoka {epoch + 1} zakończona — średni loss: {running_loss / len(train_loader):.4f}")

            if val_dataset is not None:
                acc, qwk, cm = self._validate(val_loader)
                print(f"Validation accuracy: {acc:.2f}%")
                print(f"Validation QWK: {qwk:.2f}%")

                np.savetxt(f"confusion_epoch_{epoch + 1}.txt", cm, fmt="%d")

    def _validate(self, loader):
        self.model.eval()
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                images = torch.nn.functional.interpolate(
                    images,
                    size=(256, 256),
                    mode="bilinear",
                    align_corners=False
                )

                mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[None, :, None, None]
                std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[None, :, None, None]
                images = (images - mean) / std

                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)

                all_preds.append(predicted.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)

        acc = 100 * correct / total
        qwk = cohen_kappa_score(labels, preds, weights="quadratic") * 100.0
        cm = confusion_matrix(labels, preds)

        self.model.train()
        return acc, qwk, cm

    def test(self, dataset, batch_size=8):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                images = torch.nn.functional.interpolate(
                    images,
                    size=(256, 256),
                    mode="bilinear",
                    align_corners=False
                )

                mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[None, :, None, None]
                std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[None, :, None, None]
                images = (images - mean) / std

                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total * 100
        print(f"Test accuracy: {accuracy:.2f}%")
        torch.save(self.model.state_dict(), "model_weights.pth")
        return accuracy

    def create_model(self, num_classes=5):
        model = timm.create_model(
            "convnext_small",
            pretrained=True,
            num_classes=num_classes
        )

        # --- DODAJEMY DROPOUT DO CLASSIFIERA ---
        model.classifier = torch.nn.Sequential(
            timm.layers.LayerNorm2d(768, eps=1e-6),
            torch.nn.Flatten(1),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(768, num_classes)
        )

        return model
