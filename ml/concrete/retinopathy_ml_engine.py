import traceback

import timm
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from ml.abstractions.ml_engine import MLEngine

from ml.concrete.FocalLoss import FocalLoss
from tqdm import tqdm

class RetinopathyMLEngine(MLEngine):
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.create_model().to(self.device)

    def train(self, train_dataset, val_dataset, batch_size=8, epochs=50):

        # --- Przygotowanie wag klas ---
        labels = train_dataset.df["diagnosis"].to_numpy()
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum()
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        criterion = FocalLoss(gamma=2.0)
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-5, weight_decay=1e-4)

        self.model.train()

        # --- Walidacja ---
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

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
            num_workers=4
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
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

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
                acc = self._validate(val_loader)
                print(f"Validation accuracy: {acc:.2f}%")

    def _validate(self, loader):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        self.model.train()
        return 100 * correct / total

    def test(self, dataset, batch_size=8):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

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
            "convnext_base",
            pretrained=True,
            num_classes=num_classes
        )
        return model