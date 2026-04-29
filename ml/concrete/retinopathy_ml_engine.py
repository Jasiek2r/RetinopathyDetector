import traceback
import timm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from ml.abstractions.ml_engine import MLEngine
from ml.concrete.simple_cnn import SimpleCNN


class RetinopathyMLEngine(MLEngine):
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleCNN().to(self.device)
        # self.model = self.create_model().to(self.device)

    def train(self, train_dataset, val_dataset, batch_size=4, epochs=50):

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-4)

        self.model.train()

        # --- VALIDATION LOADER ---
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )

        # --- WEIGHTED SAMPLER (tylko dla train) ---
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
            num_workers=0
        )

        # --- TRAINING LOOP ---
        for epoch in range(epochs):
            print(f"\n===== EPOKA {epoch + 1} / {epochs} =====")

            running_loss = 0.0

            for batch_idx, (images, labels) in enumerate(train_loader):
                try:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                except Exception as e:
                    print(f"\n*** ERROR in batch {batch_idx} ***")
                    print("Exception:", e)
                    traceback.print_exc()
                    raise

            print(f"✓ Epoka {epoch + 1} zakończona — średni loss: {running_loss / len(train_loader):.4f}")

            if val_loader is not None:
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

                # test set może mieć label = -1
                mask = labels >= 0
                total += mask.sum().item()
                correct += ((predicted == labels) & mask).sum().item()

        accuracy = correct / total * 100 if total > 0 else 0
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
