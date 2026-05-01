import traceback
import timm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from ml.abstractions.ml_engine import MLEngine
from ml.concrete.CORALLoss import CORALLoss


class RetinopathyMLEngine(MLEngine):
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.create_model().to(self.device)

    # =========================
    # CORAL ENCODING
    # =========================
    def encode_labels(self, labels, num_classes=5):
        # CORAL: 0–4 -> 4 binary thresholds
        return torch.stack([
            torch.tensor([1 if i < label else 0 for i in range(num_classes - 1)])
            for label in labels
        ]).float()

    # =========================
    # TRAIN
    # =========================
    def train(self, train_dataset, val_dataset, batch_size=4, epochs=50):

        criterion = CORALLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-4)

        # ---------- labels ----------
        raw_labels = train_dataset.df["diagnosis"].values

        class_counts = np.bincount(raw_labels)
        class_weights = 1.0 / (class_counts + 1e-6)

        sample_weights = class_weights[raw_labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # =========================
        # TRAIN LOOP
        # =========================
        for epoch in range(epochs):
            print(f"\n===== EPOKA {epoch + 1} / {epochs} =====")

            self.model.train()
            running_loss = 0.0

            for batch_idx, (images, labels) in enumerate(train_loader):
                try:
                    images = images.to(self.device)

                    # CORAL encoding (KLUCZ FIX)
                    labels = self.encode_labels(labels.numpy()).to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(images)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                except Exception as e:
                    print(f"\n*** ERROR batch {batch_idx} ***")
                    traceback.print_exc()
                    raise

            print(f"✓ Loss: {running_loss / len(train_loader):.4f}")

            if val_loader:
                acc = self._validate(val_loader)
                print(f"Validation accuracy: {acc:.2f}%")

    # =========================
    # CORAL PREDICTION
    # =========================
    def decode_predictions(self, outputs):
        probs = torch.sigmoid(outputs)
        return (probs > 0.5).sum(dim=1)

    # =========================
    # VALIDATION
    # =========================
    def _validate(self, loader):
        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)

                outputs = self.model(images)

                predicted = self.decode_predictions(outputs).cpu()
                labels = labels.cpu()

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        self.model.train()
        return 100 * correct / total

    # =========================
    # TEST
    # =========================
    def test(self, dataset, batch_size=8):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)

                outputs = self.model(images)
                predicted = self.decode_predictions(outputs).cpu()

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Test accuracy: {acc:.2f}%")

        torch.save(self.model.state_dict(), "model_weights.pth")
        return acc

    # =========================
    # MODEL
    # =========================
    def create_model(self, num_classes=5):

        model = timm.create_model(
            "convnext_base",
            pretrained=True,
            num_classes=num_classes - 1  # CORAL
        )

        print(model)

        # stabilizacja (ważne dla CORAL)
        final_layer = model.head

        if isinstance(final_layer, nn.Linear):
            nn.init.normal_(final_layer.weight, std=0.01)
            nn.init.constant_(final_layer.bias, 0.0)

        print("✓ CORAL model initialized correctly")

        return model