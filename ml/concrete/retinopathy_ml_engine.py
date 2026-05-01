import traceback
import timm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler


class RetinopathyMLEngine:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.create_model().to(self.device)

    # =========================
    # CORAL ENCODING (FIXED)
    # =========================
    def encode_labels(self, labels, num_classes=5):
        labels = labels.unsqueeze(1)  # [B, 1]
        thresholds = torch.arange(num_classes - 1, device=labels.device).unsqueeze(0)
        return (labels > thresholds).float()

    # =========================
    # CORAL DECODING
    # =========================
    def decode_predictions(self, outputs):
        probs = torch.sigmoid(outputs)
        return torch.sum(probs >= 0.5, dim=1)

    # =========================
    # MODEL
    # =========================
    def create_model(self, num_classes=5):
        model = timm.create_model(
            "convnext_base",
            pretrained=True,
            num_classes=num_classes - 1  # CORAL => K-1 outputs
        )
        return model

    # =========================
    # TRAIN
    # =========================
    def train(self, train_dataset, val_dataset=None, batch_size=8, epochs=50):

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
        if val_dataset:
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # LOSS (STABILNY CORAL LOSS)
        pos_weight = torch.tensor(class_weights[:-1], dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-4)

        for epoch in range(epochs):
            print(f"\n===== EPOKA {epoch+1}/{epochs} =====")

            self.model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                try:
                    images = images.to(self.device)
                    labels = labels.to(self.device).long()

                    targets = self.encode_labels(labels).to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(images)

                    loss = criterion(outputs, targets)

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                except Exception:
                    traceback.print_exc()
                    raise

            print(f"Loss: {running_loss / len(train_loader):.4f}")

            if val_loader:
                acc = self._validate(val_loader)
                print(f"Val accuracy: {acc:.2f}%")

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
                labels = labels.to(self.device).long()

                outputs = self.model(images)
                preds = self.decode_predictions(outputs).cpu()

                correct += (preds == labels.cpu()).sum().item()
                total += labels.size(0)

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
                labels = labels.to(self.device).long()

                outputs = self.model(images)
                preds = self.decode_predictions(outputs).cpu()

                correct += (preds == labels.cpu()).sum().item()
                total += labels.size(0)

        acc = 100 * correct / total
        print(f"Test accuracy: {acc:.2f}%")

        torch.save(self.model.state_dict(), "model_weights.pth")
        return acc