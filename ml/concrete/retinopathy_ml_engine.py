import traceback

import timm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from ml.abstractions.ml_engine import MLEngine
import matplotlib.pyplot as plt

from ml.concrete.CORALLoss import CORALLoss


class RetinopathyMLEngine(MLEngine):
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        #self.model = ResearchProjectNet().to(self.device)
        self.model = self.create_model().to(self.device)

    def train(self, train_dataset, val_dataset, batch_size=4, epochs=50):

        criterion = CORALLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-4)

        self.model.train()

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # --- Pobieranie etykiet dla WeightedRandomSampler ---
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

        # --- Cyclical Learning Rate (CLR) ---
        # scheduler = CyclicLR(
        #     optimizer,
        #     base_lr=1e-6,
        #     max_lr=3e-4,
        #     step_size_up=len(train_loader) * 2,
        #     mode="triangular2",
        #     cycle_momentum=False
        # )


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

                    # 🔥 CLR aktualizowany co batch
                    #scheduler.step()

                    running_loss += loss.item()

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
                predicted = torch.sum(
                    (torch.sigmoid(outputs) > 0.5),
                    dim=1
                )
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
                predicted = torch.sum(
                    (torch.sigmoid(outputs) > 0.5),
                    dim=1
                )

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
            num_classes=num_classes - 1
        )

        print(model)  # <-- tylko raz debug

        final_layer = None

        if hasattr(model, "head") and hasattr(model.head, "fc"):
            final_layer = model.head.fc
        elif hasattr(model, "head") and isinstance(model.head, nn.Linear):
            final_layer = model.head
        elif hasattr(model, "classifier"):
            final_layer = model.classifier

        if final_layer is not None:
            nn.init.constant_(final_layer.weight, 0.)

            bias_values = torch.tensor([1.5, 0.5, -0.5, -1.5])
            final_layer.bias.data = bias_values

            print("✓ CORAL bias initialized")

        else:
            print("WARNING: Could not find final classification layer!")

        return model
