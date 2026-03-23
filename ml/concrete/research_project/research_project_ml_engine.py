import traceback

import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from ml.abstractions.ml_engine import MLEngine


class ResearchProjectMLEngine(MLEngine):
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        #self.model = ResearchProjectNet().to(self.device)
        self.model = self.create_model().to(self.device)

    def train(self, train_dataset, val_dataset, batch_size=8, epochs=50):

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.0005, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
        self.model.train()

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        for epoch in range(epochs):
            print(f"\n===== EPOKA {epoch + 1} / {epochs} =====")
            loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            running_loss = 0.0

            for batch_idx, (images, labels) in enumerate(loader):
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
            scheduler.step()
            print(f"✓ Epoka {epoch + 1} zakończona — średni loss: {running_loss / len(loader):.4f}")
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
            "efficientnet_b4",
            pretrained=True,
            num_classes=num_classes
        )
        return model