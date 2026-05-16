import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from ml.abstractions.ml_engine import MLEngine
from ml.concrete.QWK import QWK
from ml.concrete.CORALLoss import CORALLoss
from ml.concrete.simple_cnn import SimpleCNN
import matplotlib.pyplot as plt


class RetinopathyMLEngine(MLEngine):
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleCNN().to(self.device)

    # ---------------- TRAIN ----------------
    def train(self, train_dataset, val_dataset=None, batch_size=4, epochs=50):

        criterion = CORALLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-4)

        # ----- sampler (class imbalance) -----
        labels = train_dataset.df["diagnosis"].values
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]

        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

        val_loader = None
        if val_dataset:
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        train_losses = []
        val_metrics = []

        best_mae = float("inf")

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)
            train_losses.append(train_loss)

            print(f"\nEpoch {epoch + 1}")
            print(f"Train loss: {train_loss:.4f}")

            if val_loader:
                mae, qoe, acc = self._validate(val_loader, criterion)
                val_metrics.append((mae, qoe, acc))
                print(f"Val MAE: {mae:.4f} | QOE: {qoe:.4f} | ACC: {acc:.4f}")

                if mae < best_mae:
                    best_mae = mae
                    torch.save(self.model.state_dict(), "best_model.pth")
                    print(f"🔥 New best model saved! MAE improved to {best_mae:.4f}")

        # --- plotting ---
        val_mae = [x[0] for x in val_metrics] if val_metrics else []
        val_qoe = [x[1] for x in val_metrics] if val_metrics else []
        val_acc = [x[2] for x in val_metrics] if val_metrics else []

        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Train Loss")

        if val_qoe:
            plt.plot(val_qoe, label="Validation Loss (QOE)")
        if val_acc:
            plt.plot(val_acc, label="Validation Accuracy")

        plt.xlabel("Epoch")
        plt.ylabel("Metric Value")
        plt.title("Training Progress")
        plt.legend()
        plt.grid(True)

        plt.savefig("training_plot.png", bbox_inches="tight")
        plt.show()

        print("Training plot saved as training_plot.png")

        return train_losses, val_metrics

    # ---------------- VALIDATION ----------------
    def _validate(self, loader, criterion):
        self.model.eval()

        total_loss = 0.0
        total_mae = 0.0
        total_acc = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                preds = torch.sigmoid(outputs).sum(dim=1)

                # MAE
                total_mae += torch.abs(preds - labels.float()).sum().item()

                # ACCURACY
                pred_classes = preds.round().long().clamp(0, 4)
                total_acc += (pred_classes == labels).sum().item()

                total += labels.size(0)

        mae = total_mae / total
        acc = total_acc / total
        qoe = total_loss / len(loader)

        self.model.train()
        return mae, qoe, acc

    # ---------------- TEST ----------------
    def test(self, dataset, batch_size=8):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()

        total_mae = 0.0
        total_acc = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                preds = torch.sigmoid(outputs).sum(dim=1)

                # MAE
                total_mae += torch.abs(preds - labels.float()).sum().item()

                # ACCURACY
                pred_classes = preds.round().long().clamp(0, 4)
                total_acc += (pred_classes == labels).sum().item()

                total += labels.size(0)

        mae = total_mae / total
        acc = total_acc / total

        print(f"Test MAE: {mae:.4f} | ACC: {acc:.4f}")

        torch.save(self.model.state_dict(), "model_weights.pth")
        return mae, acc

    def full_evaluation(self, train_dataset, val_dataset, test_dataset):
        self.model.eval()
        device = self.device
        criterion = CORALLoss()

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        def evaluate(loader):
            total = 0
            total_acc = 0
            total_loss = 0.0

            all_labels = []
            all_preds = []

            with torch.no_grad():
                for images, labels in loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = self.model(images)

                    # LOSS (CORAL)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item() * labels.size(0)

                    # PREDYKCJA
                    preds = torch.sigmoid(outputs).sum(dim=1)
                    pred_classes = preds.round().long().clamp(0, 4)

                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(pred_classes.cpu().numpy())

                    # ACC
                    total_acc += (pred_classes == labels).sum().item()
                    total += labels.size(0)

            avg_loss = total_loss / total
            acc = total_acc / total

            # QWK w procentach
            qwk = QWK.perform_measurement(all_labels, all_preds)

            return avg_loss, acc, qwk

        # Obliczenia
        train_loss, train_acc, train_qwk = evaluate(train_loader)
        val_loss, val_acc, val_qwk = evaluate(val_loader)
        test_loss, test_acc, test_qwk = evaluate(test_loader)

        print("===== FULL EVALUATION =====")
        print(f"Train LOSS: {train_loss:.4f} | ACC: {train_acc:.4f} | QWK: {train_qwk:.2f}%")
        print(f"Val   LOSS: {val_loss:.4f} | ACC: {val_acc:.4f} | QWK: {val_qwk:.2f}%")
        print(f"Test  LOSS: {test_loss:.4f} | ACC: {test_acc:.4f} | QWK: {test_qwk:.2f}%")
        print("============================")

        return {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_qwk": train_qwk,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_qwk": val_qwk,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_qwk": test_qwk
        }


