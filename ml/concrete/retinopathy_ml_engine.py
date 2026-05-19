import timm
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
        # self.model = self.create_model()

    # ---------------- TRAIN ----------------
    def train(self, train_dataset, val_dataset=None, batch_size=4, epochs=50):

        criterion = CORALLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-4)

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
                mae, qoe, acc, qwk = self._validate(val_loader, criterion)
                val_metrics.append((mae, qoe, acc, qwk))
                print(f"Val MAE: {mae:.4f} | QOE: {qoe:.4f} | ACC: {acc:.4f} | QWK : {qwk:.4f}")

                if mae < best_mae:
                    best_mae = mae
                    torch.save(self.model.state_dict(), "best_model.pth")
                    print(f"🔥 New best model saved! MAE improved to {best_mae:.4f}")
        # --- plotting (Accuracy + QWK) ---
        if val_metrics:
            val_acc = [x[2] for x in val_metrics]
            val_qwk = [x[3] for x in val_metrics]

            plt.figure(figsize=(12, 7))

            plt.plot(val_acc, label="Validation Accuracy", linewidth=2)
            plt.plot(val_qwk, label="Validation QWK", linewidth=2)

            plt.xlabel("Epoch")
            plt.ylabel("Metric Value")
            plt.title("Validation Metrics (Accuracy & QWK)")
            plt.legend()
            plt.grid(True)

            plt.savefig("training_plot.png", bbox_inches="tight")
            plt.show()

            print("Training plot saved as training_plot.png")

        return train_losses, val_metrics

    # ---------------- HACK PREDICTION ----------------
    def _coral_predict_classes(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        HACK: sum(sigmoid(outputs)) → round → clamp
        """
        probs = torch.sigmoid(outputs)          # (B, K)
        scores = probs.sum(dim=1)               # (B,)
        pred_classes = scores.round().long()    # zaokrąglenie
        pred_classes = pred_classes.clamp(0, 4) # zakres 0–4
        return pred_classes

    # ---------------- VALIDATION ----------------
    def _validate(self, loader, criterion):
        self.model.eval()

        total_loss = 0.0
        total_mae = 0.0
        total_acc = 0
        total = 0

        all_labels = []
        all_preds = []

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                pred_classes = self._coral_predict_classes(outputs)

                total_acc += (pred_classes == labels).sum().item()
                total_mae += torch.abs(pred_classes - labels).sum().item()
                total += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(pred_classes.cpu().numpy())

        mae = total_mae / total
        acc = total_acc / total
        qoe = total_loss / len(loader)

        qwk_metric = QWK()
        qwk = qwk_metric.perform_measurement(all_labels, all_preds)

        self.model.train()
        return mae, qoe, acc, qwk

    # ---------------- CREATE MODEL (NIETKNIĘTE) ----------------
    def create_model(self, num_classes=5):
        # *** ZERO HEADÓW, ZERO K-1, ZERO BEBECHÓW ***
        # ConvNeXt jako zwykły classifier 5‑klasowy
        model = timm.create_model(
            "convnext_small",
            pretrained=True,
            num_classes=num_classes,   # 5 logitów
            global_pool='avg'
        )
        return model.to(self.device)
