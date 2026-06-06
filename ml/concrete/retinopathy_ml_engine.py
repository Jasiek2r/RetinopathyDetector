import traceback
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel

from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from ml.abstractions.ml_engine import MLEngine
from ml.concrete.FocalLoss import FocalLoss


class DinoRetinopathyModel(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0]
        return self.classifier(cls_token)


class RetinopathyMLEngine(MLEngine):
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.create_conv_model().to(self.device)

        # precompute normalization (IMPORTANT)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[None, :, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[None, :, None, None]

    # -------------------------
    # TRAIN
    # -------------------------
    def train(self, train_dataset, val_dataset=None, batch_size=8, epochs=30):

        # imbalance (sampler only)
        labels_np = train_dataset.df["diagnosis"].values
        class_counts = np.bincount(labels_np)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels_np]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=12,
            pin_memory=True
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True
            )

        criterion = FocalLoss(gamma=2.0)

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=3e-5,
            weight_decay=1e-4
        )

        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in range(epochs):
            self.model.train()

            running_loss = 0.0

            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{epochs}",
                ncols=120
            )

            for batch_idx, (images, labels) in enumerate(progress_bar):

                try:
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    # resize
                    images = torch.nn.functional.interpolate(
                        images,
                        size=(224, 224),
                        mode="bilinear",
                        align_corners=False
                    )

                    # DINOv2 normalization (CRITICAL)
                    images = (images - self.mean) / self.std

                    optimizer.zero_grad()

                    outputs = self.model(images)
                    loss = criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                    progress_bar.set_postfix({
                        "loss": f"{running_loss / (batch_idx + 1):.4f}"
                    })

                except Exception as e:
                    print(f"\nERROR batch {batch_idx}")
                    traceback.print_exc()
                    raise

            scheduler.step()

            print(f"\nEpoch {epoch+1} loss: {running_loss / len(train_loader):.4f}")

            if val_loader is not None:
                acc, qwk, cm = self._validate(val_loader)
                print(f"VAL acc: {acc:.2f}% | QWK: {qwk:.2f}")
                np.savetxt(f"confusion_epoch_{epoch+1}.txt", cm, fmt="%d")

    # -------------------------
    # VALIDATION
    # -------------------------
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
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False
                )

                images = (images - self.mean) / self.std

                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

                total += labels.size(0)
                correct += (preds == labels).sum().item()

        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)

        acc = 100 * correct / total
        qwk = cohen_kappa_score(labels, preds, weights="quadratic") * 100
        cm = confusion_matrix(labels, preds)

        return acc, qwk, cm

    # -------------------------
    # TEST
    # -------------------------
    def test(self, dataset, batch_size=8):

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )

        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:

                images = images.to(self.device)
                labels = labels.to(self.device)

                images = torch.nn.functional.interpolate(
                    images,
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False
                )

                images = (images - self.mean) / self.std

                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)

                total += labels.size(0)
                correct += (preds == labels).sum().item()

        acc = 100 * correct / total
        print(f"TEST accuracy: {acc:.2f}%")

        torch.save(self.model.state_dict(), "dino_retinopathy.pth")
        return acc

    def create_conv_model(self, num_classes = 5):
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
    # -------------------------
    # MODEL
    # -------------------------
    def create_model(self, num_classes=5):
        backbone = AutoModel.from_pretrained("facebook/dinov2-base")

        # freeze backbone
        for p in backbone.parameters():
            p.requires_grad = False
        for name, param in backbone.named_parameters():
            if "blocks.11" in name or "blocks.10" in name:
                param.requires_grad = True

        hidden = backbone.config.hidden_size

        classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        return DinoRetinopathyModel(backbone, classifier)