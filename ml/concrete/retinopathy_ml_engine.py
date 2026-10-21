import traceback
import numpy as np
import torch
import torch.optim as optim

from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score

from ml.abstractions.ml_engine import MLEngine
from ml.concrete.FocalLoss import FocalLoss
from ml.concrete.model_provider import ModelProvider

import os

from ml.concrete.qwk_loss import QuadraticWeightedKappaLoss


class RetinopathyMLEngine(MLEngine):


    def __init__(self, provider: ModelProvider, device=None, precision="fp16"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = provider.create_conv_model().to(self.device)
        #self.model = provider.create_model().to(self.device)
        #self.model = provider.create_retfound().to(self.device)

        # Konfiguracja precyzji (fp16, bf16 lub None/fp32)
        self.precision = precision.lower() if precision else None

        if self.precision in ["fp16", "float16"]:
            self.dtype = torch.float16
            self.device_type = "cuda" if "cuda" in str(self.device) else "cpu"
            self.scaler = torch.amp.GradScaler(self.device_type)
        elif self.precision in ["bf16", "bfloat16"]:
            self.dtype = torch.bfloat16
            self.device_type = "cuda" if "cuda" in str(self.device) else "cpu"
            self.scaler = None  # bfloat16 nie wymaga skalera
        else:
            self.dtype = torch.float32
            self.device_type = "cpu"
            self.scaler = None

        # precompute normalization (IMPORTANT)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[None, :, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[None, :, None, None]

    # -------------------------
    # TRAIN
    # -------------------------
    def train(self, train_dataset, val_dataset=None, batch_size=64, epochs=30):

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
            num_workers=8,
            pin_memory=True
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=6,
                pin_memory=True
            )

        criterion = QuadraticWeightedKappaLoss()

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
                desc=f"Epoch {epoch + 1}/{epochs}",
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

                    with torch.amp.autocast(device_type=self.device_type, dtype=self.dtype):
                        outputs = self.model(images)
                        loss = criterion(outputs, labels)

                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
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

            print(f"\nEpoch {epoch + 1} loss: {running_loss / len(train_loader):.4f}")

            if val_loader is not None:
                acc, qwk, cm = self._validate(val_loader)
                print(f"VAL acc: {acc:.2f}% | QWK: {qwk:.2f}")
                np.savetxt(f"confusion_epoch_{epoch + 1}.txt", cm, fmt="%d")

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

                with torch.amp.autocast(device_type=self.device_type, dtype=self.dtype):
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
            num_workers=2
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

        torch.save(self.model.state_dict(), "model_weights.pth")
        return acc

    def full_evaluation(self, train_dataset, val_dataset, test_dataset):

        os.makedirs("evaluation_results", exist_ok=True)

        def get_predictions(name, dataset):
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=64,
                shuffle=False,
                num_workers=8,
                pin_memory=True
            )

            all_preds = []
            all_labels = []

            self.model.eval()
            with torch.no_grad():
                for x, y in tqdm(loader, desc=f"Evaluating {name}", ncols=120):
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)

                    x = torch.nn.functional.interpolate(
                        x, size=(224, 224), mode="bilinear", align_corners=False
                    )

                    x = (x - self.mean) / self.std

                    logits = self.model(x)
                    preds = torch.argmax(logits, dim=1)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y.cpu().numpy())

            return all_labels, all_preds

        # === PREDYKCJE ===
        y_train, y_train_pred = get_predictions("Train", train_dataset)
        y_val, y_val_pred = get_predictions("Validation", val_dataset)
        y_test, y_test_pred = get_predictions("Test", test_dataset)

        # === ACCURACY ===
        acc_train = accuracy_score(y_train, y_train_pred)
        acc_val = accuracy_score(y_val, y_val_pred)
        acc_test = accuracy_score(y_test, y_test_pred)

        # === QWK ===
        qwk_train = cohen_kappa_score(y_train, y_train_pred, weights='quadratic')
        qwk_val = cohen_kappa_score(y_val, y_val_pred, weights='quadratic')
        qwk_test = cohen_kappa_score(y_test, y_test_pred, weights='quadratic')

        # === PRINT METRICS ===
        print("=== Accuracy ===")
        print(f"Train: {acc_train:.4f}")
        print(f"Val:   {acc_val:.4f}")
        print(f"Test:  {acc_test:.4f}\n")

        print("=== QWK ===")
        print(f"Train: {qwk_train:.4f}")
        print(f"Val:   {qwk_val:.4f}")
        print(f"Test:  {qwk_test:.4f}\n")

        # === SAVE METRICS TO TXT ===
        with open("evaluation_results/metrics.txt", "w") as f:
            f.write("=== Accuracy ===\n")
            f.write(f"Train: {acc_train:.4f}\n")
            f.write(f"Val:   {acc_val:.4f}\n")
            f.write(f"Test:  {acc_test:.4f}\n\n")

            f.write("=== QWK ===\n")
            f.write(f"Train: {qwk_train:.4f}\n")
            f.write(f"Val:   {qwk_val:.4f}\n")
            f.write(f"Test:  {qwk_test:.4f}\n")

        print("[INFO] Saved metrics: evaluation_results/metrics.txt")

        # === CONFUSION MATRICES TO TXT ===
        sets = [
            ("train", y_train, y_train_pred),
            ("validation", y_val, y_val_pred),
            ("test", y_test, y_test_pred)
        ]

        for name, y_true, y_pred in sets:
            cm = confusion_matrix(y_true, y_pred)

            out_path = f"evaluation_results/confusion_matrix_{name}.txt"
            np.savetxt(out_path, cm, fmt="%d")

            print(f"[INFO] Saved confusion matrix: {out_path}")

    def evaluate_with_uncertainty(self, dataset, batch_size=32, num_tta_steps=5,
                                  coverage_steps=[1.0, 0.9, 0.8, 0.7, 0.6, 0.4, 0.2]):
        """
        Eksperyment z kwantyfikacją niepewności (Selective Classification / Risk-Coverage).
        Odfiltrowuje najbardziej niepewne predykcje i sprawdza wzrost Accuracy.
        """
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        self.model.eval()
        all_preds = []
        all_labels = []
        all_uncertainties = []

        print("\n[INFO] Rozpoczynam ewaluację z Test-Time Augmentation (TTA) i niepewnością...")

        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Uncertainty Eval", ncols=120):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                batch_size_curr = images.size(0)
                num_classes = 5  # Liczba klas w retinopatii (0-4)

                # Zbiornik na wyniki z wariantów TTA
                all_probs = torch.zeros(num_tta_steps, batch_size_curr, num_classes, device=self.device)

                for i in range(num_tta_steps):
                    x = images.clone()
                    if i % 2 == 1:
                        x = torch.flip(x, dims=[-1])  # Prosta augmentacja TTA (flip)

                    x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
                    x = (x - self.mean) / self.std

                    with torch.amp.autocast(device_type=self.device_type, dtype=self.dtype):
                        logits = self.model(x)
                    all_probs[i] = torch.softmax(logits, dim=1)

                # Średnie prawdopodobieństwa z TTA
                mean_probs = all_probs.mean(dim=0)

                # Obliczenie entropii predykcyjnej (miara niepewności)
                eps = 1e-8
                entropy = -torch.sum(mean_probs * torch.log(mean_probs + eps), dim=1)

                preds = torch.argmax(mean_probs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_uncertainties.extend(entropy.cpu().numpy())

        # Konwersja do numpy
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        uncertainties = np.array(all_uncertainties)

        os.makedirs("evaluation_results", exist_ok=True)
        results_path = "evaluation_results/risk_coverage_analysis.txt"

        print("\n=== WYNIKI ANALIZY RYZYKA-POKRYCIA (RISK-COVERAGE) ===")
        with open(results_path, "w") as f:
            f.write("=== Risk-Coverage Analysis (Uncertainty Estimation) ===\n")

            for coverage in coverage_steps:
                # Wybieramy próg niepewności dla danego poziomu pokrycia
                # np. coverage=0.9 oznacza, że bierzemy 90% najpewniejszych próbek
                k = int(len(uncertainties) * coverage)
                if k == 0:
                    continue

                # Sortujemy próbki rosnąco według niepewności (najpewnieje na początku)
                sorted_idx = np.argsort(uncertainties)
                selected_idx = sorted_idx[:k]

                sub_true = y_true[selected_idx]
                sub_pred = y_pred[selected_idx]

                acc = accuracy_score(sub_true, sub_pred) * 100
                qwk = cohen_kappa_score(sub_true, sub_pred, weights='quadratic')

                line = f"Pokrycie zbioru: {int(coverage * 100)}% | Liczba próbek: {k}/{len(y_true)} | Accuracy: {acc:.2f}% | QWK: {qwk:.4f}"
                print(line)
                f.write(line + "\n")

        print(f"\n[INFO] Zapisano raport z niepewnością do: {results_path}")
