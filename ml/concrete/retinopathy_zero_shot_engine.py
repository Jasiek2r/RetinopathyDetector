import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from tqdm import tqdm


class RetinopathyZeroShotEngine:

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # DINOv2 z torch.hub — działa na Windows
        self.model = torch.hub.load(
            'facebookresearch/dinov2',
            'dinov2_vitb14',
            pretrained=True
        ).to(self.device)
        self.model.eval()

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[None, :, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[None, :, None, None]

        self.prototypes = None

    def _embed(self, images):
        with torch.no_grad():
            emb = self.model(images)
            return F.normalize(emb, dim=1)

    def build_prototypes(self, dataset):
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        class_embs = {i: [] for i in range(5)}

        print("Budowanie prototypów zero-shot:")
        for images, labels in tqdm(loader, ncols=120):
            images = images.to(self.device)
            images = torch.nn.functional.interpolate(images, (224, 224))
            images = (images - self.mean) / self.std

            emb = self._embed(images)

            for e, lbl in zip(emb, labels):
                class_embs[int(lbl)].append(e)

        protos = []
        for i in range(5):
            proto = torch.stack(class_embs[i]).mean(dim=0)
            proto = F.normalize(proto, dim=0)
            protos.append(proto)

        self.prototypes = torch.stack(protos).to(self.device)
        print("✔ Prototypy zbudowane!")

    def evaluate(self, dataset):
        if self.prototypes is None:
            raise RuntimeError("Najpierw wywołaj build_prototypes(train_dataset)")

        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        all_preds, all_labels = [], []

        print("Zero-shot inference:")
        for images, labels in tqdm(loader, ncols=120):
            images = images.to(self.device)
            images = torch.nn.functional.interpolate(images, (224, 224))
            images = (images - self.mean) / self.std

            emb = self._embed(images)
            sims = torch.matmul(emb, self.prototypes.T)
            preds = torch.argmax(sims, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
        cm = confusion_matrix(all_labels, all_preds)

        print(f"\nZero-shot ACC: {acc:.4f}")
        print(f"Zero-shot QWK: {qwk:.4f}")
        print("Confusion matrix:")
        print(cm)

        return acc, qwk, cm
