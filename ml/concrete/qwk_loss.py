import torch
import torch.nn as nn
import torch.nn.functional as F


class QuadraticWeightedKappaLoss(nn.Module):
    def __init__(self, num_classes=5, epsilon=1e-10):
        super(QuadraticWeightedKappaLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

        # Tworzenie stałej macierzy wag (penalizacja kwadratowa odległości)
        # Rejestrujemy jako buffer, żeby PyTorch pamiętał o przenoszeniu na GPU (Cuda)
        w = torch.zeros((num_classes, num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                w[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)
        self.register_buffer('w', w)

    def forward(self, predictions, targets):
        """
        predictions: Tensor o kształcie (batch_size, num_classes) - surowe logity z modelu
        targets: Tensor o kształcie (batch_size,) - indeksy klas rzeczywistych (0-4)
        """
        batch_size = predictions.size(0)

        # 1. Zamiana logitów na ciągłe prawdopodobieństwa (Softmax)
        preds_soft = F.softmax(predictions, dim=1)

        # 2. Zamiana targetów na standardowy one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()

        # 3. Obliczenie macierzy zaobserwowanej (Observed matrix)
        # rzutowanie prawdopodobieństw batcha na klasy
        O = torch.matmul(targets_one_hot.t(), preds_soft)

        # 4. Obliczenie macierzy oczekiwanej przy losowym trafnieniu (Expected matrix)
        hist_targets = targets_one_hot.sum(dim=0, keepdim=True)
        hist_preds = preds_soft.sum(dim=0, keepdim=True)
        E = torch.matmul(hist_targets.t(), hist_preds) / batch_size

        # 5. Normalizacja macierzy sumą elementów
        O = O / (O.sum() + self.epsilon)
        E = E / (E.sum() + self.epsilon)

        # 6. Wyznaczenie licznika i mianownika dla QWK
        num = (self.w * O).sum()
        den = (self.w * E).sum()

        # 7. Wynik Loss: dążymy do Kappa = 1, więc minimalizujemy (1 - Kappa)
        # Może przyjąć postać: num / (den + self.epsilon)
        loss = num / (den + self.epsilon)

        return loss
