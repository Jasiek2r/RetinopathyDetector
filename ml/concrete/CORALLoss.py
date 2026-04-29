import torch
import torch.nn as nn

class CORALLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        logits: (B, K-1)
        targets: (B,)
        """
        B, Km1 = logits.shape

        ordinal_targets = torch.zeros_like(logits)

        for k in range(Km1):
            ordinal_targets[:, k] = (targets > k).float()

        return self.bce(logits, ordinal_targets)