import torch
import torch.nn as nn

class CORALLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        # targets MUSI być [B] int
        if targets.ndim != 1:
            targets = targets.argmax(dim=1)

        batch_size, num_ordinal = logits.shape

        ordinal_targets = torch.zeros_like(logits)

        for k in range(num_ordinal):
            ordinal_targets[:, k] = (targets > k).float()

        loss = nn.BCEWithLogitsLoss()(logits, ordinal_targets)
        return loss