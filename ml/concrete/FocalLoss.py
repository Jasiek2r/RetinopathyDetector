import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        """
        gamma: focusing parameter (2.0 is standard)
        alpha: optional class weights tensor [C]
        reduction: 'mean' | 'sum' | 'none'
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: logits [B, C]
        targets: [B] class indices
        """

        # standard cross entropy per sample
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # probabilities of correct class
        pt = torch.exp(-ce_loss)

        # focal modulation
        loss = (1 - pt) ** self.gamma * ce_loss

        # optional class weighting
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss

        # reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss