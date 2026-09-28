"""Focal loss for class imbalance."""

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """``FL(p_t) = (1 - p_t)^gamma * CE(p_t)``

    The modulating factor ``(1 - p_t)^gamma`` down-weights easy, well-classified examples so
    that training concentrates on hard ones. ``gamma = 0`` reduces this to plain
    cross-entropy.

    Note:
        This implementation derives ``p_t`` from ``exp(-CE)``, which is exact only for a
        single-class target with unweighted cross-entropy.

    Args:
        gamma: focusing parameter.
        weight: optional per-class weights passed to the underlying cross-entropy.
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)
        pt = torch.exp(-ce_loss)
        return (1 - pt) ** self.gamma * ce_loss
