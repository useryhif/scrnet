"""Loss for training a single space-group expert."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveLoss(nn.Module):
    """Cross-entropy inside the expert's label range, a pull to uniform outside it.

    Samples whose label falls in ``[start, start + length)`` are the expert's own space
    groups and contribute a normal cross-entropy term. Samples outside that range are
    negatives for this expert; instead of being pushed toward some other class, their logits
    are pulled toward the uniform value 0.5, which teaches the expert to stay undecided —
    and therefore low-confidence — on inputs that belong to another expert.

    Args:
        start: first label index belonging to this expert.
        length: number of labels belonging to this expert.
        use_mse: if True, penalise negative logits with MSE toward 0.5; if False, use the
            mean L2 norm of the logits instead.
    """

    def __init__(self, start: int, length: int, use_mse: bool = True):
        super().__init__()
        self.start = start
        self.end = start + length
        self.use_mse = use_mse

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: ``[batch_size, num_classes]``
            labels: ``[batch_size]``
        """
        valid_mask = (labels >= self.start) & (labels < self.end)
        invalid_mask = ~valid_mask

        loss = torch.tensor(0.0, device=logits.device)

        if valid_mask.any():
            loss = loss + F.cross_entropy(logits[valid_mask], labels[valid_mask])

        if invalid_mask.any():
            negative_logits = logits[invalid_mask]
            if self.use_mse:
                target = torch.full_like(negative_logits, fill_value=0.5)
                loss = loss + F.mse_loss(negative_logits, target)
            else:
                loss = loss + torch.mean(torch.norm(negative_logits, dim=1))

        return loss
