from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_alpha_from_class_counts(
    class_counts: Sequence[int] | dict[Any, int],
    normalize: bool = True,
) -> torch.Tensor:
    """Build inverse-frequency class weights for Focal Loss alpha."""

    if isinstance(class_counts, dict):
        counts = [class_counts[key] for key in sorted(class_counts)]
    else:
        counts = list(class_counts)

    if not counts:
        raise ValueError("class_counts must not be empty.")
    if any(count <= 0 for count in counts):
        raise ValueError("All class counts must be positive.")

    counts_tensor = torch.tensor(counts, dtype=torch.float32)
    alpha = counts_tensor.sum() / counts_tensor

    if normalize:
        alpha = alpha / alpha.sum() * len(counts)
    return alpha


class FocalLoss(nn.Module):
    """Multi-class Focal Loss for imbalanced classification."""

    def __init__(
        self,
        alpha: float | Sequence[float] | torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of: none, mean, sum.")

        self.gamma = gamma
        self.reduction = reduction

        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, torch.Tensor):
            self.alpha = alpha.float()
        elif isinstance(alpha, Sequence) and not isinstance(alpha, (str, bytes)):
            self.alpha = torch.tensor(list(alpha), dtype=torch.float32)
        else:
            self.alpha = torch.tensor(float(alpha), dtype=torch.float32)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError(f"logits must have shape [batch_size, num_classes], got {tuple(logits.shape)}")
        if targets.ndim != 1:
            raise ValueError(f"targets must have shape [batch_size], got {tuple(targets.shape)}")
        if logits.size(0) != targets.size(0):
            raise ValueError("Batch size of logits and targets must match.")

        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        targets = targets.long()
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_factor = (1.0 - target_probs).pow(self.gamma)
        loss = -focal_factor * target_log_probs

        if self.alpha is not None:
            alpha = self.alpha.to(device=logits.device, dtype=logits.dtype)
            if alpha.ndim == 0:
                loss = alpha * loss
            else:
                if alpha.numel() != logits.size(1):
                    raise ValueError(
                        "When alpha is a tensor or sequence, its length must match num_classes."
                    )
                loss = alpha.gather(0, targets) * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
