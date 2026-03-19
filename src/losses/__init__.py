from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.losses.focal_loss import FocalLoss, build_alpha_from_class_counts


def build_loss(
    config: dict[str, Any],
    class_counts: list[int] | dict[Any, int] | None = None,
    device: str | torch.device | None = None,
) -> nn.Module:
    """Build the configured classification loss."""

    loss_config = config.get("loss", {})
    loss_name = str(loss_config.get("name", "cross_entropy")).lower()
    reduction = str(loss_config.get("reduction", "mean"))

    if loss_name in {"cross_entropy", "ce"}:
        weight = None
        if class_counts is not None and loss_config.get("use_class_weights", False):
            weight = build_alpha_from_class_counts(class_counts)
            if device is not None:
                weight = weight.to(device)
        return nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    if loss_name == "focal":
        alpha = loss_config.get("alpha")
        if alpha is None and class_counts is not None and loss_config.get("use_alpha_from_class_counts", False):
            alpha = build_alpha_from_class_counts(class_counts)
        elif isinstance(alpha, list):
            alpha = torch.tensor(alpha, dtype=torch.float32)

        criterion = FocalLoss(
            alpha=alpha,
            gamma=float(loss_config.get("gamma", 2.0)),
            reduction=reduction,
        )
        if device is not None:
            criterion = criterion.to(device)
        return criterion

    raise ValueError(
        f"Unsupported loss: {loss_name}. Currently supported: cross_entropy, focal."
    )


__all__ = ["FocalLoss", "build_alpha_from_class_counts", "build_loss"]
