from __future__ import annotations

import torch.nn as nn

from src.models.build_model import build_model


def build_classifier(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Backward-compatible wrapper around the unified build_model entry point."""

    config = {
        "model": {
            "name": model_name,
            "pretrained": pretrained,
        },
        "data": {
            "num_classes": num_classes,
        },
    }
    return build_model(config)
