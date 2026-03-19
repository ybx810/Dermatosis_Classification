from __future__ import annotations

from typing import Any

import torch.nn as nn
from torchvision import models


def _get_model_config(config: dict[str, Any]) -> tuple[str, bool, int]:
    model_config = config.get("model", {})
    data_config = config.get("data", {})

    model_name = str(model_config.get("name", "resnet18")).lower()
    pretrained = bool(model_config.get("pretrained", True))
    num_classes = int(data_config.get("num_classes", model_config.get("num_classes", 2)))
    return model_name, pretrained, num_classes


def _build_resnet18(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_resnet50(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_efficientnet_b0(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def _build_convnext_tiny(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = models.convnext_tiny(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def build_model(config: dict[str, Any]) -> nn.Module:
    """Build a torchvision baseline model for patch classification."""

    model_name, pretrained, num_classes = _get_model_config(config)

    if model_name == "resnet18":
        return _build_resnet18(num_classes=num_classes, pretrained=pretrained)
    if model_name == "resnet50":
        return _build_resnet50(num_classes=num_classes, pretrained=pretrained)
    if model_name == "efficientnet_b0":
        return _build_efficientnet_b0(num_classes=num_classes, pretrained=pretrained)
    if model_name == "convnext_tiny":
        return _build_convnext_tiny(num_classes=num_classes, pretrained=pretrained)

    raise ValueError(
        f"Unsupported model backbone: {model_name}. "
        "Currently supported: resnet18, resnet50, efficientnet_b0, convnext_tiny."
    )
