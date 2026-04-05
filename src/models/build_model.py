from __future__ import annotations

from typing import Any

import torch.nn as nn
from torchvision import models

SUPPORTED_BACKBONES = ("resnet18", "resnet50", "efficientnet_b0", "convnext_tiny")
SUPPORTED_TASK_MODES = {"patch", "whole_image"}


def _get_task_mode(config: dict[str, Any]) -> str:
    task_config = config.get("task", {})
    task_mode = str(task_config.get("mode", "patch")).lower()
    if task_mode not in SUPPORTED_TASK_MODES:
        raise ValueError(f"Unsupported task.mode: {task_mode}. Expected one of {sorted(SUPPORTED_TASK_MODES)}")
    return task_mode


def _get_model_config(config: dict[str, Any]) -> tuple[str, bool, int, float]:
    model_config = config.get("model", {})
    data_config = config.get("data", {})

    model_name = str(model_config.get("name", "resnet18")).lower()
    pretrained = bool(model_config.get("pretrained", True))
    num_classes = int(data_config.get("num_classes", model_config.get("num_classes", 2)))
    dropout = float(model_config.get("dropout", 0.0))

    if not 0.0 <= dropout <= 1.0:
        raise ValueError(f"model.dropout must be in [0.0, 1.0], got: {dropout}")

    return model_name, pretrained, num_classes, dropout


def _build_dropout_linear(in_features: int, num_classes: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )


def _replace_classifier_with_configurable_dropout(
    classifier: nn.Sequential,
    num_classes: int,
    dropout: float,
    backbone_name: str,
) -> nn.Sequential:
    layers = list(classifier.children())
    if not layers or not isinstance(layers[-1], nn.Linear):
        raise ValueError(
            f"Expected the classifier for {backbone_name} to end with nn.Linear, got: {classifier}"
        )

    in_features = layers[-1].in_features
    preserved_layers = [layer for layer in layers[:-1] if not isinstance(layer, nn.Dropout)]
    preserved_layers.append(nn.Dropout(p=dropout))
    preserved_layers.append(nn.Linear(in_features, num_classes))
    return nn.Sequential(*preserved_layers)


def _build_resnet18(num_classes: int, pretrained: bool, dropout: float) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = _build_dropout_linear(model.fc.in_features, num_classes, dropout)
    return model


def _build_resnet50(num_classes: int, pretrained: bool, dropout: float) -> nn.Module:
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = _build_dropout_linear(model.fc.in_features, num_classes, dropout)
    return model


def _build_efficientnet_b0(num_classes: int, pretrained: bool, dropout: float) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier = _replace_classifier_with_configurable_dropout(
        classifier=model.classifier,
        num_classes=num_classes,
        dropout=dropout,
        backbone_name="efficientnet_b0",
    )
    return model


def _build_convnext_tiny(num_classes: int, pretrained: bool, dropout: float) -> nn.Module:
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = models.convnext_tiny(weights=weights)
    model.classifier = _replace_classifier_with_configurable_dropout(
        classifier=model.classifier,
        num_classes=num_classes,
        dropout=dropout,
        backbone_name="convnext_tiny",
    )
    return model


def _build_single_input_classifier(config: dict[str, Any]) -> nn.Module:
    model_name, pretrained, num_classes, dropout = _get_model_config(config)

    if model_name == "resnet18":
        return _build_resnet18(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    if model_name == "resnet50":
        return _build_resnet50(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    if model_name == "efficientnet_b0":
        return _build_efficientnet_b0(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    if model_name == "convnext_tiny":
        return _build_convnext_tiny(num_classes=num_classes, pretrained=pretrained, dropout=dropout)

    raise ValueError(
        f"Unsupported model backbone: {model_name}. "
        f"Currently supported: {', '.join(SUPPORTED_BACKBONES)}."
    )


def build_model(config: dict[str, Any]) -> nn.Module:
    """Build the single-input classifier used by patch and whole_image modes."""

    _get_task_mode(config)
    return _build_single_input_classifier(config)