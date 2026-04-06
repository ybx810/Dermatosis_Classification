from __future__ import annotations

from typing import Any

import torch.nn as nn
from torchvision import models

SUPPORTED_BACKBONES = ("resnet18", "resnet50", "efficientnet_b0", "convnext_tiny")


def _validate_task_mode(config: dict[str, Any]) -> None:
    task_mode = str(config.get("task", {}).get("mode", "whole_image")).lower()
    if task_mode != "whole_image":
        raise ValueError("This project only supports task.mode=whole_image.")


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


def _normalize_backbone_name(backbone_name: str) -> str:
    normalized = str(backbone_name).lower()
    if normalized not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported model backbone: {normalized}. "
            f"Currently supported: {', '.join(SUPPORTED_BACKBONES)}."
        )
    return normalized


def get_classifier_module(model: nn.Module, backbone_name: str) -> nn.Module:
    normalized_name = _normalize_backbone_name(backbone_name)
    if normalized_name in {"resnet18", "resnet50"}:
        classifier_module = model.fc
    else:
        classifier_module = model.classifier

    if not isinstance(classifier_module, nn.Module):
        raise ValueError(f"Invalid classifier module for {normalized_name}.")
    return classifier_module


def get_classifier_parameters(model: nn.Module, backbone_name: str) -> list[nn.Parameter]:
    classifier_parameters = list(get_classifier_module(model, backbone_name).parameters())
    if not classifier_parameters:
        raise ValueError(f"Classifier parameters are empty for {backbone_name}.")
    return classifier_parameters


def get_backbone_parameters(model: nn.Module, backbone_name: str) -> list[nn.Parameter]:
    classifier_parameters = get_classifier_parameters(model, backbone_name)
    classifier_param_ids = {id(parameter) for parameter in classifier_parameters}
    backbone_parameters = [parameter for parameter in model.parameters() if id(parameter) not in classifier_param_ids]

    if not backbone_parameters:
        raise ValueError(f"Backbone parameters are empty for {backbone_name}.")

    backbone_param_ids = {id(parameter) for parameter in backbone_parameters}
    if classifier_param_ids.intersection(backbone_param_ids):
        raise ValueError(f"Classifier and backbone parameters overlap for {backbone_name}.")

    model_param_ids = {id(parameter) for parameter in model.parameters()}
    if classifier_param_ids.union(backbone_param_ids) != model_param_ids:
        raise ValueError(f"Classifier/backbone partition is incomplete for {backbone_name}.")

    return backbone_parameters


def get_backbone_modules(model: nn.Module, backbone_name: str) -> list[nn.Module]:
    normalized_name = _normalize_backbone_name(backbone_name)
    classifier_attr = "fc" if normalized_name in {"resnet18", "resnet50"} else "classifier"
    backbone_modules = [module for child_name, module in model.named_children() if child_name != classifier_attr]
    if not backbone_modules:
        raise ValueError(f"Backbone modules are empty for {normalized_name}.")
    return backbone_modules


def set_backbone_trainable(
    model: nn.Module,
    backbone_name: str,
    trainable: bool,
    train_bn_when_frozen: bool = False,
) -> None:
    for parameter in get_classifier_parameters(model, backbone_name):
        parameter.requires_grad = True

    for parameter in get_backbone_parameters(model, backbone_name):
        parameter.requires_grad = bool(trainable)

    if not trainable and not train_bn_when_frozen:
        for backbone_module in get_backbone_modules(model, backbone_name):
            for module in backbone_module.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()


def build_model(config: dict[str, Any]) -> nn.Module:
    _validate_task_mode(config)
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
