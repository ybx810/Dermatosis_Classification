from __future__ import annotations

from typing import Any

import torch.nn as nn

VALID_FINETUNE_MODES = {"full", "head_only"}
SUPPORTED_HEAD_ONLY_BACKBONES = {"resnet18", "resnet50", "efficientnet_b0", "convnext_tiny"}
_BATCH_NORM_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)


def _resolve_requested_finetune_mode(config: dict[str, Any]) -> str:
    finetune_config = config.get("finetune", {})
    requested_mode = str(finetune_config.get("mode", "full")).lower()
    if requested_mode not in VALID_FINETUNE_MODES:
        raise ValueError(
            f"Unsupported finetune.mode: {requested_mode}. "
            f"Expected one of {sorted(VALID_FINETUNE_MODES)}."
        )
    return requested_mode


def resolve_finetune_mode(config: dict[str, Any]) -> str:
    finetune_config = config.get("finetune", {})
    enabled = bool(finetune_config.get("enabled", False))
    requested_mode = _resolve_requested_finetune_mode(config)
    if not enabled:
        return "full"
    return requested_mode


def _resolve_backbone_name(config: dict[str, Any]) -> str:
    return str(config.get("model", {}).get("name", "resnet18")).lower()


def _resolve_classification_head(model: nn.Module, backbone_name: str) -> tuple[nn.Module, str]:
    if backbone_name in {"resnet18", "resnet50"}:
        head_name = "fc"
    elif backbone_name in {"efficientnet_b0", "convnext_tiny"}:
        head_name = "classifier"
    else:
        raise ValueError(
            f"finetune.mode=head_only is not supported for backbone: {backbone_name}. "
            f"Supported backbones: {sorted(SUPPORTED_HEAD_ONLY_BACKBONES)}."
        )

    head_module = getattr(model, head_name, None)
    if not isinstance(head_module, nn.Module):
        raise ValueError(
            f"Backbone {backbone_name} is missing expected classification head attribute '{head_name}'."
        )
    return head_module, head_name


def configure_trainable_parameters(model: nn.Module, config: dict[str, Any]) -> dict[str, Any]:
    finetune_config = config.get("finetune", {})
    requested_mode = _resolve_requested_finetune_mode(config)
    mode = resolve_finetune_mode(config)
    backbone_name = _resolve_backbone_name(config)

    trainable_modules: list[str]
    if mode == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True
        trainable_modules = ["all"]
    elif mode == "head_only":
        head_module, head_name = _resolve_classification_head(model, backbone_name)
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in head_module.parameters():
            parameter.requires_grad = True
        trainable_modules = [head_name]
    else:
        raise ValueError(
            f"Unsupported effective finetune mode: {mode}. "
            f"Expected one of {sorted(VALID_FINETUNE_MODES)}."
        )

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_param_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_params <= 0:
        raise ValueError(
            "No trainable parameters found after applying finetune settings. "
            "Please check finetune.mode and model backbone."
        )

    return {
        "enabled": bool(finetune_config.get("enabled", False)),
        "requested_mode": requested_mode,
        "mode": mode,
        "backbone_name": backbone_name,
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
        "frozen_params": int(total_params - trainable_params),
        "trainable_modules": trainable_modules,
        "trainable_parameter_names": trainable_param_names,
    }


def set_frozen_backbone_eval(model: nn.Module, finetune_mode: str) -> dict[str, Any]:
    mode = str(finetune_mode).lower()
    if mode not in VALID_FINETUNE_MODES:
        raise ValueError(
            f"Unsupported finetune mode: {mode}. Expected one of {sorted(VALID_FINETUNE_MODES)}."
        )
    if mode != "head_only":
        return {"mode": mode, "bn_modules_forced_eval": 0, "applied": False}

    bn_modules_forced_eval = 0
    for module in model.modules():
        if not isinstance(module, _BATCH_NORM_TYPES):
            continue

        module_parameters = tuple(module.parameters(recurse=False))
        has_trainable_parameters = any(parameter.requires_grad for parameter in module_parameters)
        if has_trainable_parameters:
            continue

        module.eval()
        bn_modules_forced_eval += 1

    return {
        "mode": mode,
        "bn_modules_forced_eval": int(bn_modules_forced_eval),
        "applied": True,
    }
