from __future__ import annotations

from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
_MAX_UINT8_VALUE = 255.0


def _get_normalize_stats(normalize_cfg: str | dict[str, Any] | None) -> tuple[list[float], list[float]]:
    if normalize_cfg in (None, "imagenet"):
        return IMAGENET_MEAN, IMAGENET_STD

    if isinstance(normalize_cfg, dict):
        mean = normalize_cfg.get("mean", IMAGENET_MEAN)
        std = normalize_cfg.get("std", IMAGENET_STD)
        return list(mean), list(std)

    raise ValueError("normalize config must be None, 'imagenet', or a dict with mean/std.")


def _to_albumentations_std_range(std_min: float, std_max: float) -> tuple[float, float]:
    if std_min < 0 or std_max < 0:
        raise ValueError(f"noise std_min/std_max must be non-negative, got: {std_min}, {std_max}")
    if std_min > std_max:
        raise ValueError(f"noise std_min must be <= std_max, got: {std_min} > {std_max}")

    if std_max > 1.0:
        return std_min / _MAX_UINT8_VALUE, std_max / _MAX_UINT8_VALUE
    return std_min, std_max


def _build_noise_transform(transform_cfg: dict[str, Any]) -> A.BasicTransform | None:
    noise_cfg = transform_cfg.get("noise", {}) or {}
    enabled = bool(noise_cfg.get("enabled", False))
    if not enabled:
        return None

    noise_type = str(noise_cfg.get("type", "gaussian")).lower()
    probability = float(noise_cfg.get("p", 0.2))
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"augmentation.noise.p must be in [0, 1], got: {probability}")

    if noise_type != "gaussian":
        raise ValueError(f"Unsupported augmentation.noise.type: {noise_type}. Currently supported: gaussian")

    std_min = float(noise_cfg.get("std_min", 5.0))
    std_max = float(noise_cfg.get("std_max", 15.0))
    std_range = _to_albumentations_std_range(std_min, std_max)
    return A.GaussNoise(std_range=std_range, p=probability)


def build_patch_transforms(mode: str, transform_cfg: dict[str, Any] | None = None) -> A.Compose:
    transform_cfg = transform_cfg or {}
    mode = mode.lower()
    if mode not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported mode: {mode}")

    resize_height = transform_cfg.get("resize_height")
    resize_width = transform_cfg.get("resize_width")
    crop_size = transform_cfg.get("crop_size")
    horizontal_flip = float(transform_cfg.get("horizontal_flip", 0.0))
    vertical_flip = float(transform_cfg.get("vertical_flip", 0.0))
    normalize_cfg = transform_cfg.get("normalize", "imagenet")
    mean, std = _get_normalize_stats(normalize_cfg)

    transforms: list[A.BasicTransform] = []

    if resize_height and resize_width:
        transforms.append(A.Resize(height=int(resize_height), width=int(resize_width)))

    if crop_size:
        crop_size = int(crop_size)
        if mode == "train":
            transforms.append(A.RandomCrop(height=crop_size, width=crop_size))
        else:
            transforms.append(A.CenterCrop(height=crop_size, width=crop_size))

    if mode == "train":
        if horizontal_flip > 0:
            transforms.append(A.HorizontalFlip(p=horizontal_flip))
        if vertical_flip > 0:
            transforms.append(A.VerticalFlip(p=vertical_flip))

        noise_transform = _build_noise_transform(transform_cfg)
        if noise_transform is not None:
            transforms.append(noise_transform)

    transforms.extend(
        [
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms)
