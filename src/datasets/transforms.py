from __future__ import annotations

from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _get_normalize_stats(normalize_cfg: str | dict[str, Any] | None) -> tuple[list[float], list[float]]:
    if normalize_cfg in (None, "imagenet"):
        return IMAGENET_MEAN, IMAGENET_STD

    if isinstance(normalize_cfg, dict):
        mean = normalize_cfg.get("mean", IMAGENET_MEAN)
        std = normalize_cfg.get("std", IMAGENET_STD)
        return list(mean), list(std)

    raise ValueError("normalize config must be None, 'imagenet', or a dict with mean/std.")


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

    transforms.extend(
        [
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms)
