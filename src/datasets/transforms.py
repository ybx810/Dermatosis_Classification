from __future__ import annotations

from typing import Any

import albumentations as A
import cv2
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


def _resolve_interpolation(interpolation: str | None) -> int:
    interpolation_name = str(interpolation or "bilinear").lower()
    interpolation_map = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    if interpolation_name not in interpolation_map:
        raise ValueError(
            f"Unsupported interpolation: {interpolation_name}. "
            f"Expected one of {sorted(interpolation_map)}."
        )
    return interpolation_map[interpolation_name]


def build_whole_image_transforms(
    mode: str,
    transform_cfg: dict[str, Any] | None = None,
    whole_image_config: dict[str, Any] | None = None,
) -> A.Compose:
    transform_cfg = transform_cfg or {}
    whole_image_config = whole_image_config or {}
    mode = mode.lower()
    if mode not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported mode: {mode}")

    horizontal_flip = float(transform_cfg.get("horizontal_flip", 0.0))
    vertical_flip = float(transform_cfg.get("vertical_flip", 0.0))
    normalize_cfg = transform_cfg.get("normalize", "imagenet")
    mean, std = _get_normalize_stats(normalize_cfg)
    image_size = int(
        whole_image_config.get("image_size")
        or transform_cfg.get("crop_size")
        or transform_cfg.get("resize_height")
        or 512
    )
    resize_size = int(whole_image_config.get("resize_size", image_size))
    if resize_size < image_size:
        raise ValueError(
            f"whole_image.resize_size must be >= whole_image.image_size, got {resize_size} < {image_size}."
        )
    interpolation = _resolve_interpolation(whole_image_config.get("interpolation", "bilinear"))

    transforms: list[A.BasicTransform] = [
        A.Resize(height=resize_size, width=resize_size, interpolation=interpolation),
        A.CenterCrop(height=image_size, width=image_size),
    ]
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