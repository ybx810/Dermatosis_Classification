from __future__ import annotations

from typing import Any

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
BORDER_MODE_MAP = {
    "constant": cv2.BORDER_CONSTANT,
    "reflect": cv2.BORDER_REFLECT_101,
    "replicate": cv2.BORDER_REPLICATE,
}


def _get_normalize_stats(normalize_cfg: str | dict[str, Any] | None) -> tuple[list[float], list[float]]:
    if normalize_cfg in (None, "imagenet"):
        return IMAGENET_MEAN, IMAGENET_STD

    if isinstance(normalize_cfg, dict):
        mean = normalize_cfg.get("mean", IMAGENET_MEAN)
        std = normalize_cfg.get("std", IMAGENET_STD)
        return list(mean), list(std)

    raise ValueError("normalize config must be None, 'imagenet', or a dict with mean/std.")


def _resolve_pair(values: Any, default: tuple[float, float], name: str) -> tuple[float, float]:
    if values is None:
        return default
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError(f"{name} must be a list/tuple with exactly 2 values.")
    return float(values[0]), float(values[1])


def _resolve_rotate_fill(rotate_cfg: dict[str, Any], whole_image_config: dict[str, Any]) -> float | tuple[float, ...]:
    fill_value = rotate_cfg.get("fill", whole_image_config.get("pad_value", 0))
    if isinstance(fill_value, (list, tuple)):
        return tuple(float(value) for value in fill_value)
    return float(fill_value)


def _build_random_resized_crop(
    image_size: int,
    p: float,
    scale: tuple[float, float],
    ratio: tuple[float, float],
) -> A.BasicTransform:
    try:
        return A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=scale,
            ratio=ratio,
            p=p,
        )
    except TypeError:
        return A.RandomResizedCrop(
            height=image_size,
            width=image_size,
            scale=scale,
            ratio=ratio,
            p=p,
        )


def _build_rotate(
    limit: float,
    border_mode: int,
    fill: float | tuple[float, ...],
    crop_border: bool,
    p: float,
) -> A.BasicTransform:
    try:
        return A.Rotate(
            limit=limit,
            border_mode=border_mode,
            fill=fill,
            crop_border=crop_border,
            p=p,
        )
    except TypeError:
        return A.Rotate(
            limit=limit,
            border_mode=border_mode,
            value=fill,
            crop_border=crop_border,
            p=p,
        )


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

    image_size = int(whole_image_config.get("image_size", 512))
    if image_size <= 0:
        raise ValueError("whole_image.image_size must be a positive integer.")

    horizontal_flip = float(transform_cfg.get("horizontal_flip", 0.0))
    vertical_flip = float(transform_cfg.get("vertical_flip", 0.0))
    normalize_cfg = transform_cfg.get("normalize", "imagenet")
    mean, std = _get_normalize_stats(normalize_cfg)

    random_resized_crop_cfg = transform_cfg.get("random_resized_crop", {}) or {}
    random_resized_crop_enabled = bool(random_resized_crop_cfg.get("enabled", False))
    random_resized_crop_p = float(random_resized_crop_cfg.get("p", 0.5))
    random_resized_crop_scale = _resolve_pair(
        random_resized_crop_cfg.get("scale"),
        default=(0.85, 1.0),
        name="train.augmentation.random_resized_crop.scale",
    )
    random_resized_crop_ratio = _resolve_pair(
        random_resized_crop_cfg.get("ratio"),
        default=(0.95, 1.05),
        name="train.augmentation.random_resized_crop.ratio",
    )

    rotate_cfg = transform_cfg.get("rotate", {}) or {}
    rotate_enabled = bool(rotate_cfg.get("enabled", False))
    rotate_p = float(rotate_cfg.get("p", 0.5))
    rotate_limit = float(rotate_cfg.get("limit", 10))
    border_mode_name = str(rotate_cfg.get("border_mode", "constant")).lower()
    if border_mode_name not in BORDER_MODE_MAP:
        raise ValueError(
            f"Unsupported rotate.border_mode: {border_mode_name}. Expected one of {sorted(BORDER_MODE_MAP)}."
        )
    rotate_border_mode = BORDER_MODE_MAP[border_mode_name]
    rotate_fill = _resolve_rotate_fill(rotate_cfg, whole_image_config)
    rotate_crop_border = bool(rotate_cfg.get("crop_border", False))

    # Whole-image geometry is finalized offline in scripts/prepare_whole_images.py.
    # Cached images should already be square images with side length equal to whole_image.image_size.
    transforms: list[A.BasicTransform] = []
    if mode == "train":
        # Fixed order in train mode:
        # 1) RandomResizedCrop -> 2) Rotate -> 3) HorizontalFlip -> 4) VerticalFlip
        if random_resized_crop_enabled:
            transforms.append(
                _build_random_resized_crop(
                    image_size=image_size,
                    p=random_resized_crop_p,
                    scale=random_resized_crop_scale,
                    ratio=random_resized_crop_ratio,
                )
            )
        if rotate_enabled:
            transforms.append(
                _build_rotate(
                    limit=rotate_limit,
                    border_mode=rotate_border_mode,
                    fill=rotate_fill,
                    crop_border=rotate_crop_border,
                    p=rotate_p,
                )
            )
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
