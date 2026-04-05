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


def _resolve_interpolation(interpolation: str | None) -> int:
    interpolation_name = str(interpolation or "area").lower()
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


def _resolve_pad_position(pad_position: str | None) -> str:
    position = str(pad_position or "center").lower()
    valid_positions = {"center", "top_left", "top_right", "bottom_left", "bottom_right", "random"}
    if position not in valid_positions:
        raise ValueError(f"Unsupported pad_position: {position}. Expected one of {sorted(valid_positions)}.")
    return position


def _resolve_pad_fill(
    pad_value: int | float | list[int] | list[float] | tuple[int, ...] | tuple[float, ...] | None,
) -> float | tuple[float, ...]:
    if isinstance(pad_value, (list, tuple)):
        return tuple(float(value) for value in pad_value)
    return float(pad_value or 0)


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
    interpolation = _resolve_interpolation(whole_image_config.get("interpolation", "area"))
    pad_position = _resolve_pad_position(whole_image_config.get("pad_position", "center"))
    pad_fill = _resolve_pad_fill(whole_image_config.get("pad_value", 0))

    transforms: list[A.BasicTransform] = [
        A.LongestMaxSize(max_size=image_size, interpolation=interpolation),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=pad_fill,
            position=pad_position,
        ),
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