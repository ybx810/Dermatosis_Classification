from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.transforms import build_patch_transforms
from src.utils.io import load_yaml


def main() -> None:
    config = load_yaml(PROJECT_ROOT / "configs" / "default.yaml")
    augmentation_config = copy.deepcopy(config.get("augmentation", {}))

    # Disable other stochastic geometry so the check isolates noise behavior.
    augmentation_config["resize_height"] = None
    augmentation_config["resize_width"] = None
    augmentation_config["crop_size"] = None
    augmentation_config["horizontal_flip"] = 0.0
    augmentation_config["vertical_flip"] = 0.0

    noise_config = dict(augmentation_config.get("noise", {}))
    noise_config.setdefault("enabled", True)
    noise_config.setdefault("type", "gaussian")
    noise_config.setdefault("p", 1.0)
    noise_config["p"] = 1.0
    augmentation_config["noise"] = noise_config

    train_transform = build_patch_transforms("train", augmentation_config)
    val_transform = build_patch_transforms("val", augmentation_config)

    sample_image = np.full((128, 128, 3), 128, dtype=np.uint8)

    train_outputs = [train_transform(image=sample_image)["image"] for _ in range(3)]
    val_outputs = [val_transform(image=sample_image)["image"] for _ in range(3)]

    train_equal_01 = torch.equal(train_outputs[0], train_outputs[1])
    train_equal_12 = torch.equal(train_outputs[1], train_outputs[2])
    val_equal_01 = torch.equal(val_outputs[0], val_outputs[1])
    val_equal_12 = torch.equal(val_outputs[1], val_outputs[2])

    train_diff_01 = float(torch.max(torch.abs(train_outputs[0] - train_outputs[1])).item())
    train_diff_12 = float(torch.max(torch.abs(train_outputs[1] - train_outputs[2])).item())
    val_diff_01 = float(torch.max(torch.abs(val_outputs[0] - val_outputs[1])).item())
    val_diff_12 = float(torch.max(torch.abs(val_outputs[1] - val_outputs[2])).item())

    train_has_random_difference = not train_equal_01 or not train_equal_12
    val_is_deterministic = val_equal_01 and val_equal_12

    print(f"train_equal_01: {train_equal_01}")
    print(f"train_equal_12: {train_equal_12}")
    print(f"train_max_abs_diff_01: {train_diff_01:.6f}")
    print(f"train_max_abs_diff_12: {train_diff_12:.6f}")
    print(f"val_equal_01: {val_equal_01}")
    print(f"val_equal_12: {val_equal_12}")
    print(f"val_max_abs_diff_01: {val_diff_01:.6f}")
    print(f"val_max_abs_diff_12: {val_diff_12:.6f}")
    print(f"train_has_random_difference: {train_has_random_difference}")
    print(f"val_is_deterministic: {val_is_deterministic}")

    if not train_has_random_difference:
        raise AssertionError("Expected train transform outputs to differ when Gaussian noise is enabled.")
    if not val_is_deterministic:
        raise AssertionError("Expected val transform outputs to stay identical across repeated calls.")


if __name__ == "__main__":
    main()
