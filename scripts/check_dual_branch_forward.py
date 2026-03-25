from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import build_dual_branch_dataloader
from src.models import build_model
from src.utils.io import load_yaml


SUPPORTED_FUSIONS = ("concat", "weighted_sum", "cross_attention")


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> None:
    config = load_yaml(PROJECT_ROOT / "configs/default.yaml")
    config.setdefault("task", {})["mode"] = "dual_branch"
    config.setdefault("model", {})["pretrained"] = False
    config.setdefault("dual_branch", {})
    config["dual_branch"]["batch_size"] = 1
    config["dual_branch"]["num_workers"] = 0
    config["dual_branch"]["max_patches_per_image"] = min(int(config["dual_branch"].get("max_patches_per_image", 64)), 8)

    split_dir = _resolve_path(config.get("build_patch_splits", {}).get("output_dir", "data/splits"))
    label_mapping_path = _resolve_path(config.get("build_patch_splits", {}).get("label_mapping_path", "data/splits/label_mapping.json"))
    train_csv = split_dir / "train.csv"

    for fusion in SUPPORTED_FUSIONS:
        config["dual_branch"]["fusion"] = fusion
        dataloader = build_dual_branch_dataloader(
            csv_file=train_csv,
            mode="train",
            config=config,
            label_mapping_path=label_mapping_path,
            project_root=PROJECT_ROOT,
            shuffle=False,
        )
        batch = next(iter(dataloader))
        model = build_model(config)
        model.eval()
        with torch.no_grad():
            logits = model(batch["whole_image"], batch["patch_images"])

        print(f"fusion={fusion}")
        print(f"whole_image.shape: {tuple(batch['whole_image'].shape)}")
        print(f"patch_images.shapes: {[tuple(patch_tensor.shape) for patch_tensor in batch['patch_images']]}")
        print(f"logits.shape: {tuple(logits.shape)}")
        print("-")


if __name__ == "__main__":
    main()
