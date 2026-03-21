from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.skin_mil_dataset import build_mil_dataloader
from src.models.build_model import build_model
from src.utils.io import load_yaml


def main() -> None:
    config = load_yaml(PROJECT_ROOT / "configs" / "default.yaml")
    config["task"] = {**config.get("task", {}), "mode": "mil"}
    config["dataloader"] = {**config.get("dataloader", {}), "pin_memory": False}
    config["train"] = {**config.get("train", {}), "batch_size": 2, "num_workers": 0}
    config["mil"] = {
        **config.get("mil", {}),
        "bag_size": 8,
        "drop_last_incomplete_bag": False,
        "shuffle_instances_within_image": False,
        "aggregate_logits": "mean",
    }

    split_dir = PROJECT_ROOT / config.get("build_patch_splits", {}).get("output_dir", "data/splits")
    label_mapping_path = PROJECT_ROOT / config.get("build_patch_splits", {}).get("label_mapping_path", "data/splits/label_mapping.json")
    val_csv = split_dir / "val.csv"

    loader = build_mil_dataloader(
        csv_file=val_csv,
        mode="val",
        config=config,
        label_mapping_path=label_mapping_path,
        project_root=PROJECT_ROOT,
        shuffle=False,
    )
    batch = next(iter(loader))

    model = build_model(config)
    model.eval()

    with torch.no_grad():
        output = model(batch["images"], return_attention=True, return_bag_embedding=True)

    print(f"num_bags: {len(batch['images'])}")
    print(f"bag_indices: {batch['bag_index'].tolist()}")
    print(f"num_instances: {batch['num_instances'].tolist()}")
    print(f"source_images: {batch['source_image']}")
    print(f"logits shape: {tuple(output['logits'].shape)}")
    print(f"bag embedding shape: {tuple(output['bag_embedding'].shape)}")
    print(f"attention shapes: {[tuple(weights.shape) for weights in output['attention_weights']]}")


if __name__ == "__main__":
    main()
