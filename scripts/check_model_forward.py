from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.build_model import build_model
from src.utils.io import load_yaml


def main() -> None:
    config = load_yaml(PROJECT_ROOT / "configs" / "default.yaml")
    model = build_model(config)
    model.eval()

    batch_size = 2
    image_size = int(config.get("whole_image", {}).get("image_size", 512))
    dummy_input = torch.randn(batch_size, 3, image_size, image_size)

    with torch.no_grad():
        output = model(dummy_input)

    print("task_mode: whole_image")
    print(f"input shape: {tuple(dummy_input.shape)}")
    print(f"output shape: {tuple(output.shape)}")


if __name__ == "__main__":
    main()