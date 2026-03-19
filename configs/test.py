from pathlib import Path

from src.datasets.skin_patch_dataset import build_dataloader
from src.utils.io import load_yaml


def main():
    project_root = Path(r"E:\skin-image-classification")
    config = load_yaml(project_root / "configs" / "default.yaml")

    train_loader = build_dataloader(
        csv_file=project_root / "data" / "splits" / "train.csv",
        mode="train",
        config=config,
        project_root=project_root,
    )

    batch = next(iter(train_loader))
    print(batch["image"].shape)
    print(batch["label"].shape)


if __name__ == "__main__":
    main()
