from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset

from src.datasets.transforms import build_whole_image_transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_COLUMNS = {"source_image", "label"}


class SkinWholeImageDataset(Dataset):
    """Whole-image classification dataset backed by an image-level split CSV."""

    def __init__(
        self,
        csv_file: str | Path,
        mode: str,
        transform: Any = None,
        transform_config: dict[str, Any] | None = None,
        whole_image_config: dict[str, Any] | None = None,
        label_mapping: dict[str, int] | None = None,
        label_mapping_path: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> None:
        self.csv_file = Path(csv_file)
        self.mode = mode.lower()
        if self.mode not in {"train", "val", "test"}:
            raise ValueError(f"mode must be one of train/val/test, got: {mode}")

        self.project_root = Path(project_root) if project_root is not None else PROJECT_ROOT
        self.samples = self._load_samples(self.csv_file)
        self.label_mapping = self._build_label_mapping(label_mapping, label_mapping_path)
        self.transform = transform or build_whole_image_transforms(
            self.mode,
            transform_cfg=transform_config,
            whole_image_config=whole_image_config,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.samples.iloc[index]
        image_path = self._resolve_image_path(row["source_image"])

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image = np.array(image)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Failed to read source image because the file does not exist: {image_path} "
                f"(csv: {self.csv_file}, index: {index})"
            ) from exc
        except UnidentifiedImageError as exc:
            raise RuntimeError(
                f"Failed to decode source image: {image_path} "
                f"(csv: {self.csv_file}, index: {index})"
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                f"Failed to open source image: {image_path} "
                f"(csv: {self.csv_file}, index: {index})"
            ) from exc

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        label_name = str(row["label"])
        if label_name not in self.label_mapping:
            raise KeyError(
                f"Label '{label_name}' from {self.csv_file} is missing in label mapping: {self.label_mapping}"
            )

        sample = {
            "image": image,
            "label": int(self.label_mapping[label_name]),
            "label_name": label_name,
            "source_image": str(row["source_image"]),
        }
        if "patient_id" in row.index:
            sample["patient_id"] = "" if pd.isna(row["patient_id"]) else str(row["patient_id"])
        return sample

    def _load_samples(self, csv_file: Path) -> pd.DataFrame:
        if not csv_file.exists():
            raise FileNotFoundError(f"Split CSV not found: {csv_file}")

        dataframe = pd.read_csv(csv_file)
        missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
        if missing_columns:
            raise ValueError(f"Split CSV is missing required columns: {sorted(missing_columns)}")

        dataframe["source_image"] = dataframe["source_image"].fillna("").astype(str).str.strip()
        dataframe["label"] = dataframe["label"].astype(str)
        if (dataframe["source_image"].str.len() == 0).any():
            raise ValueError(f"Split CSV contains empty source_image values: {csv_file}")
        if "patient_id" in dataframe.columns:
            dataframe["patient_id"] = dataframe["patient_id"].fillna("").astype(str).str.strip()
        return dataframe.reset_index(drop=True)

    def _build_label_mapping(
        self,
        label_mapping: dict[str, int] | None,
        label_mapping_path: str | Path | None,
    ) -> dict[str, int]:
        if label_mapping is not None:
            return {str(label): int(index) for label, index in label_mapping.items()}

        if label_mapping_path is not None:
            mapping_path = Path(label_mapping_path)
            if not mapping_path.is_absolute():
                mapping_path = self.project_root / mapping_path
            payload = json.loads(mapping_path.read_text(encoding="utf-8"))
            if "label_to_index" in payload:
                payload = payload["label_to_index"]
            return {str(label): int(index) for label, index in payload.items()}

        labels = sorted(self.samples["label"].unique().tolist())
        return {label: idx for idx, label in enumerate(labels)}

    def _resolve_image_path(self, image_path: str) -> Path:
        candidate = Path(image_path)
        if candidate.is_absolute():
            return candidate

        csv_relative = self.csv_file.parent / candidate
        if csv_relative.exists():
            return csv_relative.resolve()

        return (self.project_root / candidate).resolve()


def build_whole_image_dataloader(
    csv_file: str | Path,
    mode: str,
    config: dict[str, Any],
    label_mapping_path: str | Path | None = None,
    shuffle: bool | None = None,
    drop_last: bool | None = None,
    project_root: str | Path | None = None,
) -> DataLoader:
    train_config = config.get("train", {})
    dataloader_config = config.get("dataloader", {})
    split_config = config.get("build_patch_splits", {})
    transform_config = config.get("augmentation", {})
    whole_image_config = config.get("whole_image", {})

    dataset = SkinWholeImageDataset(
        csv_file=csv_file,
        mode=mode,
        transform_config=transform_config,
        whole_image_config=whole_image_config,
        label_mapping_path=label_mapping_path or split_config.get("label_mapping_path"),
        project_root=project_root,
    )

    if shuffle is None:
        shuffle = mode.lower() == "train"
    if drop_last is None:
        drop_last = bool(mode.lower() == "train" and dataloader_config.get("drop_last", False))

    batch_size = int(
        whole_image_config.get("batch_size", train_config.get("batch_size", dataloader_config.get("batch_size", 16)))
    )
    num_workers = int(
        whole_image_config.get("num_workers", train_config.get("num_workers", dataloader_config.get("num_workers", 0)))
    )
    pin_memory = bool(dataloader_config.get("pin_memory", False))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )