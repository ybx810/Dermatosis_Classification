from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset

from src.datasets.transforms import build_patch_transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_COLUMNS = {"patch_path", "label", "source_image"}
VALID_SAMPLE_STRATEGIES = {"none", "random"}


class SkinMILDataset(Dataset):
    """Bag-level dataset that groups patch samples by source_image for MIL training."""

    def __init__(
        self,
        csv_file: str | Path,
        mode: str,
        transform: Any = None,
        transform_config: dict[str, Any] | None = None,
        label_mapping: dict[str, int] | None = None,
        label_mapping_path: str | Path | None = None,
        project_root: str | Path | None = None,
        max_instances_per_bag: int | None = None,
        min_instances_per_bag: int = 1,
        sample_strategy: str = "none",
    ) -> None:
        self.csv_file = Path(csv_file)
        self.mode = mode.lower()
        if self.mode not in {"train", "val", "test"}:
            raise ValueError(f"mode must be one of train/val/test, got: {mode}")

        if min_instances_per_bag <= 0:
            raise ValueError("min_instances_per_bag must be a positive integer.")
        if max_instances_per_bag is not None and max_instances_per_bag <= 0:
            raise ValueError("max_instances_per_bag must be positive when provided.")
        if max_instances_per_bag is not None and max_instances_per_bag < min_instances_per_bag:
            raise ValueError("max_instances_per_bag must be greater than or equal to min_instances_per_bag.")

        self.sample_strategy = str(sample_strategy).lower()
        if self.sample_strategy not in VALID_SAMPLE_STRATEGIES:
            raise ValueError(
                f"sample_strategy must be one of {sorted(VALID_SAMPLE_STRATEGIES)}, got: {sample_strategy}"
            )

        self.project_root = Path(project_root) if project_root is not None else PROJECT_ROOT
        self.max_instances_per_bag = max_instances_per_bag
        self.min_instances_per_bag = min_instances_per_bag
        self.samples = self._load_bags(self.csv_file)
        self.label_mapping = self._build_label_mapping(label_mapping, label_mapping_path)
        self.transform = transform or build_patch_transforms(self.mode, transform_config)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.samples.iloc[index]
        patch_paths = self._select_patch_paths(json.loads(row["patch_paths"]))

        images: list[torch.Tensor] = []
        for patch_path_str in patch_paths:
            patch_path = self._resolve_patch_path(patch_path_str)
            try:
                with Image.open(patch_path) as image:
                    image = image.convert("RGB")
                    image_array = np.array(image)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"Failed to read patch image because the file does not exist: {patch_path} "
                    f"(csv: {self.csv_file}, bag index: {index})"
                ) from exc
            except UnidentifiedImageError as exc:
                raise RuntimeError(
                    f"Failed to decode patch image: {patch_path} "
                    f"(csv: {self.csv_file}, bag index: {index})"
                ) from exc
            except OSError as exc:
                raise RuntimeError(
                    f"Failed to open patch image: {patch_path} "
                    f"(csv: {self.csv_file}, bag index: {index})"
                ) from exc

            if self.transform is not None:
                image_tensor = self.transform(image=image_array)["image"]
            else:
                image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
            images.append(image_tensor)

        if not images:
            raise RuntimeError(f"Bag at index {index} has no valid patch instances after sampling.")

        label_name = str(row["label"])
        if label_name not in self.label_mapping:
            raise KeyError(
                f"Label '{label_name}' from {self.csv_file} is missing in label mapping: {self.label_mapping}"
            )

        sample = {
            "images": torch.stack(images, dim=0),
            "label": int(self.label_mapping[label_name]),
            "label_name": label_name,
            "source_image": str(row["source_image"]),
            "patch_paths": patch_paths,
        }
        patient_id = str(row.get("patient_id", ""))
        if patient_id:
            sample["patient_id"] = patient_id
        return sample

    def _load_bags(self, csv_file: Path) -> pd.DataFrame:
        if not csv_file.exists():
            raise FileNotFoundError(f"Split CSV not found: {csv_file}")

        dataframe = pd.read_csv(csv_file)
        missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
        if missing_columns:
            raise ValueError(f"MIL split CSV is missing required columns: {sorted(missing_columns)}")

        dataframe["patch_path"] = dataframe["patch_path"].astype(str)
        dataframe["label"] = dataframe["label"].astype(str)
        dataframe["source_image"] = dataframe["source_image"].fillna("").astype(str).str.strip()
        if (dataframe["source_image"].str.len() == 0).any():
            raise ValueError(f"MIL dataset requires non-empty source_image values in {csv_file}.")
        if "patient_id" in dataframe.columns:
            dataframe["patient_id"] = dataframe["patient_id"].fillna("").astype(str).str.strip()
        else:
            dataframe["patient_id"] = ""

        bag_rows: list[dict[str, Any]] = []
        for source_image, group_df in dataframe.groupby("source_image", sort=True):
            label_values = sorted(group_df["label"].unique().tolist())
            if len(label_values) != 1:
                raise ValueError(
                    f"All patches inside one MIL bag must share the same label. "
                    f"source_image={source_image!r}, labels={label_values}"
                )

            patch_paths = sorted(group_df["patch_path"].tolist())
            if len(patch_paths) < self.min_instances_per_bag:
                continue

            patient_ids = [value for value in group_df["patient_id"].tolist() if value]
            patient_id = sorted(set(patient_ids))[0] if patient_ids else ""
            bag_rows.append(
                {
                    "source_image": source_image,
                    "label": label_values[0],
                    "patient_id": patient_id,
                    "patch_paths": json.dumps(patch_paths, ensure_ascii=False),
                    "num_instances": len(patch_paths),
                }
            )

        if not bag_rows:
            raise ValueError(
                f"No MIL bags were created from {csv_file}. "
                f"Check source_image availability and min_instances_per_bag={self.min_instances_per_bag}."
            )

        return pd.DataFrame(bag_rows).sort_values(["label", "source_image"]).reset_index(drop=True)

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

    def _select_patch_paths(self, patch_paths: list[str]) -> list[str]:
        if self.max_instances_per_bag is None or len(patch_paths) <= self.max_instances_per_bag:
            return patch_paths

        if self.mode == "train" and self.sample_strategy == "random":
            sampled_indices = torch.randperm(len(patch_paths))[: self.max_instances_per_bag].tolist()
            sampled_paths = [patch_paths[index] for index in sampled_indices]
            return sorted(sampled_paths)

        return patch_paths[: self.max_instances_per_bag]

    def _resolve_patch_path(self, patch_path: str) -> Path:
        candidate = Path(patch_path)
        if candidate.is_absolute():
            return candidate

        csv_relative = self.csv_file.parent / candidate
        if csv_relative.exists():
            return csv_relative.resolve()

        return (self.project_root / candidate).resolve()


def mil_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": [item["images"] for item in batch],
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "label_name": [item["label_name"] for item in batch],
        "source_image": [item["source_image"] for item in batch],
        "patch_paths": [item["patch_paths"] for item in batch],
        "patient_id": [item.get("patient_id", "") for item in batch],
    }


def build_mil_dataloader(
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
    mil_config = config.get("mil", {})

    dataset = SkinMILDataset(
        csv_file=csv_file,
        mode=mode,
        transform_config=transform_config,
        label_mapping_path=label_mapping_path or split_config.get("label_mapping_path"),
        project_root=project_root,
        max_instances_per_bag=mil_config.get("max_instances_per_bag"),
        min_instances_per_bag=int(mil_config.get("min_instances_per_bag", 1)),
        sample_strategy=str(mil_config.get("sample_strategy", "none")),
    )

    if shuffle is None:
        shuffle = mode.lower() == "train"
    if drop_last is None:
        drop_last = bool(mode.lower() == "train" and dataloader_config.get("drop_last", False))

    batch_size = int(train_config.get("batch_size", dataloader_config.get("batch_size", 16)))
    num_workers = int(train_config.get("num_workers", dataloader_config.get("num_workers", 0)))
    pin_memory = bool(dataloader_config.get("pin_memory", False))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=mil_collate_fn,
    )
