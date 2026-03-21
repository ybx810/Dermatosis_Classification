from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
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


@dataclass(frozen=True)
class SourceImageRecord:
    source_image: str
    label: str
    patient_id: str
    patch_paths: tuple[str, ...]


@dataclass(frozen=True)
class MILBagRecord:
    source_image: str
    label: str
    patient_id: str
    patch_paths: tuple[str, ...]
    bag_index: int
    num_instances: int


class SkinMILDataset(Dataset):
    """MIL dataset that splits each source_image into multiple fixed-size sub-bags."""

    def __init__(
        self,
        csv_file: str | Path,
        mode: str,
        transform: Any = None,
        transform_config: dict[str, Any] | None = None,
        label_mapping: dict[str, int] | None = None,
        label_mapping_path: str | Path | None = None,
        project_root: str | Path | None = None,
        bag_size: int = 512,
        drop_last_incomplete_bag: bool = False,
        shuffle_instances_within_image: bool | None = None,
        seed: int = 42,
        max_instances_per_bag: int | None = None,
        min_instances_per_bag: int | None = None,
        sample_strategy: str | None = None,
    ) -> None:
        self.csv_file = Path(csv_file)
        self.mode = mode.lower()
        if self.mode not in {"train", "val", "test"}:
            raise ValueError(f"mode must be one of train/val/test, got: {mode}")
        if bag_size <= 0:
            raise ValueError(f"bag_size must be positive, got: {bag_size}")

        self.project_root = Path(project_root) if project_root is not None else PROJECT_ROOT
        self.bag_size = int(bag_size)
        self.drop_last_incomplete_bag = bool(drop_last_incomplete_bag)
        self.shuffle_instances_within_image = (
            bool(shuffle_instances_within_image)
            if shuffle_instances_within_image is not None
            else self.mode == "train"
        )
        self.seed = int(seed)

        if max_instances_per_bag is not None:
            logging.warning(
                "mil.max_instances_per_bag is ignored in the multi-sub-bag MIL dataset. Use mil.bag_size instead."
            )
        if min_instances_per_bag not in {None, 1}:
            logging.warning(
                "mil.min_instances_per_bag is ignored in the multi-sub-bag MIL dataset. Every source_image with at least one patch will generate at least one bag."
            )
        if sample_strategy not in {None, "none", "", "null"}:
            logging.warning(
                "mil.sample_strategy is ignored in the multi-sub-bag MIL dataset. Instance grouping is controlled by mil.bag_size and mil.shuffle_instances_within_image."
            )

        self.source_records = self._load_source_image_records(self.csv_file)
        self.label_mapping = self._build_label_mapping(label_mapping, label_mapping_path)
        self.transform = transform or build_patch_transforms(self.mode, transform_config)
        self.bag_records: list[MILBagRecord] = []
        self.refresh_bag_metadata(epoch=0)

    def __len__(self) -> int:
        return len(self.bag_records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        bag_record = self.bag_records[index]

        images: list[torch.Tensor] = []
        for patch_path_str in bag_record.patch_paths:
            patch_path = self._resolve_patch_path(patch_path_str)
            try:
                with Image.open(patch_path) as image:
                    image = image.convert("RGB")
                    image_array = np.array(image)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"Failed to read patch image because the file does not exist: {patch_path} "
                    f"(csv: {self.csv_file}, bag index: {index}, source_image: {bag_record.source_image})"
                ) from exc
            except UnidentifiedImageError as exc:
                raise RuntimeError(
                    f"Failed to decode patch image: {patch_path} "
                    f"(csv: {self.csv_file}, bag index: {index}, source_image: {bag_record.source_image})"
                ) from exc
            except OSError as exc:
                raise RuntimeError(
                    f"Failed to open patch image: {patch_path} "
                    f"(csv: {self.csv_file}, bag index: {index}, source_image: {bag_record.source_image})"
                ) from exc

            if self.transform is not None:
                image_tensor = self.transform(image=image_array)["image"]
            else:
                image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
            images.append(image_tensor)

        if not images:
            raise RuntimeError(
                f"Bag at index {index} has no valid patch instances. source_image={bag_record.source_image}, bag_index={bag_record.bag_index}"
            )

        label_name = str(bag_record.label)
        if label_name not in self.label_mapping:
            raise KeyError(
                f"Label '{label_name}' from {self.csv_file} is missing in label mapping: {self.label_mapping}"
            )

        sample = {
            "images": torch.stack(images, dim=0),
            "label": int(self.label_mapping[label_name]),
            "label_name": label_name,
            "source_image": bag_record.source_image,
            "bag_index": int(bag_record.bag_index),
            "num_instances": int(bag_record.num_instances),
            "patch_paths": list(bag_record.patch_paths),
        }
        if bag_record.patient_id:
            sample["patient_id"] = bag_record.patient_id
        return sample

    def refresh_bag_metadata(self, epoch: int | None = None) -> None:
        """Rebuild sub-bag metadata.

        In train mode this reshuffles instances within each source_image before chunking,
        which gives different sub-bag boundaries across epochs while keeping the run reproducible.
        """

        rng_seed = self.seed if epoch is None else self.seed + int(epoch)
        rng = random.Random(rng_seed)
        bag_records: list[MILBagRecord] = []

        for source_record in self.source_records:
            patch_paths = list(source_record.patch_paths)
            if self.mode == "train" and self.shuffle_instances_within_image:
                rng.shuffle(patch_paths)

            bag_chunks = self._chunk_patch_paths(patch_paths)
            for bag_index, bag_paths in enumerate(bag_chunks):
                bag_records.append(
                    MILBagRecord(
                        source_image=source_record.source_image,
                        label=source_record.label,
                        patient_id=source_record.patient_id,
                        patch_paths=tuple(bag_paths),
                        bag_index=bag_index,
                        num_instances=len(bag_paths),
                    )
                )

        if not bag_records:
            raise ValueError(
                f"No MIL bags were created from {self.csv_file}. "
                f"Check source_image availability and mil.bag_size={self.bag_size}."
            )
        self.bag_records = bag_records

    def get_statistics(self) -> dict[str, float | int]:
        bag_counts_by_image = pd.Series([record.source_image for record in self.bag_records]).value_counts()
        instance_counts = [record.num_instances for record in self.bag_records]

        return {
            "num_source_images": int(len(self.source_records)),
            "num_bags": int(len(self.bag_records)),
            "avg_bags_per_image": float(bag_counts_by_image.mean()) if not bag_counts_by_image.empty else 0.0,
            "avg_instances_per_bag": float(sum(instance_counts) / len(instance_counts)) if instance_counts else 0.0,
            "min_instances_per_bag": int(min(instance_counts)) if instance_counts else 0,
            "max_instances_per_bag": int(max(instance_counts)) if instance_counts else 0,
            "bag_size": int(self.bag_size),
        }

    def _load_source_image_records(self, csv_file: Path) -> list[SourceImageRecord]:
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

        source_records: list[SourceImageRecord] = []
        for source_image, group_df in dataframe.groupby("source_image", sort=True):
            label_values = sorted(group_df["label"].unique().tolist())
            if len(label_values) != 1:
                raise ValueError(
                    f"All patches inside one source_image must share the same label for MIL. "
                    f"source_image={source_image!r}, labels={label_values}"
                )

            patient_ids = [value for value in group_df["patient_id"].tolist() if value]
            patient_id = sorted(set(patient_ids))[0] if patient_ids else ""
            patch_paths = tuple(sorted(group_df["patch_path"].tolist()))
            if not patch_paths:
                continue

            source_records.append(
                SourceImageRecord(
                    source_image=source_image,
                    label=label_values[0],
                    patient_id=patient_id,
                    patch_paths=patch_paths,
                )
            )

        if not source_records:
            raise ValueError(f"No source_image records were created from {csv_file}.")
        return source_records

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

        labels = sorted({record.label for record in self.source_records})
        return {label: idx for idx, label in enumerate(labels)}

    def _chunk_patch_paths(self, patch_paths: list[str]) -> list[list[str]]:
        if not patch_paths:
            return []

        chunks = [
            patch_paths[start_index : start_index + self.bag_size]
            for start_index in range(0, len(patch_paths), self.bag_size)
        ]
        if self.drop_last_incomplete_bag and len(chunks) > 1 and len(chunks[-1]) < self.bag_size:
            chunks = chunks[:-1]

        if not chunks:
            return [patch_paths]
        return chunks

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
        "bag_index": torch.tensor([item["bag_index"] for item in batch], dtype=torch.long),
        "num_instances": torch.tensor([item["num_instances"] for item in batch], dtype=torch.long),
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
        bag_size=int(mil_config.get("bag_size", 512)),
        drop_last_incomplete_bag=bool(mil_config.get("drop_last_incomplete_bag", False)),
        shuffle_instances_within_image=mil_config.get("shuffle_instances_within_image"),
        seed=int(train_config.get("seed", 42)),
        max_instances_per_bag=mil_config.get("max_instances_per_bag"),
        min_instances_per_bag=mil_config.get("min_instances_per_bag"),
        sample_strategy=mil_config.get("sample_strategy"),
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
