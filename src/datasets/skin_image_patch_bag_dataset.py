from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError

Image.MAX_IMAGE_PIXELS = None
from torch.utils.data import DataLoader, Dataset

from src.datasets.transforms import build_patch_transforms, build_whole_image_transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_COLUMNS = {"patch_path", "label", "source_image"}
SUPPORTED_SAMPLE_STRATEGIES = {"random", "head"}


@dataclass(frozen=True)
class PatchRecord:
    patch_path: str
    patch_row: int | None = None
    patch_col: int | None = None


@dataclass(frozen=True)
class ImagePatchBagRecord:
    source_image: str
    label_name: str
    label_idx: int
    patch_records: tuple[PatchRecord, ...]
    patient_id: str = ""


class SkinImagePatchBagDataset(Dataset):
    """Group patch split rows by source_image for dual-branch image-level classification."""

    def __init__(
        self,
        csv_file: str | Path,
        mode: str,
        transform_config: dict[str, Any] | None = None,
        dual_branch_config: dict[str, Any] | None = None,
        label_mapping: dict[str, int] | None = None,
        label_mapping_path: str | Path | None = None,
        project_root: str | Path | None = None,
        whole_image_transform: Any = None,
        patch_transform: Any = None,
        seed: int = 42,
    ) -> None:
        self.csv_file = Path(csv_file)
        self.mode = mode.lower()
        if self.mode not in {"train", "val", "test"}:
            raise ValueError(f"mode must be one of train/val/test, got: {mode}")

        self.project_root = Path(project_root) if project_root is not None else PROJECT_ROOT
        self.transform_config = transform_config or {}
        self.dual_branch_config = dual_branch_config or {}
        self.seed = int(seed)
        self.epoch = 0

        self.max_patches_per_image = self._resolve_max_patches_per_image(self.dual_branch_config)
        self.patch_sample_strategy = str(self.dual_branch_config.get("patch_sample_strategy", "random")).lower()
        if self.patch_sample_strategy not in SUPPORTED_SAMPLE_STRATEGIES:
            raise ValueError(
                f"Unsupported dual_branch.patch_sample_strategy: {self.patch_sample_strategy}. "
                f"Expected one of {sorted(SUPPORTED_SAMPLE_STRATEGIES)}."
            )

        self.samples = self._load_samples(self.csv_file)
        self.label_mapping = self._build_label_mapping(label_mapping, label_mapping_path)
        self.records = self._build_group_records(self.samples)
        self.whole_image_transform = whole_image_transform or build_whole_image_transforms(
            self.mode,
            self.transform_config,
            whole_image_size=self.dual_branch_config.get("whole_image_size"),
        )
        self.patch_transform = patch_transform or build_patch_transforms(self.mode, self.transform_config)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        whole_image_path = self._resolve_image_path(record.source_image)
        whole_image = self._load_rgb_image(whole_image_path, image_role="source_image", index=index)
        if self.whole_image_transform is not None:
            whole_image = self.whole_image_transform(image=whole_image)["image"]

        patch_records = self._select_patch_records(record.patch_records, sample_index=index)
        patch_images: list[torch.Tensor] = []
        patch_paths: list[str] = []
        patch_coords: list[list[float]] = []
        num_rows, num_cols = self._infer_patch_grid(record.patch_records)

        for patch_record in patch_records:
            patch_path = self._resolve_patch_path(patch_record.patch_path)
            patch_image = self._load_rgb_image(patch_path, image_role="patch", index=index)
            if self.patch_transform is not None:
                patch_image = self.patch_transform(image=patch_image)["image"]
            patch_images.append(patch_image)
            patch_paths.append(patch_record.patch_path)
            if patch_record.patch_row is not None and patch_record.patch_col is not None:
                patch_coords.append(
                    [
                        (patch_record.patch_col + 0.5) / max(1, num_cols),
                        (patch_record.patch_row + 0.5) / max(1, num_rows),
                    ]
                )

        if not patch_images:
            raise ValueError(
                f"source_image '{record.source_image}' in {self.csv_file} produced an empty patch set. "
                "Each image must keep at least one patch for dual-branch training."
            )

        sample = {
            "whole_image": whole_image,
            "patch_images": torch.stack(patch_images, dim=0),
            "label": int(record.label_idx),
            "label_name": record.label_name,
            "source_image": record.source_image,
            "patch_paths": patch_paths,
            "num_patches": len(patch_paths),
        }
        if record.patient_id:
            sample["patient_id"] = record.patient_id
        if patch_coords:
            sample["patch_coords"] = torch.tensor(patch_coords, dtype=torch.float32)
        return sample

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def get_statistics(self) -> dict[str, float | int]:
        raw_counts = np.asarray([len(record.patch_records) for record in self.records], dtype=np.float64)
        effective_counts = np.asarray([self._effective_patch_count(len(record.patch_records)) for record in self.records], dtype=np.float64)
        num_images = int(len(self.records))
        if num_images == 0:
            return {
                "num_images": 0,
                "total_patches": 0,
                "avg_patches_per_image": 0.0,
                "min_patches_per_image": 0,
                "max_patches_per_image": 0,
                "avg_selected_patches_per_image": 0.0,
                "min_selected_patches_per_image": 0,
                "max_selected_patches_per_image": 0,
            }

        return {
            "num_images": num_images,
            "total_patches": int(raw_counts.sum()),
            "avg_patches_per_image": float(raw_counts.mean()),
            "min_patches_per_image": int(raw_counts.min()),
            "max_patches_per_image": int(raw_counts.max()),
            "avg_selected_patches_per_image": float(effective_counts.mean()),
            "min_selected_patches_per_image": int(effective_counts.min()),
            "max_selected_patches_per_image": int(effective_counts.max()),
        }

    def _load_samples(self, csv_file: Path) -> pd.DataFrame:
        if not csv_file.exists():
            raise FileNotFoundError(f"Split CSV not found: {csv_file}")

        dataframe = pd.read_csv(csv_file)
        missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
        if missing_columns:
            raise ValueError(f"Split CSV is missing required columns for dual_branch mode: {sorted(missing_columns)}")

        dataframe["patch_path"] = dataframe["patch_path"].astype(str)
        dataframe["label"] = dataframe["label"].astype(str)
        dataframe["source_image"] = dataframe["source_image"].fillna("").astype(str).str.strip()
        if (dataframe["source_image"] == "").any():
            raise ValueError(
                f"Split CSV contains empty source_image values: {csv_file}. "
                "dual_branch mode requires a valid source_image for every row."
            )
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

    def _build_group_records(self, dataframe: pd.DataFrame) -> list[ImagePatchBagRecord]:
        records: list[ImagePatchBagRecord] = []
        for source_image, group in dataframe.groupby("source_image", sort=True):
            label_names = sorted(group["label"].astype(str).unique().tolist())
            if len(label_names) != 1:
                raise ValueError(
                    "dual_branch mode expects a single image-level label per source_image, but found "
                    f"multiple labels for '{source_image}': {label_names}"
                )
            label_name = label_names[0]
            if label_name not in self.label_mapping:
                raise KeyError(
                    f"Label '{label_name}' from {self.csv_file} is missing in label mapping: {self.label_mapping}"
                )

            patient_id = ""
            if "patient_id" in group.columns:
                patient_candidates = [value for value in group["patient_id"].astype(str).tolist() if value]
                patient_id = patient_candidates[0] if patient_candidates else ""

            patch_table = group.copy()
            sort_columns = ["patch_path"]
            if "patch_row" in patch_table.columns and "patch_col" in patch_table.columns:
                patch_table["patch_row"] = patch_table["patch_row"].fillna(-1).astype(int)
                patch_table["patch_col"] = patch_table["patch_col"].fillna(-1).astype(int)
                sort_columns = ["patch_row", "patch_col", "patch_path"]
            patch_table = patch_table.sort_values(sort_columns, kind="stable")

            patch_records: list[PatchRecord] = []
            for _, row in patch_table.iterrows():
                patch_row = None
                patch_col = None
                if "patch_row" in patch_table.columns and int(row["patch_row"]) >= 0:
                    patch_row = int(row["patch_row"])
                if "patch_col" in patch_table.columns and int(row["patch_col"]) >= 0:
                    patch_col = int(row["patch_col"])
                patch_records.append(
                    PatchRecord(
                        patch_path=str(row["patch_path"]),
                        patch_row=patch_row,
                        patch_col=patch_col,
                    )
                )

            records.append(
                ImagePatchBagRecord(
                    source_image=str(source_image),
                    label_name=label_name,
                    label_idx=int(self.label_mapping[label_name]),
                    patch_records=tuple(patch_records),
                    patient_id=patient_id,
                )
            )
        return records

    def _resolve_max_patches_per_image(self, dual_branch_config: dict[str, Any]) -> int | None:
        value = dual_branch_config.get("max_patches_per_image")
        if value in (None, "", 0):
            return None
        resolved = int(value)
        return resolved if resolved > 0 else None

    def _effective_patch_count(self, raw_patch_count: int) -> int:
        if self.max_patches_per_image is None:
            return int(raw_patch_count)
        return int(min(raw_patch_count, self.max_patches_per_image))

    def _select_patch_records(
        self,
        patch_records: tuple[PatchRecord, ...],
        sample_index: int,
    ) -> list[PatchRecord]:
        selected_records = list(patch_records)
        if self.max_patches_per_image is None or len(selected_records) <= self.max_patches_per_image:
            return selected_records

        if self.mode == "train" and self.patch_sample_strategy == "random":
            generator = random.Random(self.seed + (self.epoch * 1000003) + sample_index)
            return generator.sample(selected_records, self.max_patches_per_image)

        return selected_records[: self.max_patches_per_image]

    def _infer_patch_grid(self, patch_records: tuple[PatchRecord, ...]) -> tuple[int, int]:
        rows = [record.patch_row for record in patch_records if record.patch_row is not None]
        cols = [record.patch_col for record in patch_records if record.patch_col is not None]
        num_rows = (max(rows) + 1) if rows else 1
        num_cols = (max(cols) + 1) if cols else 1
        return int(num_rows), int(num_cols)

    def _resolve_patch_path(self, patch_path: str) -> Path:
        return self._resolve_relative_or_absolute_path(patch_path)

    def _resolve_image_path(self, source_image: str) -> Path:
        return self._resolve_relative_or_absolute_path(source_image)

    def _resolve_relative_or_absolute_path(self, path_value: str) -> Path:
        candidate = Path(path_value)
        if candidate.is_absolute():
            return candidate

        csv_relative = self.csv_file.parent / candidate
        if csv_relative.exists():
            return csv_relative.resolve()

        return (self.project_root / candidate).resolve()

    def _load_rgb_image(self, image_path: Path, image_role: str, index: int) -> np.ndarray:
        try:
            with Image.open(image_path) as image:
                return np.array(image.convert("RGB"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Failed to read {image_role} because the file does not exist: {image_path} "
                f"(csv: {self.csv_file}, index: {index})"
            ) from exc
        except UnidentifiedImageError as exc:
            raise RuntimeError(
                f"Failed to decode {image_role}: {image_path} "
                f"(csv: {self.csv_file}, index: {index})"
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                f"Failed to open {image_role}: {image_path} "
                f"(csv: {self.csv_file}, index: {index})"
            ) from exc


def dual_branch_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    whole_images = torch.stack([sample["whole_image"] for sample in batch], dim=0)
    labels = torch.tensor([int(sample["label"]) for sample in batch], dtype=torch.long)

    collated = {
        "whole_image": whole_images,
        "patch_images": [sample["patch_images"] for sample in batch],
        "label": labels,
        "label_name": [sample["label_name"] for sample in batch],
        "source_image": [sample["source_image"] for sample in batch],
        "patch_paths": [list(sample["patch_paths"]) for sample in batch],
        "num_patches": torch.tensor([int(sample["num_patches"]) for sample in batch], dtype=torch.long),
    }

    if any("patient_id" in sample for sample in batch):
        collated["patient_id"] = [str(sample.get("patient_id", "")) for sample in batch]
    if any("patch_coords" in sample for sample in batch):
        collated["patch_coords"] = [sample.get("patch_coords") for sample in batch]
    return collated


def build_dual_branch_dataloader(
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
    dual_branch_config = config.get("dual_branch", {})
    seed = int(train_config.get("seed", 42))

    dataset = SkinImagePatchBagDataset(
        csv_file=csv_file,
        mode=mode,
        transform_config=transform_config,
        dual_branch_config=dual_branch_config,
        label_mapping_path=label_mapping_path or split_config.get("label_mapping_path"),
        project_root=project_root,
        seed=seed,
    )

    if shuffle is None:
        shuffle = mode.lower() == "train"
    if drop_last is None:
        drop_last = bool(mode.lower() == "train" and dataloader_config.get("drop_last", False))

    batch_size = int(dual_branch_config.get("batch_size", train_config.get("batch_size", dataloader_config.get("batch_size", 2))))
    num_workers = int(dual_branch_config.get("num_workers", train_config.get("num_workers", dataloader_config.get("num_workers", 0))))
    pin_memory = bool(dataloader_config.get("pin_memory", False))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=dual_branch_collate_fn,
    )

