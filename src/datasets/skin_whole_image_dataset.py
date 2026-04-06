from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageFile, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset

from src.datasets.transforms import build_whole_image_transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_COLUMNS = {"source_image", "label"}
SUCCESS_CACHE_STATUSES = {"success", "skipped_existing"}

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _configure_max_image_pixels(max_image_pixels: int | float | str | None) -> None:
    if max_image_pixels in (None, "", "null"):
        Image.MAX_IMAGE_PIXELS = None
        return
    Image.MAX_IMAGE_PIXELS = int(max_image_pixels)


def _resolve_image_size(whole_image_config: dict[str, Any]) -> int:
    image_size = int(whole_image_config.get("image_size", 512))
    if image_size <= 0:
        raise ValueError("whole_image.image_size must be a positive integer.")

    cache_config = whole_image_config.get("cache", {})
    deprecated_cache_size = cache_config.get("size")
    if deprecated_cache_size not in (None, "", "null"):
        deprecated_cache_size = int(deprecated_cache_size)
        if deprecated_cache_size != image_size:
            raise ValueError(
                "whole_image.cache.size is deprecated and must match whole_image.image_size. "
                f"Got cache.size={deprecated_cache_size} and image_size={image_size}."
            )
        logging.warning(
            "whole_image.cache.size is deprecated. whole_image.image_size=%d is the single geometry size used "
            "for cache generation and model input.",
            image_size,
        )
    return image_size


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
        self.whole_image_config = whole_image_config or {}
        self.cache_config = self.whole_image_config.get("cache", {})
        self.image_size = _resolve_image_size(self.whole_image_config)
        self.cache_enabled = bool(self.cache_config.get("enabled", False))
        self.use_cached_images = self.cache_enabled and bool(self.cache_config.get("use_cached_for_training", True))
        self.allow_raw_fallback = bool(self.cache_config.get("allow_raw_fallback", False))
        self.require_cached_images = self.use_cached_images and not self.allow_raw_fallback

        _configure_max_image_pixels(self.whole_image_config.get("max_image_pixels"))

        self.samples = self._load_samples(self.csv_file)
        self.label_mapping = self._build_label_mapping(label_mapping, label_mapping_path)
        self.cached_image_mapping = self._load_cached_image_mapping()
        self._missing_cached_paths: set[str] = set()
        self._raw_fallback_warnings: set[str] = set()
        self.transform = transform or build_whole_image_transforms(
            self.mode,
            transform_cfg=transform_config,
            whole_image_config=self.whole_image_config,
        )

        logging.info(
            "Whole-image dataset | mode=%s samples=%d image_size=%d cache_enabled=%s use_cached_for_training=%s allow_raw_fallback=%s",
            self.mode,
            len(self.samples),
            self.image_size,
            self.cache_enabled,
            self.use_cached_images,
            self.allow_raw_fallback,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.samples.iloc[index]
        image_path, using_cached_image = self._resolve_preferred_image_path(str(row["source_image"]))
        image_kind = "cached whole image" if using_cached_image else "source image"

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                width, height = image.size
                image = np.array(image)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Failed to read {image_kind} because the file does not exist: {image_path} "
                f"(csv: {self.csv_file}, index: {index})"
            ) from exc
        except UnidentifiedImageError as exc:
            raise RuntimeError(
                f"Failed to decode {image_kind}: {image_path} "
                f"(csv: {self.csv_file}, index: {index})"
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                f"Failed to open {image_kind}: {image_path} "
                f"(csv: {self.csv_file}, index: {index})"
            ) from exc

        if height != self.image_size or width != self.image_size:
            raise RuntimeError(
                f"Whole-image input must already be preprocessed to {self.image_size}x{self.image_size}, but got "
                f"{width}x{height} from {image_kind}: {image_path}. Run scripts/prepare_whole_images.py with "
                "the current whole_image.image_size before training, validation, or testing."
            )

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

    def _load_cached_image_mapping(self) -> dict[str, str]:
        if not self.use_cached_images:
            return {}

        metadata_path_value = self.cache_config.get("metadata_path")
        if not metadata_path_value:
            message = "Whole-image cache is enabled but whole_image.cache.metadata_path is not configured."
            if self.require_cached_images:
                raise FileNotFoundError(message)
            logging.warning("%s Falling back to raw source images.", message)
            return {}

        metadata_path = self._resolve_project_path(metadata_path_value)
        if not metadata_path.exists():
            message = f"Whole-image cache metadata not found: {metadata_path}"
            if self.require_cached_images:
                raise FileNotFoundError(message)
            logging.warning("%s Falling back to raw source images.", message)
            return {}

        metadata = pd.read_csv(metadata_path)
        required_columns = {"source_image", "cached_image_path"}
        missing_columns = required_columns.difference(metadata.columns)
        if missing_columns:
            message = (
                f"Whole-image cache metadata is missing required columns {sorted(missing_columns)}: {metadata_path}"
            )
            if self.require_cached_images:
                raise ValueError(message)
            logging.warning("%s Falling back to raw source images.", message)
            return {}

        metadata["source_image"] = metadata["source_image"].fillna("").astype(str).str.strip()
        metadata["cached_image_path"] = metadata["cached_image_path"].fillna("").astype(str).str.strip()
        if "status" in metadata.columns:
            metadata["status"] = metadata["status"].fillna("").astype(str).str.strip().str.lower()
            metadata = metadata[metadata["status"].isin(SUCCESS_CACHE_STATUSES)]
        metadata = metadata[metadata["cached_image_path"].str.len() > 0]

        if self.require_cached_images:
            self._validate_cached_metadata(metadata, metadata_path)

        mapping: dict[str, str] = {}
        for row in metadata.itertuples(index=False):
            source_image = str(row.source_image).strip()
            cached_image_path = str(row.cached_image_path).strip()
            if not source_image or not cached_image_path:
                continue
            mapping[source_image] = cached_image_path
            mapping[str(self._resolve_image_path(source_image))] = cached_image_path

        logging.info("Loaded %d cached whole-image paths from %s", len(mapping), metadata_path)
        return mapping

    def _validate_cached_metadata(self, metadata: pd.DataFrame, metadata_path: Path) -> None:
        if metadata.empty and len(self.samples) > 0:
            raise RuntimeError(
                f"Whole-image cache metadata has no usable cached entries: {metadata_path}. "
                "Run scripts/prepare_whole_images.py before training."
            )

        metadata_by_source = metadata.drop_duplicates(subset="source_image", keep="first").set_index("source_image")
        missing_mappings: list[str] = []
        missing_files: list[str] = []
        wrong_sizes: list[str] = []

        for source_image in self.samples["source_image"].astype(str).tolist():
            if source_image not in metadata_by_source.index:
                missing_mappings.append(source_image)
                continue

            row = metadata_by_source.loc[source_image]
            cached_image_path = str(row.get("cached_image_path", "")).strip()
            if not cached_image_path:
                missing_mappings.append(source_image)
                continue

            cached_path = self._resolve_image_path(cached_image_path)
            if not cached_path.exists():
                missing_files.append(str(cached_path))

            cached_width = row.get("cached_width")
            cached_height = row.get("cached_height")
            target_size = row.get("target_size")
            if pd.notna(cached_width) and pd.notna(cached_height):
                if int(cached_width) != self.image_size or int(cached_height) != self.image_size:
                    wrong_sizes.append(f"{source_image} -> {int(cached_width)}x{int(cached_height)}")
                    continue
            if pd.notna(target_size) and int(target_size) != self.image_size:
                wrong_sizes.append(f"{source_image} -> target_size={int(target_size)}")

        error_lines: list[str] = []
        if missing_mappings:
            preview = ", ".join(missing_mappings[:3])
            error_lines.append(
                f"missing cached_image_path for {len(missing_mappings)} source_image entries (examples: {preview})"
            )
        if missing_files:
            preview = ", ".join(missing_files[:3])
            error_lines.append(
                f"missing cached files for {len(missing_files)} metadata rows (examples: {preview})"
            )
        if wrong_sizes:
            preview = ", ".join(wrong_sizes[:3])
            error_lines.append(
                f"cached files with unexpected size for {len(wrong_sizes)} source_image entries (examples: {preview})"
            )

        if error_lines:
            raise RuntimeError(
                "Whole-image cache validation failed while allow_raw_fallback=false. "
                + " | ".join(error_lines)
                + ". Re-run scripts/prepare_whole_images.py with the current whole_image.image_size."
            )

    def _resolve_preferred_image_path(self, source_image: str) -> tuple[Path, bool]:
        raw_path = self._resolve_image_path(source_image)
        cached_image_value = self.cached_image_mapping.get(source_image) or self.cached_image_mapping.get(str(raw_path))
        if cached_image_value:
            cached_path = self._resolve_image_path(cached_image_value)
            if cached_path.exists():
                return cached_path, True

            if self.require_cached_images:
                raise FileNotFoundError(
                    f"Cached whole-image file is missing for source_image={source_image}: {cached_path}. "
                    "Run scripts/prepare_whole_images.py before training."
                )

            cache_key = str(cached_path)
            if cache_key not in self._missing_cached_paths:
                logging.warning(
                    "Cached whole-image file does not exist: %s. Falling back to raw source_image %s",
                    cached_path,
                    raw_path,
                )
                self._missing_cached_paths.add(cache_key)
            return raw_path, False

        if self.require_cached_images:
            raise FileNotFoundError(
                f"No cached whole-image path found for source_image={source_image} in whole_image.cache.metadata_path. "
                "Run scripts/prepare_whole_images.py before training."
            )

        if self.use_cached_images:
            warning_key = str(raw_path)
            if warning_key not in self._raw_fallback_warnings:
                logging.warning(
                    "No cached whole-image path found for source_image=%s. Falling back to raw source image %s because "
                    "whole_image.cache.allow_raw_fallback=true.",
                    source_image,
                    raw_path,
                )
                self._raw_fallback_warnings.add(warning_key)
        return raw_path, False

    def _resolve_project_path(self, path_value: str | Path) -> Path:
        candidate = Path(path_value)
        if candidate.is_absolute():
            return candidate.resolve()
        return (self.project_root / candidate).resolve()

    def _resolve_image_path(self, image_path: str | Path) -> Path:
        candidate = Path(image_path)
        if candidate.is_absolute():
            return candidate.resolve()

        csv_relative = (self.csv_file.parent / candidate).resolve()
        if csv_relative.exists():
            return csv_relative

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
    split_config = config.get("build_image_splits", {})
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
