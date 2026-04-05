from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import load_yaml

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
SPLIT_NAMES = ("train", "val", "test")
COVERAGE_SPLITS = ("train", "val", "test")
LIMITED_IMAGE_PRIORITY = ("train", "test", "val")
EXTRA_IMAGE_PRIORITY = ("test", "train", "val")
EXPORT_COLUMNS = ["source_image", "label", "label_idx", "patient_id", "split"]


@dataclass(frozen=True)
class ImageSplitConfig:
    raw_dir: Path
    output_dir: Path
    label_mapping_path: Path
    summary_path: Path
    train_ratio: float
    val_ratio: float
    test_ratio: float
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build image-level train/val/test splits from raw whole images.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--raw-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--label-mapping-path", type=str, default=None)
    parser.add_argument("--summary-path", type=str, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def resolve_path(path_value: str | Path | None, default: str) -> Path:
    raw_path = Path(path_value or default)
    if raw_path.is_absolute():
        return raw_path
    return PROJECT_ROOT / raw_path


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratios = [train_ratio, val_ratio, test_ratio]
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Split ratios must be non-negative.")
    if math.isclose(sum(ratios), 0.0):
        raise ValueError("At least one split ratio must be positive.")


def build_config(args: argparse.Namespace) -> ImageSplitConfig:
    config_path = resolve_path(args.config, "configs/default.yaml")
    config = load_yaml(config_path) if config_path.exists() else {}

    data_cfg = config.get("data", {})
    split_cfg = config.get("build_image_splits", {})

    raw_dir = resolve_path(args.raw_dir or data_cfg.get("raw_dir"), "data/raw")
    output_dir = resolve_path(args.output_dir or split_cfg.get("output_dir") or data_cfg.get("split_dir"), "data/splits")
    label_mapping_path = resolve_path(
        args.label_mapping_path or split_cfg.get("label_mapping_path"),
        "data/splits/label_mapping.json",
    )
    summary_path = resolve_path(
        args.summary_path or split_cfg.get("summary_path"),
        "data/splits/split_summary.json",
    )
    train_ratio = float(args.train_ratio if args.train_ratio is not None else split_cfg.get("train_ratio", 0.7))
    val_ratio = float(args.val_ratio if args.val_ratio is not None else split_cfg.get("val_ratio", 0.15))
    test_ratio = float(args.test_ratio if args.test_ratio is not None else split_cfg.get("test_ratio", 0.15))
    seed = int(args.seed if args.seed is not None else split_cfg.get("seed", 42))

    validate_ratios(train_ratio, val_ratio, test_ratio)
    return ImageSplitConfig(
        raw_dir=raw_dir,
        output_dir=output_dir,
        label_mapping_path=label_mapping_path,
        summary_path=summary_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )


def discover_images(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw image directory does not exist: {raw_dir}")

    image_paths = [
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not image_paths:
        raise FileNotFoundError(f"No supported image files were found under: {raw_dir}")
    return sorted(image_paths)


def path_to_project_string(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def infer_label(image_path: Path, raw_dir: Path) -> str:
    relative_path = image_path.relative_to(raw_dir)
    if len(relative_path.parts) > 1:
        return str(relative_path.parts[0])
    if image_path.parent != raw_dir:
        return str(image_path.parent.name)
    return "unknown"


def infer_patient_id(image_path: Path) -> str:
    tokens = [token for token in re.split(r"[_\-\s]+", image_path.stem) if token]
    return tokens[0] if tokens else ""


def build_image_dataframe(raw_dir: Path) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    for image_path in discover_images(raw_dir):
        records.append(
            {
                "source_image": path_to_project_string(image_path),
                "label": infer_label(image_path, raw_dir),
                "patient_id": infer_patient_id(image_path),
            }
        )

    dataframe = pd.DataFrame(records)
    dataframe["source_image"] = dataframe["source_image"].astype(str)
    dataframe["label"] = dataframe["label"].astype(str)
    dataframe["patient_id"] = dataframe["patient_id"].fillna("").astype(str)
    dataframe = dataframe.drop_duplicates(subset="source_image", keep="first").reset_index(drop=True)
    return dataframe


def build_label_mapping(dataframe: pd.DataFrame) -> dict[str, int]:
    labels = sorted(dataframe["label"].unique().tolist())
    return {label: index for index, label in enumerate(labels)}


def build_split_ratios(config: ImageSplitConfig) -> dict[str, float]:
    ratios = {
        "train": float(config.train_ratio),
        "val": float(config.val_ratio),
        "test": float(config.test_ratio),
    }
    total = sum(ratios.values())
    return {split_name: ratio / total for split_name, ratio in ratios.items()}


def allocate_remainder_by_priority(counts: dict[str, int], remainder: int, priority: tuple[str, ...]) -> None:
    priority_index = 0
    while remainder > 0:
        split_name = priority[priority_index % len(priority)]
        counts[split_name] += 1
        remainder -= 1
        priority_index += 1


def select_splits_for_limited_images(num_images: int, split_ratios: dict[str, float]) -> list[str]:
    ordered_splits = sorted(
        SPLIT_NAMES,
        key=lambda split_name: (-split_ratios[split_name], LIMITED_IMAGE_PRIORITY.index(split_name)),
    )
    return ordered_splits[:num_images]


def plan_class_image_allocation(label: str, num_images: int, split_ratios: dict[str, float]) -> dict[str, int]:
    if num_images <= 0:
        raise ValueError(f"num_images must be positive for label {label!r}, got: {num_images}")

    counts = {split_name: 0 for split_name in SPLIT_NAMES}
    if num_images == 1:
        selected_split = select_splits_for_limited_images(num_images, split_ratios)[0]
        counts[selected_split] = 1
        logging.warning("Label '%s' has only 1 image. Assigning it to %s.", label, selected_split)
        return counts

    if num_images == 2:
        selected_splits = select_splits_for_limited_images(num_images, split_ratios)
        for split_name in selected_splits:
            counts[split_name] += 1
        logging.warning("Label '%s' has only 2 images. Assigning them to %s.", label, selected_splits)
        return counts

    for split_name in COVERAGE_SPLITS:
        counts[split_name] = 1

    remaining_images = num_images - len(COVERAGE_SPLITS)
    base_allocations = {
        split_name: int(math.floor(remaining_images * split_ratios[split_name]))
        for split_name in SPLIT_NAMES
    }
    for split_name, value in base_allocations.items():
        counts[split_name] += value

    allocated_images = len(COVERAGE_SPLITS) + sum(base_allocations.values())
    remainder = num_images - allocated_images
    allocate_remainder_by_priority(counts, remainder, EXTRA_IMAGE_PRIORITY)
    return counts


def assign_images_to_splits(dataframe: pd.DataFrame, config: ImageSplitConfig) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    split_ratios = build_split_ratios(config)
    rng = random.Random(config.seed)
    split_assignments: dict[str, str] = {}
    per_class_split_counts: dict[str, dict[str, int]] = {}

    for label in sorted(dataframe["label"].unique().tolist()):
        label_rows = dataframe[dataframe["label"] == label].copy()
        records = label_rows.to_dict(orient="records")
        rng.shuffle(records)
        allocation = plan_class_image_allocation(label, len(records), split_ratios)
        per_class_split_counts[label] = allocation.copy()

        start = 0
        for split_name in SPLIT_NAMES:
            count = allocation[split_name]
            for record in records[start : start + count]:
                split_assignments[str(record["source_image"])] = split_name
            start += count

    assigned = dataframe.copy()
    assigned["split"] = assigned["source_image"].map(split_assignments)
    if assigned["split"].isna().any():
        missing_sources = assigned.loc[assigned["split"].isna(), "source_image"].tolist()
        raise RuntimeError(f"Failed to assign splits for source_image entries: {missing_sources[:5]}")

    return assigned, per_class_split_counts


def write_split_csvs(dataframe: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_paths: dict[str, Path] = {}
    for split_name in SPLIT_NAMES:
        split_frame = dataframe[dataframe["split"] == split_name].copy().reset_index(drop=True)
        split_path = output_dir / f"{split_name}_images.csv"
        split_frame.to_csv(split_path, index=False, encoding="utf-8", columns=EXPORT_COLUMNS)
        split_paths[split_name] = split_path
    return split_paths


def write_label_mapping(label_mapping: dict[str, int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label_to_index": label_mapping,
        "index_to_label": {str(index): label for label, index in label_mapping.items()},
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary(
    summary_path: Path,
    config: ImageSplitConfig,
    dataframe: pd.DataFrame,
    label_mapping: dict[str, int],
    per_class_split_counts: dict[str, dict[str, int]],
    split_paths: dict[str, Path],
) -> None:
    payload = {
        "raw_dir": str(config.raw_dir),
        "output_dir": str(config.output_dir),
        "label_mapping_path": str(config.label_mapping_path),
        "train_ratio": config.train_ratio,
        "val_ratio": config.val_ratio,
        "test_ratio": config.test_ratio,
        "seed": config.seed,
        "num_classes": len(label_mapping),
        "total_images": int(len(dataframe)),
        "labels": list(label_mapping.keys()),
        "split_counts": {key: int(value) for key, value in dataframe["split"].value_counts().to_dict().items()},
        "per_class_image_counts": {key: int(value) for key, value in dataframe["label"].value_counts().to_dict().items()},
        "per_class_split_counts": per_class_split_counts,
        "train_csv": str(split_paths["train"]),
        "val_csv": str(split_paths["val"]),
        "test_csv": str(split_paths["test"]),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_image_splits(config: ImageSplitConfig) -> dict[str, Path]:
    dataframe = build_image_dataframe(config.raw_dir)
    label_mapping = build_label_mapping(dataframe)
    dataframe["label_idx"] = dataframe["label"].map(label_mapping).astype(int)
    assigned, per_class_split_counts = assign_images_to_splits(dataframe, config)
    assigned = assigned[EXPORT_COLUMNS].reset_index(drop=True)

    split_paths = write_split_csvs(assigned, config.output_dir)
    write_label_mapping(label_mapping, config.label_mapping_path)
    write_summary(config.summary_path, config, assigned, label_mapping, per_class_split_counts, split_paths)
    return split_paths


def main() -> None:
    args = parse_args()
    setup_logging()
    config = build_config(args)
    split_paths = build_image_splits(config)
    logging.info("Built image-level splits from raw_dir=%s", config.raw_dir)
    logging.info("Train CSV: %s", split_paths["train"])
    logging.info("Val CSV: %s", split_paths["val"])
    logging.info("Test CSV: %s", split_paths["test"])
    logging.info("Label mapping: %s", config.label_mapping_path)
    logging.info("Summary: %s", config.summary_path)


if __name__ == "__main__":
    main()