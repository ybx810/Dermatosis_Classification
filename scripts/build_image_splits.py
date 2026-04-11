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
from typing import Any

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
SUPPORTED_MODES = {"single", "kfold"}
EXPORT_COLUMNS = ["source_image", "label", "label_idx", "patient_id", "split"]


@dataclass(frozen=True)
class ImageSplitConfig:
    raw_dir: Path
    output_dir: Path
    folds_dir: Path
    label_mapping_path: Path
    summary_path: Path
    train_ratio: float
    val_ratio: float
    test_ratio: float
    seed: int
    mode: str
    n_splits: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build image-level splits from raw whole images.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--mode", type=str, choices=sorted(SUPPORTED_MODES), default=None)
    parser.add_argument("--raw-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--folds-dir", type=str, default=None)
    parser.add_argument("--label-mapping-path", type=str, default=None)
    parser.add_argument("--summary-path", type=str, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument("--n-splits", type=int, default=None)
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

    mode = str(args.mode or split_cfg.get("mode", "single")).strip().lower()
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported build_image_splits.mode={mode!r}. Expected one of {sorted(SUPPORTED_MODES)}.")

    raw_dir = resolve_path(args.raw_dir or data_cfg.get("raw_dir"), "data/raw")
    output_dir = resolve_path(args.output_dir or split_cfg.get("output_dir") or data_cfg.get("split_dir"), "data/splits")
    folds_dir = resolve_path(args.folds_dir or split_cfg.get("folds_dir"), "data/splits/cv3")
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
    n_splits = int(args.n_splits if args.n_splits is not None else split_cfg.get("n_splits", 3))
    seed = int(args.seed if args.seed is not None else split_cfg.get("seed", 42))

    validate_ratios(train_ratio, val_ratio, test_ratio)
    if mode == "kfold" and n_splits < 2:
        raise ValueError(f"build_image_splits.n_splits must be >= 2 for kfold mode, got: {n_splits}")

    return ImageSplitConfig(
        raw_dir=raw_dir,
        output_dir=output_dir,
        folds_dir=folds_dir,
        label_mapping_path=label_mapping_path,
        summary_path=summary_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        mode=mode,
        n_splits=n_splits,
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


def assign_images_to_kfolds(dataframe: pd.DataFrame, config: ImageSplitConfig) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    rng = random.Random(config.seed)
    fold_assignments: dict[str, int] = {}
    per_class_per_fold_counts: dict[str, dict[str, int]] = {}

    for label in sorted(dataframe["label"].unique().tolist()):
        label_rows = dataframe[dataframe["label"] == label].copy()
        records = label_rows.to_dict(orient="records")
        if len(records) < config.n_splits:
            raise ValueError(
                f"Label '{label}' has {len(records)} source_image entries, which is less than n_splits={config.n_splits}. "
                "This class cannot be stratified across all folds."
            )

        rng.shuffle(records)
        per_fold_counts = {f"fold_{fold_idx}": 0 for fold_idx in range(config.n_splits)}
        for record_index, record in enumerate(records):
            fold_idx = record_index % config.n_splits
            source_image = str(record["source_image"])
            fold_assignments[source_image] = fold_idx
            per_fold_counts[f"fold_{fold_idx}"] += 1
        per_class_per_fold_counts[label] = per_fold_counts

    assigned = dataframe.copy()
    assigned["fold"] = assigned["source_image"].map(fold_assignments)
    if assigned["fold"].isna().any():
        missing_sources = assigned.loc[assigned["fold"].isna(), "source_image"].tolist()
        raise RuntimeError(f"Failed to assign folds for source_image entries: {missing_sources[:5]}")

    assigned["fold"] = assigned["fold"].astype(int)
    return assigned, per_class_per_fold_counts


def write_split_csvs(dataframe: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_paths: dict[str, Path] = {}
    for split_name in SPLIT_NAMES:
        split_frame = dataframe[dataframe["split"] == split_name].copy().reset_index(drop=True)
        split_path = output_dir / f"{split_name}_images.csv"
        split_frame.to_csv(split_path, index=False, encoding="utf-8", columns=EXPORT_COLUMNS)
        split_paths[split_name] = split_path
    return split_paths


def write_kfold_csvs(
    dataframe: pd.DataFrame,
    folds_dir: Path,
    n_splits: int,
) -> dict[str, dict[str, Any]]:
    folds_dir.mkdir(parents=True, exist_ok=True)
    fold_csv_paths: dict[str, dict[str, Any]] = {}
    for fold_idx in range(n_splits):
        fold_key = f"fold_{fold_idx}"
        val_frame = dataframe[dataframe["fold"] == fold_idx].copy()
        train_frame = dataframe[dataframe["fold"] != fold_idx].copy()

        train_frame["split"] = "train"
        val_frame["split"] = "val"
        train_frame = train_frame.sort_values(by=["label", "source_image"]).reset_index(drop=True)
        val_frame = val_frame.sort_values(by=["label", "source_image"]).reset_index(drop=True)

        train_path = folds_dir / f"{fold_key}_train_images.csv"
        val_path = folds_dir / f"{fold_key}_val_images.csv"
        train_frame.to_csv(train_path, index=False, encoding="utf-8", columns=EXPORT_COLUMNS)
        val_frame.to_csv(val_path, index=False, encoding="utf-8", columns=EXPORT_COLUMNS)
        fold_csv_paths[fold_key] = {
            "train_csv": train_path,
            "val_csv": val_path,
            "train_count": int(len(train_frame)),
            "val_count": int(len(val_frame)),
        }
    return fold_csv_paths


def write_label_mapping(label_mapping: dict[str, int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label_to_index": label_mapping,
        "index_to_label": {str(index): label for label, index in label_mapping.items()},
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_single_summary(
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
        "mode": "single",
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


def write_kfold_summary(
    summary_path: Path,
    config: ImageSplitConfig,
    dataframe: pd.DataFrame,
    label_mapping: dict[str, int],
    per_class_per_fold_counts: dict[str, dict[str, int]],
    fold_csv_paths: dict[str, dict[str, Any]],
    compatibility_split_paths: dict[str, Path],
) -> Path:
    cv_summary_path = config.folds_dir / "cv_split_summary.json"
    payload = {
        "raw_dir": str(config.raw_dir),
        "output_dir": str(config.output_dir),
        "folds_dir": str(config.folds_dir),
        "mode": "kfold",
        "n_splits": int(config.n_splits),
        "seed": int(config.seed),
        "label_mapping_path": str(config.label_mapping_path),
        "total_images": int(len(dataframe)),
        "num_classes": len(label_mapping),
        "labels": list(label_mapping.keys()),
        "per_class_image_counts": {key: int(value) for key, value in dataframe["label"].value_counts().to_dict().items()},
        "per_fold_total_counts": {
            f"fold_{fold_idx}": int((dataframe["fold"] == fold_idx).sum()) for fold_idx in range(config.n_splits)
        },
        "per_class_per_fold_counts": per_class_per_fold_counts,
        "folds": {
            fold_key: {
                "train_csv": str(payload_item["train_csv"]),
                "val_csv": str(payload_item["val_csv"]),
                "train_count": int(payload_item["train_count"]),
                "val_count": int(payload_item["val_count"]),
            }
            for fold_key, payload_item in fold_csv_paths.items()
        },
        "compat_single_split_csvs": {split_name: str(path) for split_name, path in compatibility_split_paths.items()},
    }

    targets = [cv_summary_path]
    if summary_path.resolve() != cv_summary_path.resolve():
        targets.append(summary_path)

    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return cv_summary_path


def build_image_splits(config: ImageSplitConfig) -> dict[str, Any]:
    dataframe = build_image_dataframe(config.raw_dir)
    label_mapping = build_label_mapping(dataframe)
    dataframe["label_idx"] = dataframe["label"].map(label_mapping).astype(int)
    write_label_mapping(label_mapping, config.label_mapping_path)

    if config.mode == "single":
        assigned, per_class_split_counts = assign_images_to_splits(dataframe, config)
        assigned = assigned[EXPORT_COLUMNS].reset_index(drop=True)

        split_paths = write_split_csvs(assigned, config.output_dir)
        write_single_summary(config.summary_path, config, assigned, label_mapping, per_class_split_counts, split_paths)
        return {
            "mode": "single",
            "split_paths": split_paths,
            "summary_path": config.summary_path,
        }

    assigned, per_class_per_fold_counts = assign_images_to_kfolds(dataframe, config)
    fold_csv_paths = write_kfold_csvs(assigned, config.folds_dir, config.n_splits)

    compatibility_assigned, _ = assign_images_to_splits(dataframe, config)
    compatibility_assigned = compatibility_assigned[EXPORT_COLUMNS].reset_index(drop=True)
    compatibility_split_paths = write_split_csvs(compatibility_assigned, config.output_dir)

    cv_summary_path = write_kfold_summary(
        summary_path=config.summary_path,
        config=config,
        dataframe=assigned,
        label_mapping=label_mapping,
        per_class_per_fold_counts=per_class_per_fold_counts,
        fold_csv_paths=fold_csv_paths,
        compatibility_split_paths=compatibility_split_paths,
    )
    return {
        "mode": "kfold",
        "fold_csv_paths": fold_csv_paths,
        "split_paths": compatibility_split_paths,
        "summary_path": config.summary_path,
        "cv_summary_path": cv_summary_path,
    }


def main() -> None:
    args = parse_args()
    setup_logging()
    config = build_config(args)
    result = build_image_splits(config)

    logging.info("Built image-level splits from raw_dir=%s | mode=%s", config.raw_dir, result["mode"])
    logging.info("Label mapping: %s", config.label_mapping_path)
    if result["mode"] == "single":
        split_paths = result["split_paths"]
        logging.info("Train CSV: %s", split_paths["train"])
        logging.info("Val CSV: %s", split_paths["val"])
        logging.info("Test CSV: %s", split_paths["test"])
        logging.info("Summary: %s", result["summary_path"])
    else:
        logging.info("Folds directory: %s", config.folds_dir)
        for fold_key, fold_payload in result["fold_csv_paths"].items():
            logging.info(
                "%s | train_csv=%s val_csv=%s train_count=%d val_count=%d",
                fold_key,
                fold_payload["train_csv"],
                fold_payload["val_csv"],
                fold_payload["train_count"],
                fold_payload["val_count"],
            )
        split_paths = result["split_paths"]
        logging.info(
            "Compatibility single split CSVs | train=%s val=%s test=%s",
            split_paths["train"],
            split_paths["val"],
            split_paths["test"],
        )
        logging.info("CV Summary: %s", result["cv_summary_path"])
        if Path(result["summary_path"]).resolve() != Path(result["cv_summary_path"]).resolve():
            logging.info("Summary alias: %s", result["summary_path"])


if __name__ == "__main__":
    main()

