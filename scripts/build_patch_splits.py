from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import load_yaml

REQUIRED_COLUMNS = ["patch_path", "label", "source_image"]
SPLIT_EXPORT_COLUMNS = ["patch_path", "label", "label_idx", "source_image", "patient_id", "patch_row", "patch_col", "split"]
IMAGE_SPLIT_EXPORT_COLUMNS = ["source_image", "label", "label_idx", "patient_id", "split"]
SPLIT_NAMES = ("train", "val", "test")
COVERAGE_SPLITS = ("train", "val", "test")
LIMITED_IMAGE_PRIORITY = ("train", "test", "val")
EXTRA_IMAGE_PRIORITY = ("test", "train", "val")
SPLIT_UNIT = "source_image"


@dataclass
class SplitBuildConfig:
    metadata_path: Path
    output_dir: Path
    all_patches_path: Path
    label_mapping_path: Path
    summary_path: Path | None
    train_ratio: float
    val_ratio: float
    test_ratio: float
    seed: int
    group_by: str = "auto"


@dataclass
class GroupRecord:
    group_id: str
    row_indices: list[int]
    label: str
    patch_count: int
    source_image: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build grouped train/val/test splits for patch metadata.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--metadata-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--all-patches-path", type=str, default=None)
    parser.add_argument("--label-mapping-path", type=str, default=None)
    parser.add_argument("--summary-path", type=str, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--group-by", choices=["auto", "patient_id", "source_image"], default=None)
    parser.add_argument("--run-self-check", action="store_true", help="Run internal allocation checks and exit.")
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


def build_config(args: argparse.Namespace) -> SplitBuildConfig:
    config_path = resolve_path(args.config, "configs/default.yaml")
    config = load_yaml(config_path) if config_path.exists() else {}

    data_cfg = config.get("data", {})
    split_cfg = config.get("build_patch_splits", {})

    metadata_path = resolve_path(
        args.metadata_path or split_cfg.get("metadata_path") or "data/metadata/patch_metadata.csv",
        "data/metadata/patch_metadata.csv",
    )
    output_dir = resolve_path(
        args.output_dir or split_cfg.get("output_dir") or data_cfg.get("split_dir") or "data/splits",
        "data/splits",
    )
    all_patches_path = resolve_path(
        args.all_patches_path or split_cfg.get("all_patches_path") or "data/splits/all_patches.csv",
        "data/splits/all_patches.csv",
    )
    label_mapping_path = resolve_path(
        args.label_mapping_path or split_cfg.get("label_mapping_path") or "data/splits/label_mapping.json",
        "data/splits/label_mapping.json",
    )

    summary_path_value = args.summary_path
    if summary_path_value is None:
        summary_path_value = split_cfg.get("summary_path")
    summary_path = resolve_path(summary_path_value, "data/splits/split_summary.json")

    train_ratio = args.train_ratio if args.train_ratio is not None else split_cfg.get("train_ratio", 0.7)
    val_ratio = args.val_ratio if args.val_ratio is not None else split_cfg.get("val_ratio", 0.15)
    test_ratio = args.test_ratio if args.test_ratio is not None else split_cfg.get("test_ratio", 0.15)
    seed = args.seed if args.seed is not None else split_cfg.get("seed", 42)
    group_by = args.group_by or split_cfg.get("group_by") or "auto"

    validate_ratios(train_ratio, val_ratio, test_ratio)

    return SplitBuildConfig(
        metadata_path=metadata_path,
        output_dir=output_dir,
        all_patches_path=all_patches_path,
        label_mapping_path=label_mapping_path,
        summary_path=summary_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        group_by=group_by,
    )


def load_patch_metadata(metadata_path: Path) -> pd.DataFrame:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Patch metadata file not found: {metadata_path}")

    dataframe = pd.read_csv(metadata_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Patch metadata is missing required columns: {missing_columns}")

    if "patient_id" not in dataframe.columns:
        dataframe["patient_id"] = ""

    dataframe["label"] = dataframe["label"].astype(str)
    dataframe["source_image"] = dataframe["source_image"].fillna("").astype(str).str.strip()
    dataframe["patch_path"] = dataframe["patch_path"].astype(str)
    dataframe["patient_id"] = dataframe["patient_id"].fillna("").astype(str).str.strip()

    if (dataframe["source_image"].str.len() == 0).any():
        raise ValueError("Patch metadata contains empty source_image values.")
    return dataframe


def build_label_mapping(dataframe: pd.DataFrame) -> dict[str, int]:
    labels = sorted(dataframe["label"].unique().tolist())
    return {label: index for index, label in enumerate(labels)}


def assign_group_keys(dataframe: pd.DataFrame, requested_group_by: str) -> tuple[pd.DataFrame, str]:
    dataframe = dataframe.copy()
    if requested_group_by != SPLIT_UNIT:
        logging.warning(
            "Requested group_by=%s, but split construction is forced to source_image. patient_id is kept only as metadata.",
            requested_group_by,
        )

    dataframe["group_id"] = dataframe["source_image"].map(lambda value: f"source::{value}")
    return dataframe, SPLIT_UNIT


def build_group_records(dataframe: pd.DataFrame) -> list[GroupRecord]:
    group_records: list[GroupRecord] = []
    for group_id, group_df in dataframe.groupby("group_id", sort=True):
        labels = sorted(group_df["label"].astype(str).unique().tolist())
        if len(labels) != 1:
            source_images = sorted(group_df["source_image"].astype(str).unique().tolist())
            raise ValueError(
                "Each source_image must contain exactly one class label. "
                f"group_id={group_id!r}, source_images={source_images}, labels={labels}"
            )

        source_images = sorted(group_df["source_image"].astype(str).unique().tolist())
        if len(source_images) != 1:
            raise RuntimeError(f"Internal grouping error: group_id {group_id!r} contains multiple source_image values {source_images}")

        group_records.append(
            GroupRecord(
                group_id=group_id,
                row_indices=group_df.index.to_list(),
                label=labels[0],
                patch_count=len(group_df),
                source_image=source_images[0],
            )
        )
    return group_records


def build_split_ratios(config: SplitBuildConfig) -> dict[str, float]:
    ratios = {
        "train": float(config.train_ratio),
        "val": float(config.val_ratio),
        "test": float(config.test_ratio),
    }
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("Split ratio sum must be positive.")
    return {split_name: ratio / total for split_name, ratio in ratios.items()}


def build_label_to_groups(group_records: list[GroupRecord]) -> dict[str, list[GroupRecord]]:
    groups_by_label: dict[str, list[GroupRecord]] = defaultdict(list)
    for group_record in group_records:
        groups_by_label[group_record.label].append(group_record)
    return {label: groups_by_label[label] for label in sorted(groups_by_label)}


def allocate_remainder_by_priority(
    counts: dict[str, int],
    remainder: int,
    priority: tuple[str, ...],
) -> None:
    priority_index = 0
    while remainder > 0:
        split_name = priority[priority_index % len(priority)]
        counts[split_name] += 1
        remainder -= 1
        priority_index += 1


def select_splits_for_limited_images(num_groups: int, split_ratios: dict[str, float]) -> list[str]:
    ordered_splits = sorted(
        SPLIT_NAMES,
        key=lambda split_name: (-split_ratios[split_name], LIMITED_IMAGE_PRIORITY.index(split_name)),
    )
    return ordered_splits[:num_groups]


def plan_class_image_allocation(
    label: str,
    num_groups: int,
    split_ratios: dict[str, float],
) -> dict[str, int]:
    if num_groups <= 0:
        raise ValueError(f"num_groups must be positive for label {label!r}, got: {num_groups}")

    counts = {split_name: 0 for split_name in SPLIT_NAMES}

    if num_groups == 1:
        selected_split = select_splits_for_limited_images(num_groups, split_ratios)[0]
        counts[selected_split] = 1
        logging.warning(
            "Label '%s' has only 1 source_image. Cannot cover train/val/test simultaneously; assigning to %s only.",
            label,
            selected_split,
        )
        return counts

    if num_groups == 2:
        selected_splits = select_splits_for_limited_images(num_groups, split_ratios)
        for split_name in selected_splits:
            counts[split_name] += 1
        logging.warning(
            "Label '%s' has only 2 source_image entries. Cannot cover all three splits; assigning to %s.",
            label,
            selected_splits,
        )
        return counts

    for split_name in COVERAGE_SPLITS:
        counts[split_name] = 1

    remaining_groups = num_groups - len(COVERAGE_SPLITS)
    base_allocations = {
        split_name: int(math.floor(remaining_groups * split_ratios[split_name]))
        for split_name in SPLIT_NAMES
    }
    for split_name, value in base_allocations.items():
        counts[split_name] += value

    allocated_groups = len(COVERAGE_SPLITS) + sum(base_allocations.values())
    remainder = num_groups - allocated_groups
    allocate_remainder_by_priority(counts, remainder, EXTRA_IMAGE_PRIORITY)

    if sum(counts.values()) != num_groups:
        raise RuntimeError(
            f"Internal allocation error for label {label!r}: expected {num_groups} source_image entries, got {counts}"
        )
    return counts


def assign_groups_to_splits(
    group_records: list[GroupRecord],
    config: SplitBuildConfig,
) -> tuple[dict[str, str], dict[str, dict[str, int]], dict[str, int]]:
    split_ratios = build_split_ratios(config)
    groups_by_label = build_label_to_groups(group_records)
    rng = random.Random(config.seed)

    group_to_split: dict[str, str] = {}
    per_class_image_distribution: dict[str, dict[str, int]] = {}
    per_class_image_counts: dict[str, int] = {}

    for label in sorted(groups_by_label):
        label_groups = list(groups_by_label[label])
        rng.shuffle(label_groups)

        per_class_image_counts[label] = len(label_groups)
        split_counts = plan_class_image_allocation(
            label=label,
            num_groups=len(label_groups),
            split_ratios=split_ratios,
        )
        per_class_image_distribution[label] = dict(split_counts)

        start_index = 0
        for split_name in SPLIT_NAMES:
            count = split_counts[split_name]
            selected_groups = label_groups[start_index : start_index + count]
            for group_record in selected_groups:
                if group_record.group_id in group_to_split:
                    raise RuntimeError(f"Group {group_record.group_id!r} was assigned more than once.")
                group_to_split[group_record.group_id] = split_name
            start_index += count

        if start_index != len(label_groups):
            raise RuntimeError(
                f"Label '{label}' allocation did not consume all source_image groups: consumed={start_index}, total={len(label_groups)}"
            )

    return group_to_split, per_class_image_distribution, per_class_image_counts


def validate_patch_split_consistency(dataframe: pd.DataFrame) -> None:
    for group_column in ["group_id", "source_image"]:
        split_counts = dataframe.groupby(group_column)["split"].nunique()
        inconsistent = split_counts[split_counts > 1]
        if not inconsistent.empty:
            preview = inconsistent.index.tolist()[:5]
            raise ValueError(
                f"Found {len(inconsistent)} {group_column} entries assigned to multiple splits. Examples: {preview}"
            )


def save_dataframe(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def format_distribution(counter: Counter) -> dict[str, int]:
    return {label: int(count) for label, count in sorted(counter.items())}


def compute_per_class_image_distribution(dataframe: pd.DataFrame) -> dict[str, dict[str, int]]:
    distribution: dict[str, dict[str, int]] = {}
    for label in sorted(dataframe["label"].astype(str).unique().tolist()):
        label_df = dataframe[dataframe["label"].astype(str) == label]
        distribution[label] = {
            split_name: int(label_df[label_df["split"] == split_name]["source_image"].nunique())
            for split_name in SPLIT_NAMES
        }
    return distribution


def summarize_splits(
    dataframe: pd.DataFrame,
    requested_group_by: str,
    effective_group_by: str,
    seed: int,
) -> dict[str, Any]:
    split_summaries: dict[str, dict[str, Any]] = {}
    for split_name in SPLIT_NAMES:
        split_df = dataframe[dataframe["split"] == split_name]
        if split_df.empty:
            split_summaries[split_name] = {
                "num_patches": 0,
                "num_groups": 0,
                "label_distribution": {},
            }
            continue

        split_summaries[split_name] = {
            "num_patches": int(len(split_df)),
            "num_groups": int(split_df["source_image"].nunique()),
            "label_distribution": format_distribution(Counter(split_df["label"].tolist())),
        }

    return {
        "seed": seed,
        "split_unit": SPLIT_UNIT,
        "requested_group_by": requested_group_by,
        "effective_group_by": effective_group_by,
        "total_patches": int(len(dataframe)),
        "total_groups": int(dataframe["source_image"].nunique()),
        "label_distribution": format_distribution(Counter(dataframe["label"].tolist())),
        "per_class_image_distribution": compute_per_class_image_distribution(dataframe),
        "splits": split_summaries,
    }


def save_label_mapping(label_mapping: dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label_to_index": label_mapping,
        "index_to_label": {str(index): label for label, index in label_mapping.items()},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def log_per_class_image_summary(
    per_class_image_counts: dict[str, int],
    per_class_image_distribution: dict[str, dict[str, int]],
) -> None:
    logging.info("Per-class source_image counts:")
    for label in sorted(per_class_image_counts):
        logging.info("  %s | source_images=%s", label, per_class_image_counts[label])

    logging.info("Per-class split allocation (source_image counts):")
    for label in sorted(per_class_image_distribution):
        distribution = per_class_image_distribution[label]
        logging.info(
            "  %s | train=%s val=%s test=%s",
            label,
            distribution.get("train", 0),
            distribution.get("val", 0),
            distribution.get("test", 0),
        )


def log_split_summary(summary: dict[str, Any]) -> None:
    logging.info("Split unit: %s", summary["split_unit"])
    logging.info("Requested group_by: %s", summary["requested_group_by"])
    logging.info("Effective group_by: %s", summary["effective_group_by"])
    logging.info("Total patches: %s", summary["total_patches"])
    logging.info("Total source_image groups: %s", summary["total_groups"])
    logging.info("Overall label distribution: %s", summary["label_distribution"])

    for split_name in SPLIT_NAMES:
        split_summary = summary["splits"][split_name]
        logging.info(
            "%s | patches=%s source_images=%s label_distribution=%s",
            split_name,
            split_summary["num_patches"],
            split_summary["num_groups"],
            split_summary["label_distribution"],
        )


def export_split_files(dataframe: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    export_columns = [column for column in SPLIT_EXPORT_COLUMNS if column in dataframe.columns]
    for split_name in SPLIT_NAMES:
        split_df = dataframe[dataframe["split"] == split_name].copy()
        save_dataframe(split_df[export_columns], output_dir / f"{split_name}.csv")


def build_image_level_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    image_records: list[dict[str, Any]] = []
    for source_image, group_df in dataframe.groupby("source_image", sort=True):
        labels = sorted(group_df["label"].astype(str).unique().tolist())
        if len(labels) != 1:
            raise ValueError(
                "Each source_image must contain exactly one class label before exporting image-level splits. "
                f"source_image={source_image!r}, labels={labels}"
            )

        split_values = sorted(group_df["split"].astype(str).unique().tolist())
        if len(split_values) != 1:
            raise ValueError(
                "Each source_image must belong to exactly one split before exporting image-level splits. "
                f"source_image={source_image!r}, splits={split_values}"
            )

        record: dict[str, Any] = {
            "source_image": str(source_image),
            "label": labels[0],
            "split": split_values[0],
        }
        if "label_idx" in group_df.columns:
            label_indices = sorted(group_df["label_idx"].astype(int).unique().tolist())
            if len(label_indices) != 1:
                raise ValueError(
                    "Each source_image must map to exactly one label_idx before exporting image-level splits. "
                    f"source_image={source_image!r}, label_idx={label_indices}"
                )
            record["label_idx"] = int(label_indices[0])
        if "patient_id" in group_df.columns:
            patient_values = [value for value in group_df["patient_id"].fillna("").astype(str).str.strip().tolist() if value]
            record["patient_id"] = patient_values[0] if patient_values else ""
        image_records.append(record)

    image_dataframe = pd.DataFrame(image_records)
    ordered_columns = [column for column in IMAGE_SPLIT_EXPORT_COLUMNS if column in image_dataframe.columns]
    extra_columns = [column for column in image_dataframe.columns if column not in ordered_columns]
    return image_dataframe[ordered_columns + extra_columns].sort_values(by=["split", "label", "source_image"]).reset_index(drop=True)


def export_image_split_files(dataframe: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dataframe = build_image_level_dataframe(dataframe)
    export_columns = [column for column in IMAGE_SPLIT_EXPORT_COLUMNS if column in image_dataframe.columns]
    for split_name in SPLIT_NAMES:
        split_df = image_dataframe[image_dataframe["split"] == split_name].copy()
        save_dataframe(split_df[export_columns], output_dir / f"{split_name}_images.csv")


def run_split_builder(config: SplitBuildConfig) -> dict[str, Any]:
    dataframe = load_patch_metadata(config.metadata_path)
    dataframe, effective_group_by = assign_group_keys(dataframe, config.group_by)

    label_mapping = build_label_mapping(dataframe)
    dataframe["label_idx"] = dataframe["label"].map(label_mapping)

    group_records = build_group_records(dataframe)
    group_to_split, per_class_image_distribution, per_class_image_counts = assign_groups_to_splits(
        group_records=group_records,
        config=config,
    )
    dataframe["split"] = dataframe["group_id"].map(group_to_split)

    if dataframe["split"].isna().any():
        missing_groups = sorted(dataframe.loc[dataframe["split"].isna(), "group_id"].astype(str).unique().tolist())
        raise RuntimeError(f"Some source_image groups were not assigned to any split: {missing_groups[:10]}")

    validate_patch_split_consistency(dataframe)

    ordered_columns = [
        column
        for column in ["patch_path", "label", "label_idx", "source_image", "patient_id", "patch_row", "patch_col", "group_id", "split"]
        if column in dataframe.columns
    ]
    extra_columns = [column for column in dataframe.columns if column not in ordered_columns]
    dataframe = dataframe[ordered_columns + extra_columns].sort_values(
        by=["split", "label", "source_image", "patch_path"]
    )

    save_dataframe(dataframe, config.all_patches_path)
    export_split_files(dataframe, config.output_dir)
    export_image_split_files(dataframe, config.output_dir)
    save_label_mapping(label_mapping, config.label_mapping_path)

    summary = summarize_splits(
        dataframe=dataframe,
        requested_group_by=config.group_by,
        effective_group_by=effective_group_by,
        seed=config.seed,
    )
    if config.summary_path is not None:
        config.summary_path.parent.mkdir(parents=True, exist_ok=True)
        config.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    log_per_class_image_summary(per_class_image_counts, per_class_image_distribution)
    log_split_summary(summary)
    logging.info("Saved all patches file to %s", config.all_patches_path)
    logging.info("Saved patch-level split files to %s", config.output_dir)
    logging.info("Saved image-level split files to %s", config.output_dir)
    logging.info("Saved label mapping to %s", config.label_mapping_path)
    return summary


def run_allocation_self_checks() -> None:
    split_ratios = {"train": 0.7, "val": 0.15, "test": 0.15}

    counts_three = plan_class_image_allocation("three_case", 3, split_ratios)
    assert counts_three == {"train": 1, "val": 1, "test": 1}, counts_three

    counts_four = plan_class_image_allocation("four_case", 4, split_ratios)
    assert counts_four == {"train": 1, "val": 1, "test": 2}, counts_four

    counts_two = plan_class_image_allocation("two_case", 2, split_ratios)
    assert counts_two == {"train": 1, "val": 0, "test": 1}, counts_two

    patient_mixed_df = pd.DataFrame(
        [
            {"patch_path": "a1.png", "label": "A", "source_image": "img_a", "patient_id": "p1"},
            {"patch_path": "a2.png", "label": "A", "source_image": "img_a", "patient_id": "p1"},
            {"patch_path": "b1.png", "label": "B", "source_image": "img_b", "patient_id": "p1"},
            {"patch_path": "b2.png", "label": "B", "source_image": "img_b", "patient_id": "p1"},
        ]
    )
    grouped_df, effective_group_by = assign_group_keys(patient_mixed_df, "patient_id")
    assert effective_group_by == "source_image", effective_group_by
    build_group_records(grouped_df)

    inheritance_df = pd.DataFrame(
        [
            {"patch_path": "a_patch_1.png", "label": "A", "source_image": "img_a", "group_id": "source::img_a", "split": "train"},
            {"patch_path": "a_patch_2.png", "label": "A", "source_image": "img_a", "group_id": "source::img_a", "split": "train"},
            {"patch_path": "b_patch_1.png", "label": "B", "source_image": "img_b", "group_id": "source::img_b", "split": "test"},
            {"patch_path": "b_patch_2.png", "label": "B", "source_image": "img_b", "group_id": "source::img_b", "split": "test"},
        ]
    )
    validate_patch_split_consistency(inheritance_df)

    logging.info("All internal split-allocation self-checks passed.")


def main() -> None:
    setup_logging()
    args = parse_args()

    if args.run_self_check:
        run_allocation_self_checks()
        return

    config = build_config(args)
    run_split_builder(config)


if __name__ == "__main__":
    main()
