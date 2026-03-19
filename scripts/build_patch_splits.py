from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import load_yaml

REQUIRED_COLUMNS = ["patch_path", "label", "source_image"]
SPLIT_EXPORT_COLUMNS = ["patch_path", "label", "label_idx", "source_image", "patient_id", "patch_row", "patch_col", "split"]
SPLIT_NAMES = ("train", "val", "test")


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
    label_counts: Counter
    patch_count: int


@dataclass
class SplitState:
    name: str
    target_ratio: float
    patch_count: int = 0
    group_ids: list[str] = field(default_factory=list)
    label_counts: Counter = field(default_factory=Counter)


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
    dataframe["source_image"] = dataframe["source_image"].astype(str)
    dataframe["patch_path"] = dataframe["patch_path"].astype(str)
    dataframe["patient_id"] = dataframe["patient_id"].fillna("").astype(str).str.strip()
    return dataframe


def build_label_mapping(dataframe: pd.DataFrame) -> dict[str, int]:
    labels = sorted(dataframe["label"].unique().tolist())
    return {label: index for index, label in enumerate(labels)}


def assign_group_keys(dataframe: pd.DataFrame, group_by: str) -> tuple[pd.DataFrame, str]:
    dataframe = dataframe.copy()
    patient_available = dataframe["patient_id"].str.len() > 0

    if group_by == "source_image":
        dataframe["group_id"] = dataframe["source_image"].map(lambda value: f"source::{value}")
        return dataframe, "source_image"

    if group_by == "patient_id":
        if not patient_available.any():
            logging.warning("No patient_id values found. Falling back to source_image grouping.")
            dataframe["group_id"] = dataframe["source_image"].map(lambda value: f"source::{value}")
            return dataframe, "source_image"

        dataframe["group_id"] = dataframe.apply(
            lambda row: f"patient::{row['patient_id']}" if row["patient_id"] else f"source::{row['source_image']}",
            axis=1,
        )
        if (~patient_available).any():
            logging.warning("Some rows have empty patient_id. Those rows are grouped by source_image.")
        return dataframe, "patient_id"

    if patient_available.any():
        dataframe["group_id"] = dataframe.apply(
            lambda row: f"patient::{row['patient_id']}" if row["patient_id"] else f"source::{row['source_image']}",
            axis=1,
        )
        if (~patient_available).any():
            logging.warning("patient_id is partially missing. Falling back to source_image for rows without patient_id.")
        return dataframe, "patient_id(auto)"

    dataframe["group_id"] = dataframe["source_image"].map(lambda value: f"source::{value}")
    return dataframe, "source_image(auto)"


def build_group_records(dataframe: pd.DataFrame) -> list[GroupRecord]:
    group_records: list[GroupRecord] = []
    for group_id, group_df in dataframe.groupby("group_id", sort=True):
        group_records.append(
            GroupRecord(
                group_id=group_id,
                row_indices=group_df.index.to_list(),
                label_counts=Counter(group_df["label"].tolist()),
                patch_count=len(group_df),
            )
        )
    return group_records


def active_splits(config: SplitBuildConfig) -> list[SplitState]:
    ratios = {
        "train": config.train_ratio,
        "val": config.val_ratio,
        "test": config.test_ratio,
    }
    ratio_sum = sum(ratios.values())
    return [
        SplitState(name=name, target_ratio=ratio / ratio_sum)
        for name, ratio in ratios.items()
        if ratio > 0
    ]


def clone_split_states(states: dict[str, SplitState]) -> dict[str, SplitState]:
    cloned: dict[str, SplitState] = {}
    for name, state in states.items():
        cloned[name] = SplitState(
            name=state.name,
            target_ratio=state.target_ratio,
            patch_count=state.patch_count,
            group_ids=state.group_ids.copy(),
            label_counts=Counter(state.label_counts),
        )
    return cloned


def compute_assignment_score(
    states: dict[str, SplitState],
    label_totals: Counter,
    total_patch_count: int,
) -> float:
    total_score = 0.0
    for state in states.values():
        target_size = state.target_ratio * total_patch_count
        size_penalty = abs(state.patch_count - target_size) / max(1, total_patch_count)
        overflow_penalty = max(0.0, state.patch_count - target_size) / max(1, total_patch_count)
        empty_penalty = 0.25 if not state.group_ids else 0.0

        label_penalty = 0.0
        for label, total_for_label in label_totals.items():
            target_label_count = state.target_ratio * total_for_label
            label_penalty += abs(state.label_counts[label] - target_label_count) / max(1, total_for_label)

        total_score += label_penalty + size_penalty + (2.0 * overflow_penalty) + empty_penalty

    return total_score


def choose_best_split(
    group_record: GroupRecord,
    states: dict[str, SplitState],
    split_order: list[str],
    label_totals: Counter,
    total_patch_count: int,
) -> str:
    best_split = split_order[0]
    best_score: tuple[float, str] | None = None

    for split_name in split_order:
        projected_states = clone_split_states(states)
        projected_state = projected_states[split_name]
        projected_state.group_ids.append(group_record.group_id)
        projected_state.patch_count += group_record.patch_count
        projected_state.label_counts.update(group_record.label_counts)

        score = compute_assignment_score(projected_states, label_totals, total_patch_count)
        score_tuple = (score, split_name)
        if best_score is None or score_tuple < best_score:
            best_score = score_tuple
            best_split = split_name

    return best_split


def assign_groups_to_splits(
    group_records: list[GroupRecord],
    config: SplitBuildConfig,
    label_totals: Counter,
    total_patch_count: int,
) -> dict[str, str]:
    split_state_template = active_splits(config)
    split_order = [state.name for state in split_state_template]
    target_sorted_splits = [state.name for state in sorted(split_state_template, key=lambda item: (-item.target_ratio, item.name))]

    trial_count = min(256, max(32, len(group_records) * 8))
    best_group_to_split: dict[str, str] | None = None
    best_score: float | None = None

    for trial_index in range(trial_count):
        trial_rng = random.Random(config.seed + trial_index)
        shuffled_groups = group_records[:]
        trial_rng.shuffle(shuffled_groups)
        ordered_groups = sorted(
            shuffled_groups,
            key=lambda record: (
                -max(record.label_counts.values()),
                -record.patch_count,
                record.group_id,
            ),
        )

        split_states = {
            state.name: SplitState(name=state.name, target_ratio=state.target_ratio)
            for state in split_state_template
        }
        group_to_split: dict[str, str] = {}
        remaining_groups = ordered_groups[:]

        for split_name in target_sorted_splits:
            if not remaining_groups:
                break
            group_record = remaining_groups.pop(0)
            state = split_states[split_name]
            state.group_ids.append(group_record.group_id)
            state.patch_count += group_record.patch_count
            state.label_counts.update(group_record.label_counts)
            group_to_split[group_record.group_id] = split_name

        for group_record in remaining_groups:
            split_name = choose_best_split(
                group_record=group_record,
                states=split_states,
                split_order=split_order,
                label_totals=label_totals,
                total_patch_count=total_patch_count,
            )
            state = split_states[split_name]
            state.group_ids.append(group_record.group_id)
            state.patch_count += group_record.patch_count
            state.label_counts.update(group_record.label_counts)
            group_to_split[group_record.group_id] = split_name

        score = compute_assignment_score(split_states, label_totals, total_patch_count)
        if best_score is None or score < best_score:
            best_score = score
            best_group_to_split = group_to_split

    if best_group_to_split is None:
        return {}
    return best_group_to_split


def save_dataframe(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def format_distribution(counter: Counter) -> dict[str, int]:
    return {label: int(count) for label, count in sorted(counter.items())}


def summarize_splits(dataframe: pd.DataFrame, grouping_strategy: str, seed: int) -> dict:
    split_summaries: dict[str, dict] = {}
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
            "num_groups": int(split_df["group_id"].nunique()),
            "label_distribution": format_distribution(Counter(split_df["label"].tolist())),
        }

    return {
        "seed": seed,
        "grouping_strategy": grouping_strategy,
        "total_patches": int(len(dataframe)),
        "total_groups": int(dataframe["group_id"].nunique()),
        "label_distribution": format_distribution(Counter(dataframe["label"].tolist())),
        "splits": split_summaries,
    }


def save_label_mapping(label_mapping: dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label_to_index": label_mapping,
        "index_to_label": {str(index): label for label, index in label_mapping.items()},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def log_split_summary(summary: dict) -> None:
    logging.info("Grouping strategy: %s", summary["grouping_strategy"])
    logging.info("Total patches: %s", summary["total_patches"])
    logging.info("Total groups: %s", summary["total_groups"])
    logging.info("Overall label distribution: %s", summary["label_distribution"])

    for split_name in SPLIT_NAMES:
        split_summary = summary["splits"][split_name]
        logging.info(
            "%s | patches=%s groups=%s label_distribution=%s",
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


def run_split_builder(config: SplitBuildConfig) -> dict:
    dataframe = load_patch_metadata(config.metadata_path)
    dataframe, grouping_strategy = assign_group_keys(dataframe, config.group_by)

    label_mapping = build_label_mapping(dataframe)
    dataframe["label_idx"] = dataframe["label"].map(label_mapping)

    group_records = build_group_records(dataframe)
    label_totals = Counter(dataframe["label"].tolist())
    group_to_split = assign_groups_to_splits(
        group_records=group_records,
        config=config,
        label_totals=label_totals,
        total_patch_count=len(dataframe),
    )
    dataframe["split"] = dataframe["group_id"].map(group_to_split)

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
    save_label_mapping(label_mapping, config.label_mapping_path)

    summary = summarize_splits(dataframe, grouping_strategy, config.seed)
    if config.summary_path is not None:
        config.summary_path.parent.mkdir(parents=True, exist_ok=True)
        config.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    log_split_summary(summary)
    logging.info("Saved all patches file to %s", config.all_patches_path)
    logging.info("Saved label mapping to %s", config.label_mapping_path)
    return summary


def main() -> None:
    setup_logging()
    args = parse_args()
    config = build_config(args)
    run_split_builder(config)


if __name__ == "__main__":
    main()


