from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.test import run_test_from_checkpoint
from src.main import run_training
from src.utils.io import load_yaml
from src.utils.label_merge import (
    apply_label_merge_to_dataframe,
    build_label_merge_mapping,
    get_label_names_from_mapping,
    is_label_merge_enabled,
    save_label_merge_mapping,
    update_config_num_classes_from_mapping,
    validate_label_merge_coverage,
)

REQUIRED_SPLIT_COLUMNS = {"patch_path", "label", "source_image"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cross-validation from existing train/val splits and a fixed test split."
    )
    parser.add_argument("--config", type=str, default="configs/cv_label_merge.yaml")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def resolve_path(path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_split_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required split CSV not found: {path}")
    dataframe = pd.read_csv(path)
    missing_columns = REQUIRED_SPLIT_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Split CSV {path} is missing required columns: {sorted(missing_columns)}")
    dataframe["patch_path"] = dataframe["patch_path"].astype(str)
    dataframe["label"] = dataframe["label"].astype(str)
    dataframe["source_image"] = dataframe["source_image"].fillna("").astype(str).str.strip()
    if dataframe["source_image"].eq("").any():
        raise ValueError(f"Split CSV {path} contains empty source_image values.")
    return dataframe


def resolve_patch_path(csv_path: Path, patch_path: str) -> Path:
    candidate = Path(patch_path)
    if candidate.is_absolute():
        return candidate
    csv_relative = csv_path.parent / candidate
    if csv_relative.exists():
        return csv_relative.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def check_patch_files(
    csv_path: Path,
    dataframe: pd.DataFrame,
    full_check: bool,
    sample_size: int,
) -> None:
    if full_check or len(dataframe) <= sample_size:
        check_df = dataframe
    else:
        check_df = dataframe.sample(n=sample_size, random_state=0)

    missing: list[str] = []
    for patch_path in check_df["patch_path"].astype(str).tolist():
        resolved = resolve_patch_path(csv_path, patch_path)
        if not resolved.exists():
            missing.append(str(resolved))
            if len(missing) >= 20:
                break

    if missing:
        check_type = "full" if full_check else f"sampled({len(check_df)})"
        raise FileNotFoundError(
            f"{check_type} patch existence check failed for {csv_path}. "
            f"First missing patch paths: {missing}"
        )


def ensure_source_image_disjoint(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fold: int,
) -> None:
    train_sources = set(train_df["source_image"].astype(str))
    val_sources = set(val_df["source_image"].astype(str))
    test_sources = set(test_df["source_image"].astype(str))

    train_val_overlap = sorted(train_sources.intersection(val_sources))
    if train_val_overlap:
        raise ValueError(
            f"fold_{fold} train.csv and val.csv have overlapping source_image values. "
            f"First overlaps: {train_val_overlap[:20]}"
        )

    test_overlap = sorted(test_sources.intersection(train_sources.union(val_sources)))
    if test_overlap:
        raise ValueError(
            f"fixed_test.csv source_image values appear in fold_{fold} train/val splits. "
            f"First overlaps: {test_overlap[:20]}"
        )


def check_merged_class_presence(
    split_name: str,
    dataframe: pd.DataFrame,
    mapping: dict[str, Any],
    strict: bool,
    require_all_classes: bool,
) -> None:
    merged = apply_label_merge_to_dataframe(dataframe, mapping, strict=strict)
    present_indices = set(merged["merged_label_idx"].astype(int).tolist())
    expected_indices = set(range(int(mapping["num_classes"])))
    missing_indices = sorted(expected_indices.difference(present_indices))
    if not missing_indices:
        return

    label_names = get_label_names_from_mapping(mapping)
    missing_names = [label_names[index] for index in missing_indices]
    message = f"{split_name} is missing merged classes: {missing_names}"
    if require_all_classes:
        raise ValueError(message)
    logging.warning(message)


def replace_label_merge_groups_with_effective_mapping(
    config: dict[str, Any],
    mapping: dict[str, Any],
) -> None:
    """Freeze the effective mapping into config so every fold rebuilds identical indices."""

    index_to_name = {
        int(index): str(name)
        for index, name in mapping["index_to_merged_name"].items()
    }
    effective_groups: dict[str, list[str]] = {
        index_to_name[index]: []
        for index in range(int(mapping["num_classes"]))
    }
    for original_label, merged_name in sorted(mapping["original_to_merged_name"].items()):
        effective_groups[str(merged_name)].append(str(original_label))
    config.setdefault("label_merge", {})["groups"] = effective_groups


def build_fold_paths(config: dict[str, Any]) -> tuple[list[dict[str, Path]], Path]:
    cv_config = config.get("cross_validation", {}) or {}
    split_root = resolve_path(cv_config.get("split_root"))
    if split_root is None:
        raise ValueError("cross_validation.split_root is required.")

    num_folds = int(cv_config.get("num_folds", 3))
    fold_dir_pattern = str(cv_config.get("fold_dir_pattern", "fold_{fold}"))
    train_csv_name = str(cv_config.get("train_csv_name", "train.csv"))
    val_csv_name = str(cv_config.get("val_csv_name", "val.csv"))
    fixed_test_csv = resolve_path(cv_config.get("fixed_test_csv"))
    if fixed_test_csv is None:
        raise ValueError("cross_validation.fixed_test_csv is required.")

    fold_paths: list[dict[str, Path]] = []
    for fold in range(num_folds):
        fold_dir = split_root / fold_dir_pattern.format(fold=fold)
        fold_paths.append(
            {
                "train_csv": fold_dir / train_csv_name,
                "val_csv": fold_dir / val_csv_name,
            }
        )
    return fold_paths, fixed_test_csv


def prepare_output_root(config: dict[str, Any]) -> Path:
    cv_config = config.get("cross_validation", {}) or {}
    output_root = resolve_path(cv_config.get("output_root", "outputs/cross_validation"))
    if output_root is None:
        raise ValueError("cross_validation.output_root could not be resolved.")

    label_merge_config = config.get("label_merge", {}) or {}
    run_name = str(label_merge_config.get("name") or datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"folds": rows}
    metric_columns = [
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_macro_f1",
        "test_auc_ovr",
        "val_image_accuracy",
        "val_image_precision",
        "val_image_recall",
        "val_image_macro_f1",
        "val_image_auc_ovr",
    ]
    dataframe = pd.DataFrame(rows)
    aggregate: dict[str, dict[str, float | None]] = {}
    for column in metric_columns:
        if column not in dataframe.columns:
            continue
        numeric = pd.to_numeric(dataframe[column], errors="coerce")
        aggregate[column] = {
            "mean": None if numeric.dropna().empty else float(numeric.mean()),
            "std": None if numeric.dropna().empty else float(numeric.std(ddof=1)),
        }
    summary["aggregate"] = aggregate
    return summary


def get_metric(metrics: dict[str, Any], key: str, prefer_image: bool = False) -> Any:
    if prefer_image and f"image_{key}" in metrics:
        return metrics.get(f"image_{key}")
    return metrics.get(key)


def main() -> None:
    args = parse_args()
    setup_logging()

    config = load_yaml(resolve_path(args.config))
    cv_config = config.get("cross_validation", {}) or {}
    if not bool(cv_config.get("enabled", False)):
        raise ValueError("cross_validation.enabled must be true for scripts/cross_validate_existing_splits.py.")
    if str(cv_config.get("mode", "existing_splits")) != "existing_splits":
        raise ValueError("Only cross_validation.mode=existing_splits is supported by this script.")

    fold_paths, fixed_test_csv = build_fold_paths(config)
    test_df = read_split_csv(fixed_test_csv)

    full_check_patch_exists = bool(cv_config.get("full_check_patch_exists", False))
    patch_exists_sample_size = int(cv_config.get("patch_exists_sample_size", 1000))
    check_patch_files(fixed_test_csv, test_df, full_check_patch_exists, patch_exists_sample_size)

    fold_dataframes: list[dict[str, pd.DataFrame]] = []
    all_dataframes: list[pd.DataFrame] = [test_df]
    for fold, paths in enumerate(fold_paths):
        train_df = read_split_csv(paths["train_csv"])
        val_df = read_split_csv(paths["val_csv"])
        ensure_source_image_disjoint(train_df, val_df, test_df, fold=fold)
        check_patch_files(paths["train_csv"], train_df, full_check_patch_exists, patch_exists_sample_size)
        check_patch_files(paths["val_csv"], val_df, full_check_patch_exists, patch_exists_sample_size)
        fold_dataframes.append({"train": train_df, "val": val_df})
        all_dataframes.extend([train_df, val_df])

    label_merge_mapping = None
    if is_label_merge_enabled(config):
        label_merge_mapping = build_label_merge_mapping(config)
        strict_label_merge = bool((config.get("label_merge", {}) or {}).get("strict", True))
        validate_label_merge_coverage(all_dataframes, label_merge_mapping, strict=strict_label_merge)
        update_config_num_classes_from_mapping(config, label_merge_mapping)
        for fold, dataframes in enumerate(fold_dataframes):
            check_merged_class_presence(
                f"fold_{fold}/train.csv",
                dataframes["train"],
                label_merge_mapping,
                strict_label_merge,
                require_all_classes=True,
            )
            check_merged_class_presence(
                f"fold_{fold}/val.csv",
                dataframes["val"],
                label_merge_mapping,
                strict_label_merge,
                require_all_classes=False,
            )
        check_merged_class_presence(
            "fixed_test.csv",
            test_df,
            label_merge_mapping,
            strict_label_merge,
            require_all_classes=False,
        )
        replace_label_merge_groups_with_effective_mapping(config, label_merge_mapping)

    output_root = prepare_output_root(config)
    if label_merge_mapping is not None:
        save_label_merge_mapping(label_merge_mapping, output_root / "label_merge_mapping.json")

    summary_rows: list[dict[str, Any]] = []
    for fold, paths in enumerate(fold_paths):
        fold_dir = output_root / f"fold_{fold}"
        fold_config = copy.deepcopy(config)
        fold_config.setdefault("project", {})["run_dir"] = str(fold_dir)
        fold_config.setdefault("build_patch_splits", {})["train_csv"] = str(paths["train_csv"])
        fold_config.setdefault("build_patch_splits", {})["val_csv"] = str(paths["val_csv"])
        fold_config["build_patch_splits"]["test_csv"] = str(fixed_test_csv)

        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "config_resolved.yaml").write_text(
            yaml.safe_dump(fold_config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        if label_merge_mapping is not None:
            save_label_merge_mapping(label_merge_mapping, fold_dir / "label_mapping_effective.json")

        logging.info("Starting fold_%s | train=%s | val=%s | test=%s", fold, paths["train_csv"], paths["val_csv"], fixed_test_csv)
        run_dir = run_training(fold_config)
        checkpoint_path = run_dir / "best_model.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Expected best checkpoint was not created: {checkpoint_path}")

        val_metrics = run_test_from_checkpoint(
            config=fold_config,
            checkpoint_path=checkpoint_path,
            run_dir=run_dir,
            test_csv=paths["val_csv"],
            output_dir=fold_dir,
            artifact_prefix="val",
            save_predictions=True,
        )

        test_metrics: dict[str, Any] = {}
        if bool(cv_config.get("evaluate_fixed_test_each_fold", True)):
            test_metrics = run_test_from_checkpoint(
                config=fold_config,
                checkpoint_path=checkpoint_path,
                run_dir=run_dir,
                test_csv=fixed_test_csv,
                output_dir=fold_dir,
                artifact_prefix="test",
                save_predictions=True,
            )

        summary_rows.append(
            {
                "fold": fold,
                "run_dir": str(run_dir),
                "best_model": str(checkpoint_path),
                "val_image_accuracy": val_metrics.get("image_accuracy"),
                "val_image_precision": val_metrics.get("image_precision"),
                "val_image_recall": val_metrics.get("image_recall"),
                "val_image_macro_f1": val_metrics.get("image_macro_f1"),
                "val_image_auc_ovr": val_metrics.get("image_auc_ovr"),
                "test_accuracy": get_metric(test_metrics, "accuracy", prefer_image=True),
                "test_precision": get_metric(test_metrics, "precision", prefer_image=True),
                "test_recall": get_metric(test_metrics, "recall", prefer_image=True),
                "test_macro_f1": get_metric(test_metrics, "macro_f1", prefer_image=True),
                "test_auc_ovr": get_metric(test_metrics, "auc_ovr", prefer_image=True),
            }
        )

    summary = summarize_metrics(summary_rows)
    pd.DataFrame(summary_rows).to_csv(output_root / "cv_summary.csv", index=False, encoding="utf-8")
    (output_root / "cv_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logging.info("Saved cross-validation summary to %s", output_root)


if __name__ == "__main__":
    main()
