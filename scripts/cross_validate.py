from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main import run_training
from src.utils.io import load_yaml

CV_METRIC_KEYS = ("accuracy", "precision", "recall", "macro_f1", "auc_ovr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run k-fold cross-validation training for whole-image classification.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--folds-dir", type=str, default=None)
    parser.add_argument("--n-splits", type=int, default=None)
    return parser.parse_args()


def resolve_path(path_value: str | Path | None, default: str | Path | None = None) -> Path:
    raw_value = path_value if path_value is not None else default
    if raw_value is None:
        raise ValueError("resolve_path requires either path_value or default.")
    path = Path(raw_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def setup_crossval_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cross_validate")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def build_crossval_root(config: dict[str, Any]) -> Path:
    project_cfg = config.get("project", {})
    output_root = resolve_path(project_cfg.get("output_dir", "outputs"))
    project_name = str(project_cfg.get("name", "whole-image-classification"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    crossval_root = output_root / project_name / "crossval" / timestamp
    crossval_root.mkdir(parents=True, exist_ok=True)
    return crossval_root


def resolve_folds_dir(config: dict[str, Any], args: argparse.Namespace) -> Path:
    split_cfg = config.get("build_image_splits", {})
    folds_dir_value = args.folds_dir or split_cfg.get("folds_dir") or "data/splits/cv3"
    return resolve_path(folds_dir_value)


def resolve_n_splits(config: dict[str, Any], args: argparse.Namespace) -> int:
    split_cfg = config.get("build_image_splits", {})
    n_splits = int(args.n_splits if args.n_splits is not None else split_cfg.get("n_splits", 3))
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got: {n_splits}")
    return n_splits


def collect_fold_csvs(folds_dir: Path, n_splits: int) -> list[dict[str, Any]]:
    if not folds_dir.exists():
        raise FileNotFoundError(
            f"Fold directory does not exist: {folds_dir}. "
            "Run scripts/build_image_splits.py with build_image_splits.mode=kfold first."
        )

    fold_items: list[dict[str, Any]] = []
    for fold_idx in range(n_splits):
        fold_name = f"fold_{fold_idx}"
        train_csv = folds_dir / f"{fold_name}_train_images.csv"
        val_csv = folds_dir / f"{fold_name}_val_images.csv"
        if not train_csv.exists() or not val_csv.exists():
            raise FileNotFoundError(
                f"Missing fold CSV files for {fold_name}. Expected:\n"
                f"- {train_csv}\n"
                f"- {val_csv}\n"
                "Please regenerate folds via scripts/build_image_splits.py."
            )

        fold_items.append(
            {
                "fold_idx": fold_idx,
                "fold_name": fold_name,
                "train_csv": train_csv,
                "val_csv": val_csv,
            }
        )
    return fold_items


def write_csv(rows: list[dict[str, Any]], fieldnames: list[str], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return output_path


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def compute_metric_statistics(per_fold_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for metric_name in CV_METRIC_KEYS:
        values: list[float] = []
        for row in per_fold_rows:
            numeric = _to_float(row.get(metric_name))
            if numeric is not None:
                values.append(numeric)

        if values:
            mean_value = float(statistics.mean(values))
            std_value = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
        else:
            mean_value = None
            std_value = None

        summary[metric_name] = {
            "values": values,
            "mean": mean_value,
            "std": std_value,
            "num_folds_with_value": len(values),
        }
    return summary


def run_cross_validation(config: dict[str, Any], args: argparse.Namespace) -> Path:
    crossval_root = build_crossval_root(config)
    logger = setup_crossval_logger(crossval_root / "crossval.log")

    split_cfg = config.get("build_image_splits", {})
    split_mode = str(split_cfg.get("mode", "single")).lower()
    if split_mode != "kfold":
        logger.warning(
            "build_image_splits.mode is '%s' instead of 'kfold'. Continuing with folds_dir-based cross-validation.",
            split_mode,
        )

    folds_dir = resolve_folds_dir(config, args)
    n_splits = resolve_n_splits(config, args)
    fold_items = collect_fold_csvs(folds_dir, n_splits)

    crossval_config_path = crossval_root / "crossval_config.yaml"
    crossval_config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    logger.info("Cross-validation root: %s", crossval_root)
    logger.info("Folds directory: %s", folds_dir)
    logger.info("n_splits: %d", n_splits)

    per_fold_rows: list[dict[str, Any]] = []
    for fold_item in fold_items:
        fold_idx = int(fold_item["fold_idx"])
        fold_name = str(fold_item["fold_name"])
        train_csv = Path(fold_item["train_csv"])
        val_csv = Path(fold_item["val_csv"])

        logger.info("Starting %s | train_csv=%s | val_csv=%s", fold_name, train_csv, val_csv)
        fold_config = copy.deepcopy(config)
        fold_config.setdefault("whole_image", {})
        fold_config["whole_image"]["train_csv"] = str(train_csv)
        fold_config["whole_image"]["val_csv"] = str(val_csv)

        result = run_training(
            fold_config,
            run_dir=crossval_root,
            run_name=fold_name,
        )

        best_metrics = result.get("best_metrics", {}) or {}
        row: dict[str, Any] = {
            "fold": fold_name,
            "fold_index": fold_idx,
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
            "run_dir": result.get("run_dir"),
            "best_model_path": result.get("best_model_path"),
            "best_epoch": result.get("best_epoch"),
            "primary_metric": result.get("primary_metric"),
            "best_metric_value": result.get("best_metric_value"),
            "history_path": result.get("final_history_path"),
        }
        for metric_name in CV_METRIC_KEYS:
            row[metric_name] = best_metrics.get(metric_name)
        per_fold_rows.append(row)

        logger.info(
            "Completed %s | best_epoch=%s %s=%.4f",
            fold_name,
            row.get("best_epoch"),
            row.get("primary_metric"),
            float(row.get("best_metric_value", 0.0)),
        )

    per_fold_rows = sorted(per_fold_rows, key=lambda item: int(item["fold_index"]))
    per_fold_fields = [
        "fold",
        "fold_index",
        "train_csv",
        "val_csv",
        "run_dir",
        "best_model_path",
        "best_epoch",
        "primary_metric",
        "best_metric_value",
        "accuracy",
        "precision",
        "recall",
        "macro_f1",
        "auc_ovr",
        "history_path",
    ]
    per_fold_csv_path = write_csv(per_fold_rows, per_fold_fields, crossval_root / "per_fold_results.csv")

    metric_statistics = compute_metric_statistics(per_fold_rows)
    summary_rows = [
        {
            "metric": metric_name,
            "mean": metric_payload.get("mean"),
            "std": metric_payload.get("std"),
            "num_folds_with_value": metric_payload.get("num_folds_with_value"),
        }
        for metric_name, metric_payload in metric_statistics.items()
    ]
    crossval_summary_csv = write_csv(
        summary_rows,
        ["metric", "mean", "std", "num_folds_with_value"],
        crossval_root / "crossval_summary.csv",
    )

    summary_json_payload = {
        "project_name": str(config.get("project", {}).get("name", "whole-image-classification")),
        "crossval_root": str(crossval_root),
        "crossval_config_path": str(crossval_config_path),
        "folds_dir": str(folds_dir),
        "n_splits": int(n_splits),
        "seed": int(split_cfg.get("seed", config.get("train", {}).get("seed", 42))),
        "primary_metric": str(config.get("evaluation", {}).get("primary_metric", "macro_f1")).lower(),
        "metrics": metric_statistics,
        "per_fold_results": per_fold_rows,
        "per_fold_results_csv": str(per_fold_csv_path),
        "crossval_summary_csv": str(crossval_summary_csv),
    }
    summary_json_path = crossval_root / "crossval_summary.json"
    summary_json_path.write_text(json.dumps(summary_json_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Saved per-fold results to %s", per_fold_csv_path)
    logger.info("Saved cross-validation summary JSON to %s", summary_json_path)
    logger.info("Saved cross-validation summary CSV to %s", crossval_summary_csv)
    return crossval_root


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config, "configs/default.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_yaml(config_path)
    run_cross_validation(config, args)


if __name__ == "__main__":
    main()

