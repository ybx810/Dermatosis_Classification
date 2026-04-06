from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.classifiers import describe_preprocessor, run_model_selection, transform_features
from src.ml.evaluate import build_predictions_dataframe, compute_metrics_for_split, get_prediction_scores, save_evaluation_artifacts
from src.ml.features import extract_all_splits
from src.ml.io import (
    build_feature_run_dir,
    build_ml_run_dir,
    load_feature_split,
    load_label_names,
    resolve_project_path,
    resolve_split_paths,
    save_dataframe,
    save_json,
    save_yaml_snapshot,
)
from src.utils.io import load_yaml


FEATURE_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate sklearn classifiers on extracted whole-image CNN features.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--feature-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--extract-first", action="store_true")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def resolve_path(path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None
    return resolve_project_path(path_value, PROJECT_ROOT)


def _feature_files_exist(feature_dir: Path) -> bool:
    return all((feature_dir / f"{split}_features.npz").exists() for split in FEATURE_SPLITS)


def _find_latest_feature_dir(config: dict[str, Any]) -> Path | None:
    project_name = str(config.get("project", {}).get("name", "whole-image-classification"))
    ml_output_root = resolve_path(config.get("ml_experiment", {}).get("output_dir") or config.get("project", {}).get("output_dir", "outputs"))
    if ml_output_root is None:
        return None

    features_root = ml_output_root / project_name / "ml_features"
    if not features_root.exists():
        return None

    candidates = [path for path in features_root.iterdir() if path.is_dir() and _feature_files_exist(path)]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)[0]


def _resolve_feature_dir(config: dict[str, Any], args: argparse.Namespace) -> tuple[Path, dict[str, Any] | None]:
    configured_feature_dir = resolve_path(args.feature_dir or config.get("ml_experiment", {}).get("feature_dir"))

    if args.extract_first:
        feature_dir = configured_feature_dir
        if feature_dir is None:
            feature_dir = build_feature_run_dir(config, run_name=args.run_name, project_root=PROJECT_ROOT)
        logging.info("Extracting features before ML training into: %s", feature_dir)
        extract_info = extract_all_splits(config=config, output_dir=feature_dir, run_tag=args.run_name)
        return feature_dir, extract_info

    if configured_feature_dir is not None and _feature_files_exist(configured_feature_dir):
        return configured_feature_dir, None

    reuse_saved = bool(config.get("ml_experiment", {}).get("reuse_saved_features", False))
    if reuse_saved:
        latest = _find_latest_feature_dir(config)
        if latest is not None:
            logging.info("Using latest saved feature directory: %s", latest)
            return latest, None

    if configured_feature_dir is not None and not reuse_saved:
        logging.info("Configured feature_dir does not contain complete features, extracting now: %s", configured_feature_dir)
        extract_info = extract_all_splits(config=config, output_dir=configured_feature_dir, run_tag=args.run_name)
        return configured_feature_dir, extract_info

    if not reuse_saved:
        auto_dir = build_feature_run_dir(config, run_name=args.run_name, project_root=PROJECT_ROOT)
        logging.info("No reusable features specified. Extracting features into: %s", auto_dir)
        extract_info = extract_all_splits(config=config, output_dir=auto_dir, run_tag=args.run_name)
        return auto_dir, extract_info

    raise FileNotFoundError(
        "No usable feature files found. Use --extract-first, or set ml_experiment.feature_dir/reuse_saved_features accordingly."
    )


def _load_feature_info(feature_dir: Path) -> dict[str, Any]:
    feature_info_path = feature_dir / "feature_info.json"
    if not feature_info_path.exists():
        return {}
    return json.loads(feature_info_path.read_text(encoding="utf-8"))


def _resolve_label_names(config: dict[str, Any], feature_data: dict[str, dict[str, Any]]) -> list[str] | None:
    split_paths = resolve_split_paths(config, project_root=PROJECT_ROOT)
    label_names = load_label_names(split_paths["label_mapping_path"], num_classes=config.get("data", {}).get("num_classes"))
    if label_names is not None:
        return label_names

    train_y = np.asarray(feature_data["train"]["y"], dtype=np.int64)
    train_label_names = feature_data["train"].get("label_name", [])
    if train_y.size == 0 or not train_label_names:
        return None

    index_to_label: dict[int, str] = {}
    for index, label_name in zip(train_y.tolist(), train_label_names):
        index_to_label[int(index)] = str(label_name)

    max_index = int(train_y.max())
    return [index_to_label.get(index, str(index)) for index in range(max_index + 1)]


def _verify_feature_counts_with_split_csv(config: dict[str, Any], feature_data: dict[str, dict[str, Any]]) -> None:
    split_paths = resolve_split_paths(config, project_root=PROJECT_ROOT)
    for split_name in FEATURE_SPLITS:
        csv_path = split_paths[f"{split_name}_csv"]
        expected = int(pd.read_csv(csv_path).shape[0])
        actual = int(feature_data[split_name]["X"].shape[0])
        if expected != actual:
            raise ValueError(
                f"{split_name} sample count mismatch: CSV has {expected} rows but features have {actual} rows ({csv_path})"
            )


def main() -> None:
    args = parse_args()
    setup_logging()

    config = load_yaml(resolve_path(args.config))
    ml_enabled = bool(config.get("ml_experiment", {}).get("enabled", False))
    if not ml_enabled:
        logging.warning("ml_experiment.enabled is false. Running ML training because this script was called explicitly.")

    feature_dir, extraction_info = _resolve_feature_dir(config, args)
    if not _feature_files_exist(feature_dir):
        raise FileNotFoundError(f"Incomplete feature files under {feature_dir}")

    run_dir = build_ml_run_dir(config, run_name=args.run_name, project_root=PROJECT_ROOT)
    save_yaml_snapshot(config, run_dir / "ml_config.yaml")

    feature_data = {split_name: load_feature_split(feature_dir, split_name) for split_name in FEATURE_SPLITS}
    _verify_feature_counts_with_split_csv(config, feature_data)

    label_names = _resolve_label_names(config, feature_data)

    train_X = np.asarray(feature_data["train"]["X"], dtype=np.float32)
    train_y = np.asarray(feature_data["train"]["y"], dtype=np.int64)
    val_X = np.asarray(feature_data["val"]["X"], dtype=np.float32)
    val_y = np.asarray(feature_data["val"]["y"], dtype=np.int64)
    test_X_raw = np.asarray(feature_data["test"]["X"], dtype=np.float32)
    test_y = np.asarray(feature_data["test"]["y"], dtype=np.int64)

    selection = run_model_selection(
        train_features=train_X,
        train_labels=train_y,
        val_features=val_X,
        val_labels=val_y,
        config=config,
        label_names=label_names,
    )

    test_X = transform_features(test_X_raw, selection.preprocessor)
    test_predictions = selection.best_model.predict(test_X)
    test_scores = get_prediction_scores(selection.best_model, test_X)

    test_metrics = compute_metrics_for_split(
        y_true=test_y,
        y_pred=test_predictions,
        label_names=label_names,
        score_matrix=test_scores,
    )

    feature_info = extraction_info or _load_feature_info(feature_dir)
    test_metrics["selected_classifier"] = selection.best_classifier
    test_metrics["selected_hyperparams"] = selection.best_hyperparams
    test_metrics["feature_source"] = feature_info.get("feature_source")
    test_metrics["backbone"] = feature_info.get("backbone")
    test_metrics["num_features"] = int(test_X.shape[1]) if test_X.ndim == 2 else 0

    artifact_paths = save_evaluation_artifacts(test_metrics, run_dir)

    val_selection_df = pd.DataFrame(selection.val_records)
    val_selection_path = save_dataframe(val_selection_df, run_dir / "val_model_selection.csv")

    predictions_df = build_predictions_dataframe(
        source_images=feature_data["test"]["source_image"],
        y_true=test_y,
        y_pred=test_predictions,
        label_names=label_names,
        score_matrix=test_scores,
    )
    predictions_path = save_dataframe(predictions_df, run_dir / "test_predictions.csv")

    model_path = run_dir / "best_model.joblib"
    preprocessors_path = run_dir / "preprocessors.joblib"
    joblib.dump(selection.best_model, model_path)
    joblib.dump(
        {
            "scaler": selection.preprocessor.scaler,
            "pca": selection.preprocessor.pca,
        },
        preprocessors_path,
    )

    run_summary = {
        "run_dir": str(run_dir),
        "feature_dir": str(feature_dir),
        "selected_classifier": selection.best_classifier,
        "selected_hyperparams": selection.best_hyperparams,
        "primary_metric": str(config.get("ml_experiment", {}).get("model_selection", {}).get("primary_metric", "macro_f1")),
        "best_val_metrics": selection.best_val_metrics,
        "test_metrics": {
            "accuracy": test_metrics.get("accuracy"),
            "precision": test_metrics.get("precision"),
            "recall": test_metrics.get("recall"),
            "macro_f1": test_metrics.get("macro_f1"),
            "auc_ovr": test_metrics.get("auc_ovr"),
        },
        "preprocessing": describe_preprocessor(selection.preprocessor),
        "artifacts": {
            "metrics_json": str(artifact_paths["metrics"]),
            "per_class_metrics_csv": str(artifact_paths["per_class_metrics"]),
            "confusion_matrix_png": str(artifact_paths["confusion_matrix"]),
            "val_model_selection_csv": str(val_selection_path),
            "test_predictions_csv": str(predictions_path),
            "best_model_joblib": str(model_path),
            "preprocessors_joblib": str(preprocessors_path),
        },
    }
    save_json(run_summary, run_dir / "run_summary.json")

    logging.info("ML run directory: %s", run_dir)
    logging.info(
        "Selected model: %s %s | test acc=%.4f macro_f1=%.4f auc_ovr=%s",
        selection.best_classifier,
        selection.best_hyperparams,
        test_metrics.get("accuracy", 0.0),
        test_metrics.get("macro_f1", 0.0),
        "N/A" if test_metrics.get("auc_ovr") is None else f"{test_metrics['auc_ovr']:.4f}",
    )


if __name__ == "__main__":
    main()
