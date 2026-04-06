from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path_value: str | Path | None, project_root: Path = PROJECT_ROOT) -> Path | None:
    if path_value is None:
        return None
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return project_root / candidate


def _resolve_output_root(config: dict[str, Any], project_root: Path = PROJECT_ROOT) -> Path:
    ml_config = config.get("ml_experiment", {})
    project_config = config.get("project", {})
    output_dir = ml_config.get("output_dir") or project_config.get("output_dir", "outputs")
    resolved = resolve_project_path(output_dir, project_root)
    if resolved is None:
        raise ValueError("Failed to resolve output directory.")
    return resolved


def _resolve_project_name(config: dict[str, Any]) -> str:
    return str(config.get("project", {}).get("name", "whole-image-classification"))


def build_feature_run_dir(
    config: dict[str, Any],
    run_name: str | None = None,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _resolve_output_root(config, project_root) / _resolve_project_name(config) / "ml_features" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_ml_run_dir(
    config: dict[str, Any],
    run_name: str | None = None,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _resolve_output_root(config, project_root) / _resolve_project_name(config) / "ml_runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_split_paths(config: dict[str, Any], project_root: Path = PROJECT_ROOT) -> dict[str, Path]:
    split_dir = resolve_project_path(
        config.get("data", {}).get("split_dir") or config.get("build_image_splits", {}).get("output_dir") or "data/splits",
        project_root,
    )
    if split_dir is None:
        raise ValueError("Failed to resolve split directory.")

    whole_image_config = config.get("whole_image", {})
    label_mapping_path = resolve_project_path(
        config.get("build_image_splits", {}).get("label_mapping_path") or split_dir / "label_mapping.json",
        project_root,
    )
    train_csv = resolve_project_path(whole_image_config.get("train_csv") or split_dir / "train_images.csv", project_root)
    val_csv = resolve_project_path(whole_image_config.get("val_csv") or split_dir / "val_images.csv", project_root)
    test_csv = resolve_project_path(whole_image_config.get("test_csv") or split_dir / "test_images.csv", project_root)

    if train_csv is None or val_csv is None or test_csv is None or label_mapping_path is None:
        raise ValueError("Failed to resolve split CSV paths.")

    return {
        "split_dir": split_dir,
        "label_mapping_path": label_mapping_path,
        "train_csv": train_csv,
        "val_csv": val_csv,
        "test_csv": test_csv,
    }


def load_label_names(label_mapping_path: Path | None, num_classes: int | None = None) -> list[str] | None:
    if label_mapping_path is None or not label_mapping_path.exists():
        return None

    payload = json.loads(label_mapping_path.read_text(encoding="utf-8"))
    if "index_to_label" in payload:
        index_to_label = {int(index): label for index, label in payload["index_to_label"].items()}
        if num_classes is None:
            num_classes = max(index_to_label.keys()) + 1 if index_to_label else 0
        return [str(index_to_label.get(index, str(index))) for index in range(num_classes)]

    if "label_to_index" in payload:
        label_to_index = {str(label): int(index) for label, index in payload["label_to_index"].items()}
        return [label for label, _ in sorted(label_to_index.items(), key=lambda item: item[1])]

    return None


def save_yaml_snapshot(config: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return output_path


def save_json(payload: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def save_dataframe(dataframe: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def save_feature_split(
    output_dir: str | Path,
    split_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    source_images: list[str],
    label_names: list[str],
    image_paths: list[str] | None = None,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split = str(split_name).lower()
    feature_path = output_dir / f"{split}_features.npz"
    metadata_path = output_dir / f"{split}_metadata.csv"

    if features.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix for {split}, got shape={features.shape}")

    num_samples = int(features.shape[0])
    if labels.shape[0] != num_samples:
        raise ValueError(f"Label count mismatch for {split}: {labels.shape[0]} != {num_samples}")
    if len(source_images) != num_samples:
        raise ValueError(f"source_image count mismatch for {split}: {len(source_images)} != {num_samples}")
    if len(label_names) != num_samples:
        raise ValueError(f"label_name count mismatch for {split}: {len(label_names)} != {num_samples}")

    source_array = np.asarray(source_images, dtype=object)
    label_name_array = np.asarray(label_names, dtype=object)
    sample_index = np.arange(num_samples, dtype=np.int64)

    payload: dict[str, Any] = {
        "X": np.asarray(features, dtype=np.float32),
        "y": np.asarray(labels, dtype=np.int64),
        "source_image": source_array,
        "label_name": label_name_array,
        "sample_index": sample_index,
    }

    metadata_dict = {
        "sample_index": sample_index,
        "source_image": source_images,
        "label_name": label_names,
        "label_idx": np.asarray(labels, dtype=np.int64),
    }

    if image_paths is not None:
        if len(image_paths) != num_samples:
            raise ValueError(f"image_path count mismatch for {split}: {len(image_paths)} != {num_samples}")
        payload["image_path"] = np.asarray(image_paths, dtype=object)
        metadata_dict["image_path"] = image_paths

    np.savez_compressed(feature_path, **payload)
    metadata = pd.DataFrame(metadata_dict)
    metadata.to_csv(metadata_path, index=False, encoding="utf-8")

    return feature_path, metadata_path


def load_feature_split(feature_dir: str | Path, split_name: str) -> dict[str, Any]:
    feature_path = Path(feature_dir) / f"{split_name.lower()}_features.npz"
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

    with np.load(feature_path, allow_pickle=True) as payload:
        required_keys = {"X", "y", "source_image", "label_name"}
        missing = required_keys.difference(payload.files)
        if missing:
            raise ValueError(f"Feature file {feature_path} missing keys: {sorted(missing)}")

        data: dict[str, Any] = {
            "X": np.asarray(payload["X"], dtype=np.float32),
            "y": np.asarray(payload["y"], dtype=np.int64),
            "source_image": payload["source_image"].astype(str).tolist(),
            "label_name": payload["label_name"].astype(str).tolist(),
            "sample_index": (
                np.asarray(payload["sample_index"], dtype=np.int64)
                if "sample_index" in payload.files
                else np.arange(int(payload["X"].shape[0]), dtype=np.int64)
            ),
            "feature_path": feature_path,
        }
        if "image_path" in payload.files:
            data["image_path"] = payload["image_path"].astype(str).tolist()
        return data


def verify_feature_alignment(data: dict[str, Any], split_name: str) -> None:
    split = str(split_name)
    num_samples = int(data["X"].shape[0])
    if int(data["y"].shape[0]) != num_samples:
        raise ValueError(f"{split} y length mismatch.")
    if len(data["source_image"]) != num_samples:
        raise ValueError(f"{split} source_image length mismatch.")
    if len(data["label_name"]) != num_samples:
        raise ValueError(f"{split} label_name length mismatch.")
