from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from src.datasets import build_patch_dataloader, build_whole_image_dataloader
from src.losses import build_loss
from src.models import build_model
from src.utils.metrics import (
    build_single_level_evaluation_result,
    compute_multilevel_classification_metrics,
    save_confusion_matrix_figure,
    save_metrics_json,
)


def _resolve_path(project_root: Path, path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root / path


def _select_device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("train", {}).get("device", "cuda")).lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _get_task_mode(config: dict[str, Any]) -> str:
    task_mode = str(config.get("task", {}).get("mode", "patch")).lower()
    if task_mode not in {"patch", "whole_image"}:
        raise ValueError(f"Unsupported task.mode: {task_mode}")
    return task_mode


def _compute_class_counts(
    train_csv: Path,
    num_classes: int | None = None,
    label_names: list[str] | None = None,
) -> list[int]:
    dataframe = pd.read_csv(train_csv)
    if "label_idx" in dataframe.columns:
        label_counts = Counter(dataframe["label_idx"].astype(int).tolist())
        total_classes = int(num_classes) if num_classes is not None else (max(label_counts.keys()) + 1 if label_counts else 0)
        return [int(label_counts.get(index, 0)) for index in range(total_classes)]

    label_series = dataframe["label"].astype(str)
    counts = Counter(label_series.tolist())
    if label_names is not None:
        unknown_labels = sorted(set(counts).difference(label_names))
        if unknown_labels:
            raise ValueError(f"Found labels in {train_csv} that are missing from label_names: {unknown_labels}")
        return [int(counts.get(label_name, 0)) for label_name in label_names]

    labels = sorted(counts)
    return [int(counts.get(label, 0)) for label in labels]


def _load_label_names(label_mapping_path: Path | None, num_classes: int | None = None) -> list[str] | None:
    if label_mapping_path is None or not label_mapping_path.exists():
        return None

    payload = json.loads(label_mapping_path.read_text(encoding="utf-8"))
    if "index_to_label" in payload:
        index_to_label = {int(index): label for index, label in payload["index_to_label"].items()}
        if num_classes is None:
            num_classes = max(index_to_label.keys()) + 1 if index_to_label else 0
        return [index_to_label.get(index, str(index)) for index in range(num_classes)]

    if "label_to_index" in payload:
        label_to_index = {label: int(index) for label, index in payload["label_to_index"].items()}
        return [label for label, _ in sorted(label_to_index.items(), key=lambda item: item[1])]

    return None


def _save_per_class_metrics_csv(metrics: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "class_index",
        "class_name",
        "precision",
        "recall",
        "f1",
        "support",
        "predicted_count",
        "true_count",
        "specificity",
        "one_vs_rest_accuracy",
    ]
    dataframe = pd.DataFrame(metrics.get("per_class_metrics", []))
    if dataframe.empty:
        dataframe = pd.DataFrame(columns=columns)
    else:
        dataframe = dataframe.reindex(columns=columns)
    dataframe.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def _save_level_artifacts(metrics: dict[str, Any], output_dir: Path, prefix: str) -> None:
    metrics_path = save_metrics_json(metrics, output_dir / f"{prefix}_metrics.json")
    per_class_csv_path = _save_per_class_metrics_csv(metrics, output_dir / f"{prefix}_per_class_metrics.csv")
    confusion_path = save_confusion_matrix_figure(
        confusion=metrics["confusion_matrix"],
        label_names=metrics["labels"],
        output_path=output_dir / f"{prefix}_confusion_matrix.png",
    )
    logging.info("Saved %s metrics to %s", prefix, metrics_path)
    logging.info("Saved %s per-class metrics to %s", prefix, per_class_csv_path)
    logging.info("Saved %s confusion matrix to %s", prefix, confusion_path)


def _resolve_split_dir(project_root: Path, config: dict[str, Any]) -> Path:
    return _resolve_path(
        project_root,
        config.get("data", {}).get("split_dir") or config.get("build_patch_splits", {}).get("output_dir") or "data/splits",
    )


def _resolve_label_mapping_path(project_root: Path, config: dict[str, Any], split_dir: Path) -> Path:
    return _resolve_path(
        project_root,
        config.get("build_patch_splits", {}).get("label_mapping_path") or split_dir / "label_mapping.json",
    )


def _resolve_split_csvs(
    project_root: Path,
    config: dict[str, Any],
    task_mode: str,
) -> tuple[Path, Path, Path]:
    split_dir = _resolve_split_dir(project_root, config)
    label_mapping_path = _resolve_label_mapping_path(project_root, config, split_dir)

    if task_mode == "whole_image":
        whole_image_config = config.get("whole_image", {})
        train_csv = _resolve_path(project_root, whole_image_config.get("train_csv") or split_dir / "train_images.csv")
        test_csv = _resolve_path(project_root, whole_image_config.get("test_csv") or split_dir / "test_images.csv")
        return train_csv, test_csv, label_mapping_path

    train_csv = split_dir / "train.csv"
    test_csv = split_dir / "test.csv"
    return train_csv, test_csv, label_mapping_path


@torch.no_grad()
def test_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    output_dir: str | Path,
    label_names: list[str] | None = None,
    use_amp: bool = False,
    evaluation_config: dict[str, Any] | None = None,
    task_mode: str = "patch",
) -> dict[str, Any]:
    """Run evaluation on the test set and save metrics/artifacts."""

    task_mode = str(task_mode).lower()
    if task_mode not in {"patch", "whole_image"}:
        raise ValueError(f"Unsupported task_mode: {task_mode}")

    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    running_loss = 0.0
    predictions: list[int] = []
    targets: list[int] = []
    probabilities: list[list[float]] = []
    source_images: list[str] = []
    patch_paths: list[str] = []

    progress = tqdm(dataloader, desc="Test", leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        batch_size = images.size(0)

        if task_mode == "patch":
            if "source_image" in batch:
                source_images.extend([str(value) for value in batch["source_image"]])
            if "patch_path" in batch:
                patch_paths.extend([str(value) for value in batch["patch_path"]])

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        running_loss += float(loss.detach().item()) * batch_size
        predictions.extend(preds.cpu().tolist())
        targets.extend(labels.cpu().tolist())
        probabilities.extend(probs.cpu().tolist())
        progress.set_postfix(loss=f"{loss.detach().item():.4f}")

    test_loss = running_loss / max(1, len(dataloader.dataset))
    if task_mode == "whole_image":
        metrics = build_single_level_evaluation_result(
            targets=targets,
            predictions=predictions,
            probabilities=probabilities,
            label_names=label_names,
            loss=float(test_loss),
            sample_level="image",
            aggregation="model_direct",
        )
    else:
        metrics = compute_multilevel_classification_metrics(
            targets=targets,
            predictions=predictions,
            probabilities=probabilities,
            label_names=label_names,
            source_images=source_images if source_images else None,
            patch_paths=patch_paths if patch_paths else None,
            loss=float(test_loss),
            evaluation_config=evaluation_config,
        )

    metrics_path = save_metrics_json(metrics, output_dir / "metrics.json")
    logging.info("Saved combined test metrics to %s", metrics_path)

    if "patch_metrics" in metrics:
        _save_level_artifacts(metrics["patch_metrics"], output_dir, "patch")
    if "image_metrics" in metrics:
        _save_level_artifacts(metrics["image_metrics"], output_dir, "image")

    logging.info(
        "Test summary | primary=%s acc=%.4f macro_f1=%.4f loss=%.4f",
        metrics["primary_metric_level"],
        metrics["accuracy"],
        metrics["macro_f1"],
        metrics["loss"],
    )
    if "patch_accuracy" in metrics:
        logging.info("Patch metrics | acc=%.4f macro_f1=%.4f", metrics["patch_accuracy"], metrics["patch_macro_f1"])
    if "image_accuracy" in metrics:
        logging.info(
            "Image metrics | aggregation=%s images=%s acc=%.4f macro_f1=%.4f",
            metrics["aggregation"],
            metrics.get("num_images", 0),
            metrics["image_accuracy"],
            metrics["image_macro_f1"],
        )
    return metrics


def run_test_from_checkpoint(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    run_dir: str | Path,
) -> dict[str, Any]:
    """Build the test pipeline from config and evaluate a saved checkpoint."""

    project_root = Path(__file__).resolve().parents[2]
    run_dir = Path(run_dir)
    device = _select_device(config)
    use_amp = bool(config.get("train", {}).get("mixed_precision", True) and device.type == "cuda")
    task_mode = _get_task_mode(config)

    train_csv, test_csv, label_mapping_path = _resolve_split_csvs(project_root, config, task_mode)

    if task_mode == "whole_image":
        test_loader = build_whole_image_dataloader(
            csv_file=test_csv,
            mode="test",
            config=config,
            label_mapping_path=label_mapping_path,
            project_root=project_root,
            shuffle=False,
        )
    else:
        test_loader = build_patch_dataloader(
            csv_file=test_csv,
            mode="test",
            config=config,
            label_mapping_path=label_mapping_path,
            project_root=project_root,
            shuffle=False,
        )

    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    label_names = _load_label_names(label_mapping_path, num_classes=config.get("data", {}).get("num_classes"))
    class_counts = _compute_class_counts(
        train_csv,
        num_classes=config.get("data", {}).get("num_classes"),
        label_names=label_names,
    )
    criterion = build_loss(config, class_counts=class_counts, device=device)

    logging.info("Testing checkpoint: %s", checkpoint_path)
    logging.info("Task mode: %s", task_mode)
    if task_mode == "whole_image":
        logging.info("Test images: %s", len(test_loader.dataset))
    else:
        logging.info("Test samples: %s", len(test_loader.dataset))

    return test_model(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        output_dir=run_dir / "test",
        label_names=label_names,
        use_amp=use_amp,
        evaluation_config=config,
        task_mode=task_mode,
    )