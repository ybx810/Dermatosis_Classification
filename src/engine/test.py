from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from src.datasets.skin_mil_dataset import build_mil_dataloader
from src.datasets.skin_patch_dataset import build_dataloader
from src.losses import build_loss
from src.models import build_model
from src.utils.metrics import (
    aggregate_bag_logits_to_image,
    build_single_level_evaluation_result,
    compute_classification_metrics,
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


def _get_task_mode(config: dict[str, Any]) -> str:
    task_config = config.get("task", {})
    mode = str(task_config.get("mode", "patch")).lower() if isinstance(task_config, dict) else "patch"
    if mode not in {"patch", "mil"}:
        raise ValueError(f"Unsupported task mode: {mode}. Expected one of ['patch', 'mil']")
    return mode


def _select_device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("train", {}).get("device", "cuda")).lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _compute_class_counts(
    train_csv: Path,
    task_mode: str,
    num_classes: int | None = None,
    label_names: list[str] | None = None,
) -> list[int]:
    dataframe = pd.read_csv(train_csv)
    if task_mode == "mil":
        if "source_image" not in dataframe.columns:
            raise ValueError(f"MIL mode requires source_image in split CSV: {train_csv}")

        dataframe["source_image"] = dataframe["source_image"].fillna("").astype(str).str.strip()
        if (dataframe["source_image"].str.len() == 0).any():
            raise ValueError(f"MIL mode requires non-empty source_image values in {train_csv}")

        if "label_idx" in dataframe.columns:
            bag_label_indices: list[int] = []
            for source_image, group_df in dataframe.groupby("source_image", sort=True):
                label_values = sorted(group_df["label_idx"].astype(int).unique().tolist())
                if len(label_values) != 1:
                    raise ValueError(
                        f"MIL class count computation requires a single label_idx per source_image. "
                        f"source_image={source_image!r}, label_idx={label_values}"
                    )
                bag_label_indices.append(int(label_values[0]))

            counts = Counter(bag_label_indices)
            total_classes = int(num_classes) if num_classes is not None else (max(counts.keys()) + 1 if counts else 0)
            return [int(counts.get(index, 0)) for index in range(total_classes)]

        dataframe["label"] = dataframe["label"].astype(str)
        bag_labels: list[str] = []
        for source_image, group_df in dataframe.groupby("source_image", sort=True):
            labels = sorted(group_df["label"].unique().tolist())
            if len(labels) != 1:
                raise ValueError(
                    f"MIL class count computation requires a single label per source_image. "
                    f"source_image={source_image!r}, labels={labels}"
                )
            bag_labels.append(labels[0])

        counts = Counter(bag_labels)
        if label_names is not None:
            unknown_labels = sorted(set(counts).difference(label_names))
            if unknown_labels:
                raise ValueError(
                    f"Found labels in {train_csv} that are missing from label_names: {unknown_labels}"
                )
            return [int(counts.get(label_name, 0)) for label_name in label_names]

        labels = sorted(counts)
        return [int(counts.get(label, 0)) for label in labels]

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


def _save_level_artifacts(metrics: dict[str, Any], output_dir: Path, prefix: str) -> None:
    metrics_path = save_metrics_json(metrics, output_dir / f"{prefix}_metrics.json")
    confusion_path = save_confusion_matrix_figure(
        confusion=metrics["confusion_matrix"],
        label_names=metrics["labels"],
        output_path=output_dir / f"{prefix}_confusion_matrix.png",
    )
    logging.info("Saved %s metrics to %s", prefix, metrics_path)
    logging.info("Saved %s confusion matrix to %s", prefix, confusion_path)


def _move_bag_batch_to_device(bag_batch: list[torch.Tensor], device: torch.device) -> list[torch.Tensor]:
    return [bag.to(device, non_blocking=True) for bag in bag_batch]


def _resolve_mil_aggregation(config: dict[str, Any] | None = None) -> str:
    mil_config = {} if config is None else config.get("mil", {})
    aggregation = str(mil_config.get("aggregate_logits", "mean")).lower()
    if aggregation != "mean":
        raise ValueError(f"Unsupported mil.aggregate_logits: {aggregation}. Currently supported: mean")
    return aggregation


def _summarize_counts(counts: list[int]) -> dict[str, float | int]:
    if not counts:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {
        "min": int(min(counts)),
        "max": int(max(counts)),
        "mean": float(sum(counts) / len(counts)),
    }


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
) -> dict[str, Any]:
    """Run evaluation on the test set and save metrics/artifacts."""

    model.eval()
    task_mode = _get_task_mode(evaluation_config or {})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    running_loss = 0.0
    predictions: list[int] = []
    targets: list[int] = []
    probabilities: list[list[float]] = []
    source_images: list[str] = []
    patch_paths: list[str] = []
    bag_logits: list[list[float]] = []
    bag_targets: list[int] = []
    bag_source_images: list[str] = []
    bag_indices: list[int] = []

    progress = tqdm(dataloader, desc="Test", leave=False)
    for batch in progress:
        if task_mode == "mil":
            bag_images = _move_bag_batch_to_device(batch["images"], device)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(bag_images, return_attention=False)
                bag_loss = criterion(logits, labels)

            bag_logits.extend(logits.detach().float().cpu().tolist())
            bag_targets.extend(labels.detach().cpu().tolist())
            bag_source_images.extend([str(value) for value in batch["source_image"]])
            bag_indices.extend(batch["bag_index"].detach().cpu().tolist())
            progress.set_postfix(loss=f"{bag_loss.detach().item():.4f}")
            continue

        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        batch_size = images.size(0)
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

    if task_mode == "mil":
        aggregation = _resolve_mil_aggregation(evaluation_config)
        aggregated = aggregate_bag_logits_to_image(
            logits=bag_logits,
            targets=bag_targets,
            source_images=bag_source_images,
            bag_indices=bag_indices,
        )

        if aggregated["logits"]:
            image_logits_tensor = torch.tensor(aggregated["logits"], dtype=torch.float32, device=device)
            image_labels_tensor = torch.tensor(aggregated["targets"], dtype=torch.long, device=device)
            image_loss = float(criterion(image_logits_tensor, image_labels_tensor).item())
            image_probabilities_tensor = torch.softmax(image_logits_tensor, dim=1)
            image_predictions = torch.argmax(image_probabilities_tensor, dim=1).cpu().tolist()
            image_probabilities = image_probabilities_tensor.cpu().tolist()
        else:
            image_loss = 0.0
            image_predictions = []
            image_probabilities = []

        image_metrics = compute_classification_metrics(
            targets=aggregated["targets"],
            predictions=image_predictions,
            probabilities=image_probabilities,
            label_names=label_names,
        )
        image_metrics["num_samples"] = int(len(aggregated["targets"]))
        image_metrics["sample_level"] = "image"
        image_metrics["aggregation"] = f"{aggregation}_logits"
        image_metrics["num_bags"] = int(len(bag_targets))
        image_metrics["bags_per_image"] = _summarize_counts(aggregated["bag_counts"])

        metrics = build_single_level_evaluation_result(
            metrics=image_metrics,
            loss=float(image_loss),
            sample_level="image",
            aggregation=f"{aggregation}_logits",
        )
        metrics["num_bags"] = int(len(bag_targets))
        metrics["bags_per_image"] = image_metrics["bags_per_image"]
    else:
        test_loss = running_loss / max(1, len(dataloader.dataset))
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
    task_mode = _get_task_mode(config)
    device = _select_device(config)
    use_amp = bool(config.get("train", {}).get("mixed_precision", True) and device.type == "cuda")

    split_dir = _resolve_path(project_root, config.get("build_patch_splits", {}).get("output_dir", "data/splits"))
    label_mapping_path = _resolve_path(
        project_root,
        config.get("build_patch_splits", {}).get("label_mapping_path", "data/splits/label_mapping.json"),
    )

    train_csv = split_dir / "train.csv"
    test_csv = split_dir / "test.csv"

    if task_mode == "mil":
        test_loader = build_mil_dataloader(
            csv_file=test_csv,
            mode="test",
            config=config,
            label_mapping_path=label_mapping_path,
            project_root=project_root,
            shuffle=False,
        )
    else:
        test_loader = build_dataloader(
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
        task_mode=task_mode,
        num_classes=config.get("data", {}).get("num_classes"),
        label_names=label_names,
    )
    criterion = build_loss(config, class_counts=class_counts, device=device)

    logging.info("Testing checkpoint: %s", checkpoint_path)
    if task_mode == "mil" and hasattr(test_loader.dataset, "get_statistics"):
        stats = test_loader.dataset.get_statistics()
        logging.info(
            "Test bags: %s | Test source_images: %s | Avg bags/image: %.2f | Avg instances/bag: %.2f | Min/Max instances: %s/%s",
            stats["num_bags"],
            stats["num_source_images"],
            stats["avg_bags_per_image"],
            stats["avg_instances_per_bag"],
            stats["min_instances_per_bag"],
            stats["max_instances_per_bag"],
        )
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
    )
