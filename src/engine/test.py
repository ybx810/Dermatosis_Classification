from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from src.datasets.skin_patch_dataset import build_dataloader
from src.losses import build_loss
from src.models import build_model
from src.utils.label_merge import (
    apply_label_merge_to_dataframe,
    build_label_merge_mapping,
    get_label_names_from_mapping,
    is_label_merge_enabled,
    update_config_num_classes_from_mapping,
    validate_label_merge_coverage,
)
from src.utils.metrics import (
    aggregate_patch_predictions_to_image,
    compute_multilevel_classification_metrics,
    save_confusion_matrix_figure,
    save_metrics_json,
    save_per_class_metrics_csv,
    save_per_class_metrics_json,
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


def _compute_class_counts(
    train_csv: Path,
    num_classes: int | None = None,
    label_names: list[str] | None = None,
    label_merge_mapping: dict[str, Any] | None = None,
    strict_label_merge: bool = True,
) -> list[int]:
    dataframe = pd.read_csv(train_csv)
    if label_merge_mapping is not None:
        merged_dataframe = apply_label_merge_to_dataframe(
            dataframe,
            mapping=label_merge_mapping,
            strict=strict_label_merge,
        )
        label_counts = Counter(merged_dataframe["merged_label_idx"].astype(int).tolist())
        total_classes = int(num_classes) if num_classes is not None else int(label_merge_mapping["num_classes"])
        return [int(label_counts.get(index, 0)) for index in range(total_classes)]

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
    per_class_metrics = list(metrics.get("per_class_metrics", []))
    per_class_json_path = save_per_class_metrics_json(
        per_class_metrics,
        output_dir / f"{prefix}_per_class_metrics.json",
    )
    per_class_csv_path = save_per_class_metrics_csv(
        per_class_metrics,
        output_dir / f"{prefix}_per_class_metrics.csv",
    )
    logging.info("Saved %s metrics to %s", prefix, metrics_path)
    logging.info("Saved %s confusion matrix to %s", prefix, confusion_path)
    logging.info("Saved %s per-class metrics json to %s", prefix, per_class_json_path)
    logging.info("Saved %s per-class metrics csv to %s", prefix, per_class_csv_path)


def _save_prediction_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def _build_patch_prediction_rows(
    targets: list[int],
    predictions: list[int],
    probabilities: list[list[float]],
    source_images: list[str],
    patch_paths: list[str],
    label_names: list[str] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, target in enumerate(targets):
        prediction = int(predictions[index])
        row: dict[str, Any] = {
            "patch_path": patch_paths[index] if index < len(patch_paths) else "",
            "source_image": source_images[index] if index < len(source_images) else "",
            "target_idx": int(target),
            "target_label": label_names[int(target)] if label_names and int(target) < len(label_names) else str(target),
            "pred_idx": prediction,
            "pred_label": label_names[prediction] if label_names and prediction < len(label_names) else str(prediction),
        }
        for class_index, probability in enumerate(probabilities[index]):
            class_name = label_names[class_index] if label_names and class_index < len(label_names) else str(class_index)
            row[f"prob_{class_index}_{class_name}"] = float(probability)
        rows.append(row)
    return rows


def _build_image_prediction_rows(
    aggregated: dict[str, Any],
    label_names: list[str] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, source_image in enumerate(aggregated["source_images"]):
        target = int(aggregated["targets"][index])
        prediction = int(aggregated["predictions"][index])
        row: dict[str, Any] = {
            "source_image": source_image,
            "patch_count": int(aggregated["patch_counts"][index]),
            "target_idx": target,
            "target_label": label_names[target] if label_names and target < len(label_names) else str(target),
            "pred_idx": prediction,
            "pred_label": label_names[prediction] if label_names and prediction < len(label_names) else str(prediction),
        }
        for class_index, probability in enumerate(aggregated["probabilities"][index]):
            class_name = label_names[class_index] if label_names and class_index < len(label_names) else str(class_index)
            row[f"prob_{class_index}_{class_name}"] = float(probability)
        rows.append(row)
    return rows


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
    artifact_prefix: str | None = None,
    save_predictions: bool = False,
) -> dict[str, Any]:
    """Run evaluation on the test set and save metrics/artifacts."""

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

    suffix = f"_{artifact_prefix}" if artifact_prefix else ""
    metrics_path = save_metrics_json(metrics, output_dir / f"metrics{suffix}.json")
    logging.info("Saved combined test metrics to %s", metrics_path)

    if "patch_metrics" in metrics:
        _save_level_artifacts(metrics["patch_metrics"], output_dir, f"patch{suffix}")
    if "image_metrics" in metrics:
        _save_level_artifacts(metrics["image_metrics"], output_dir, f"image{suffix}")

    if save_predictions:
        patch_rows = _build_patch_prediction_rows(
            targets=targets,
            predictions=predictions,
            probabilities=probabilities,
            source_images=source_images,
            patch_paths=patch_paths,
            label_names=label_names,
        )
        patch_predictions_path = _save_prediction_csv(
            patch_rows,
            output_dir / f"predictions_patch{suffix}.csv",
        )
        logging.info("Saved patch predictions to %s", patch_predictions_path)

        if source_images:
            resolved_config = (evaluation_config or {}).get("evaluation", evaluation_config or {})
            aggregation = str(resolved_config.get("aggregation", "mean_prob")).lower()
            aggregated = aggregate_patch_predictions_to_image(
                targets=targets,
                predictions=predictions,
                probabilities=probabilities,
                source_images=source_images,
                aggregation=aggregation,
                patch_paths=patch_paths if patch_paths else None,
            )
            image_rows = _build_image_prediction_rows(aggregated, label_names=label_names)
            image_predictions_path = _save_prediction_csv(
                image_rows,
                output_dir / f"predictions_image{suffix}.csv",
            )
            logging.info("Saved image predictions to %s", image_predictions_path)

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
    test_csv: str | Path | None = None,
    output_dir: str | Path | None = None,
    artifact_prefix: str | None = None,
    save_predictions: bool = False,
) -> dict[str, Any]:
    """Build the test pipeline from config and evaluate a saved checkpoint."""

    project_root = Path(__file__).resolve().parents[2]
    run_dir = Path(run_dir)
    device = _select_device(config)
    use_amp = bool(config.get("train", {}).get("mixed_precision", True) and device.type == "cuda")

    split_dir = _resolve_path(project_root, config.get("build_patch_splits", {}).get("output_dir", "data/splits"))
    label_mapping_path = _resolve_path(
        project_root,
        config.get("build_patch_splits", {}).get("label_mapping_path", "data/splits/label_mapping.json"),
    )

    train_csv = split_dir / "train.csv"
    if config.get("build_patch_splits", {}).get("train_csv"):
        train_csv = _resolve_path(project_root, config.get("build_patch_splits", {}).get("train_csv"))
    resolved_test_csv = _resolve_path(project_root, test_csv) if test_csv is not None else split_dir / "test.csv"

    label_merge_mapping = None
    label_names = None
    label_merge_config = config.get("label_merge", {}) or {}
    strict_label_merge = bool(label_merge_config.get("strict", True))
    if is_label_merge_enabled(config):
        label_merge_mapping = build_label_merge_mapping(config)
        dataframes = [pd.read_csv(train_csv), pd.read_csv(resolved_test_csv)]
        validate_label_merge_coverage(dataframes, label_merge_mapping, strict=strict_label_merge)
        update_config_num_classes_from_mapping(config, label_merge_mapping)
        label_names = get_label_names_from_mapping(label_merge_mapping)

    test_loader = build_dataloader(
        csv_file=resolved_test_csv,
        mode="test",
        config=config,
        label_mapping_path=label_mapping_path,
        label_merge_mapping=label_merge_mapping,
        use_merged_label=label_merge_mapping is not None,
        strict_label_merge=strict_label_merge,
        project_root=project_root,
        shuffle=False,
    )

    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if label_names is None:
        label_names = _load_label_names(label_mapping_path, num_classes=config.get("data", {}).get("num_classes"))
    class_counts = _compute_class_counts(
        train_csv,
        num_classes=config.get("data", {}).get("num_classes"),
        label_names=label_names,
        label_merge_mapping=label_merge_mapping,
        strict_label_merge=strict_label_merge,
    )
    criterion = build_loss(config, class_counts=class_counts, device=device)

    logging.info("Testing checkpoint: %s", checkpoint_path)
    logging.info("Test split: %s", resolved_test_csv)
    logging.info("Test samples: %s", len(test_loader.dataset))

    return test_model(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        output_dir=Path(output_dir) if output_dir is not None else run_dir / "test",
        label_names=label_names,
        use_amp=use_amp,
        evaluation_config=config,
        artifact_prefix=artifact_prefix,
        save_predictions=save_predictions,
    )
