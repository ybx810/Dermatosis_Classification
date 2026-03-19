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
from src.utils.metrics import (
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



def _compute_class_counts(train_csv: Path) -> list[int]:
    dataframe = pd.read_csv(train_csv)
    if "label_idx" in dataframe.columns:
        label_counts = Counter(dataframe["label_idx"].tolist())
        num_classes = max(label_counts.keys()) + 1 if label_counts else 0
        return [int(label_counts.get(index, 0)) for index in range(num_classes)]

    labels = sorted(dataframe["label"].astype(str).unique().tolist())
    counts = dataframe["label"].astype(str).value_counts().to_dict()
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

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        batch_size = images.size(0)
        running_loss += float(loss.detach().item()) * batch_size
        predictions.extend(preds.cpu().tolist())
        targets.extend(labels.cpu().tolist())
        probabilities.extend(probs.cpu().tolist())

        if "source_image" in batch:
            source_images.extend([str(value) for value in batch["source_image"]])
        if "patch_path" in batch:
            patch_paths.extend([str(value) for value in batch["patch_path"]])

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

    split_dir = _resolve_path(project_root, config.get("build_patch_splits", {}).get("output_dir", "data/splits"))
    label_mapping_path = _resolve_path(
        project_root,
        config.get("build_patch_splits", {}).get("label_mapping_path", "data/splits/label_mapping.json"),
    )

    train_csv = split_dir / "train.csv"
    test_csv = split_dir / "test.csv"

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

    class_counts = _compute_class_counts(train_csv)
    criterion = build_loss(config, class_counts=class_counts, device=device)
    label_names = _load_label_names(label_mapping_path, num_classes=config.get("data", {}).get("num_classes"))

    logging.info("Testing checkpoint: %s", checkpoint_path)
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
