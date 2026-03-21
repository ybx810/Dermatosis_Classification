from __future__ import annotations

from typing import Any

import torch
from tqdm import tqdm

from src.utils.metrics import (
    aggregate_bag_logits_to_image,
    build_single_level_evaluation_result,
    compute_classification_metrics,
    compute_multilevel_classification_metrics,
)


def _get_task_mode(config: dict[str, Any] | None = None) -> str:
    task_config = {} if config is None else config.get("task", {})
    mode = str(task_config.get("mode", "patch")).lower() if isinstance(task_config, dict) else "patch"
    if mode not in {"patch", "mil"}:
        raise ValueError(f"Unsupported task mode: {mode}. Expected one of ['patch', 'mil']")
    return mode


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
def validate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    epoch: int,
    use_amp: bool = False,
    evaluation_config: dict[str, Any] | None = None,
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    model.eval()

    task_mode = _get_task_mode(evaluation_config)
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

    progress = tqdm(dataloader, desc=f"Valid {epoch:03d}", leave=False)
    for batch in progress:
        if task_mode == "mil":
            bag_images = _move_bag_batch_to_device(batch["images"], device)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(bag_images, return_attention=False)
                bag_loss = criterion(logits, labels)

            batch_size = labels.size(0)
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

        result = build_single_level_evaluation_result(
            metrics=image_metrics,
            loss=float(image_loss),
            sample_level="image",
            aggregation=f"{aggregation}_logits",
        )
        result["num_bags"] = int(len(bag_targets))
        result["bags_per_image"] = image_metrics["bags_per_image"]
        return result

    epoch_loss = running_loss / max(1, len(dataloader.dataset))
    return compute_multilevel_classification_metrics(
        targets=targets,
        predictions=predictions,
        probabilities=probabilities,
        label_names=label_names,
        source_images=source_images if source_images else None,
        patch_paths=patch_paths if patch_paths else None,
        loss=float(epoch_loss),
        evaluation_config=evaluation_config,
    )
