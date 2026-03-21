from __future__ import annotations

from typing import Any

import torch
from tqdm import tqdm

from src.utils.metrics import (
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

    progress = tqdm(dataloader, desc=f"Valid {epoch:03d}", leave=False)
    for batch in progress:
        if task_mode == "mil":
            bag_images = _move_bag_batch_to_device(batch["images"], device)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(bag_images, return_attention=False)
                loss = criterion(logits, labels)

            batch_size = labels.size(0)
        else:
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

    epoch_loss = running_loss / max(1, len(dataloader.dataset))
    if task_mode == "mil":
        image_metrics = compute_classification_metrics(
            targets=targets,
            predictions=predictions,
            probabilities=probabilities,
            label_names=label_names,
        )
        image_metrics["num_samples"] = int(len(targets))
        image_metrics["sample_level"] = "image"
        image_metrics["aggregation"] = "attention_mil"
        return build_single_level_evaluation_result(
            metrics=image_metrics,
            loss=float(epoch_loss),
            sample_level="image",
            aggregation="attention_mil",
        )

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
