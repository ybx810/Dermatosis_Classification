from __future__ import annotations

from typing import Any

import torch
from tqdm import tqdm

from src.utils.metrics import build_single_level_evaluation_result, compute_multilevel_classification_metrics


def _move_patch_images_to_device(
    patch_images: list[torch.Tensor],
    device: torch.device,
) -> list[torch.Tensor]:
    return [patch_tensor.to(device, non_blocking=True) for patch_tensor in patch_images]


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
    task_mode: str = "patch",
) -> dict[str, Any]:
    task_mode = str(task_mode).lower()
    if task_mode not in {"patch", "dual_branch"}:
        raise ValueError(f"Unsupported task_mode: {task_mode}")

    model.eval()

    running_loss = 0.0
    predictions: list[int] = []
    targets: list[int] = []
    probabilities: list[list[float]] = []
    source_images: list[str] = []
    patch_paths: list[str] = []

    progress = tqdm(dataloader, desc=f"Valid {epoch:03d}", leave=False)
    for batch in progress:
        labels = batch["label"].to(device, non_blocking=True)

        if task_mode == "dual_branch":
            whole_images = batch["whole_image"].to(device, non_blocking=True)
            patch_images = _move_patch_images_to_device(batch["patch_images"], device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(whole_images, patch_images)
                loss = criterion(logits, labels)
            batch_size = whole_images.size(0)
        else:
            images = batch["image"].to(device, non_blocking=True)
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
    if task_mode == "dual_branch":
        return build_single_level_evaluation_result(
            targets=targets,
            predictions=predictions,
            probabilities=probabilities,
            label_names=label_names,
            loss=float(epoch_loss),
            sample_level="image",
            aggregation="model_direct",
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
