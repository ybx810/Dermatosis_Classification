from __future__ import annotations

from typing import Any

import torch
from sklearn.metrics import f1_score
from tqdm import tqdm


@torch.no_grad()
def _compute_epoch_metrics(predictions: list[int], targets: list[int]) -> dict[str, float]:
    if not targets:
        return {"accuracy": 0.0, "macro_f1": 0.0}

    correct = sum(int(pred == target) for pred, target in zip(predictions, targets))
    accuracy = correct / len(targets)
    macro_f1 = f1_score(targets, predictions, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
    }


def _get_task_mode(task_mode: str | None = None, config: dict[str, Any] | None = None) -> str:
    if task_mode is not None:
        mode = str(task_mode).lower()
    elif config is not None:
        task_config = config.get("task", {})
        mode = str(task_config.get("mode", "patch")).lower() if isinstance(task_config, dict) else "patch"
    else:
        mode = "patch"

    if mode not in {"patch", "mil"}:
        raise ValueError(f"Unsupported task mode: {mode}. Expected one of ['patch', 'mil']")
    return mode


def _move_bag_batch_to_device(bag_batch: list[torch.Tensor], device: torch.device) -> list[torch.Tensor]:
    return [bag.to(device, non_blocking=True) for bag in bag_batch]


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    task_mode: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, float]:
    model.train()

    mode = _get_task_mode(task_mode=task_mode, config=config)
    running_loss = 0.0
    predictions: list[int] = []
    targets: list[int] = []

    progress = tqdm(dataloader, desc=f"Train {epoch:03d}", leave=False)
    for batch in progress:
        optimizer.zero_grad(set_to_none=True)

        if mode == "mil":
            bag_images = _move_bag_batch_to_device(batch["images"], device)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(bag_images, return_attention=False)
                loss = criterion(logits, labels)

            batch_size = labels.size(0)
            batch_predictions = torch.argmax(logits.detach(), dim=1)
            predictions.extend(batch_predictions.cpu().tolist())
            targets.extend(labels.detach().cpu().tolist())
        else:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            batch_size = images.size(0)
            batch_predictions = torch.argmax(logits.detach(), dim=1)
            predictions.extend(batch_predictions.cpu().tolist())
            targets.extend(labels.detach().cpu().tolist())

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += float(loss.detach().item()) * batch_size
        progress.set_postfix(loss=f"{loss.detach().item():.4f}")

    epoch_loss = running_loss / max(1, len(dataloader.dataset))
    metrics = _compute_epoch_metrics(predictions, targets)
    metrics["loss"] = float(epoch_loss)
    return metrics
