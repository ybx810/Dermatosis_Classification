from __future__ import annotations

from typing import Callable

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


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    train_mode_configurer: Callable[[torch.nn.Module], None] | None = None,
) -> dict[str, float]:
    model.train()
    if train_mode_configurer is not None:
        train_mode_configurer(model)

    running_loss = 0.0
    predictions: list[int] = []
    targets: list[int] = []

    progress = tqdm(dataloader, desc=f"Train {epoch:03d}", leave=False)
    for batch in progress:
        optimizer.zero_grad(set_to_none=True)
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
