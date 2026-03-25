from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import build_dataloader, build_dual_branch_dataloader
from src.engine import train_one_epoch, validate
from src.losses import build_loss
from src.models import build_model
from src.utils.io import load_yaml
from src.utils.seed import seed_everything
from src.utils.visualize import export_training_visualizations


SUPPORTED_TASK_MODES = {"patch", "dual_branch"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a medical image classifier.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )


def resolve_path(path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def build_run_dir(config: dict[str, Any]) -> Path:
    project_name = str(config.get("project", {}).get("name", "patch-classification"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = resolve_path(config.get("project", {}).get("output_dir", "outputs"))
    run_dir = output_root / project_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config_snapshot(config: dict[str, Any], run_dir: Path) -> None:
    config_path = run_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def select_device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("train", {}).get("device", "cuda")).lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_optimizer(config: dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    optimizer_config = config.get("optimizer", {})
    name = str(optimizer_config.get("name", "adam")).lower()
    lr = float(optimizer_config.get("lr", 3e-4))
    weight_decay = float(optimizer_config.get("weight_decay", 0.0))

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        momentum = float(optimizer_config.get("momentum", 0.9))
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
) -> torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    scheduler_config = config.get("scheduler", {})
    name = str(scheduler_config.get("name", "none")).lower()

    if name in {"none", "", "null"}:
        return None
    if name == "steplr":
        step_size = int(scheduler_config.get("step_size", max(1, num_epochs // 3)))
        gamma = float(scheduler_config.get("gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if name == "cosine":
        eta_min = float(scheduler_config.get("eta_min", 0.0))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs), eta_min=eta_min)
    if name == "plateau":
        factor = float(scheduler_config.get("factor", 0.1))
        patience = int(scheduler_config.get("patience", 2))
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=factor, patience=patience)

    raise ValueError(f"Unsupported scheduler: {name}")


def _get_task_mode(config: dict[str, Any]) -> str:
    task_mode = str(config.get("task", {}).get("mode", "patch")).lower()
    if task_mode not in SUPPORTED_TASK_MODES:
        raise ValueError(f"Unsupported task.mode: {task_mode}. Expected one of {sorted(SUPPORTED_TASK_MODES)}")
    return task_mode


def compute_patch_class_counts(
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


def compute_image_class_counts(
    train_csv: Path,
    num_classes: int | None = None,
    label_names: list[str] | None = None,
) -> list[int]:
    dataframe = pd.read_csv(train_csv)
    if "source_image" not in dataframe.columns:
        raise ValueError("dual_branch mode requires source_image in the split CSV to compute image-level class counts.")

    grouped_counts: Counter[str] = Counter()
    for source_image, group in dataframe.groupby("source_image", sort=False):
        labels = sorted(group["label"].astype(str).unique().tolist())
        if len(labels) != 1:
            raise ValueError(
                "dual_branch mode expects a single image-level label per source_image, but found "
                f"multiple labels for '{source_image}': {labels}"
            )
        grouped_counts[labels[0]] += 1

    if label_names is not None:
        unknown_labels = sorted(set(grouped_counts).difference(label_names))
        if unknown_labels:
            raise ValueError(f"Found labels in {train_csv} that are missing from label_names: {unknown_labels}")
        return [int(grouped_counts.get(label_name, 0)) for label_name in label_names]

    if num_classes is not None:
        labels = sorted(grouped_counts)
        if len(labels) > num_classes:
            raise ValueError(f"Found {len(labels)} labels in {train_csv}, which exceeds num_classes={num_classes}.")
        return [int(grouped_counts.get(label, 0)) for label in labels]

    labels = sorted(grouped_counts)
    return [int(grouped_counts.get(label, 0)) for label in labels]


def load_label_names(label_mapping_path: Path | None, num_classes: int | None = None) -> list[str] | None:
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


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def _append_validation_metrics(epoch_record: dict[str, Any], val_metrics: dict[str, Any]) -> None:
    epoch_record["val_primary_level"] = val_metrics.get("primary_metric_level", "patch")

    if "patch_accuracy" in val_metrics:
        epoch_record["val_patch_accuracy"] = val_metrics["patch_accuracy"]
        epoch_record["val_patch_macro_f1"] = val_metrics["patch_macro_f1"]
        epoch_record["val_patch_precision"] = val_metrics["patch_precision"]
        epoch_record["val_patch_recall"] = val_metrics["patch_recall"]
        epoch_record["val_patch_auc_ovr"] = val_metrics.get("patch_auc_ovr")

    if "image_accuracy" in val_metrics:
        epoch_record["val_image_accuracy"] = val_metrics["image_accuracy"]
        epoch_record["val_image_macro_f1"] = val_metrics["image_macro_f1"]
        epoch_record["val_image_precision"] = val_metrics["image_precision"]
        epoch_record["val_image_recall"] = val_metrics["image_recall"]
        epoch_record["val_image_auc_ovr"] = val_metrics.get("image_auc_ovr")


def _build_train_and_val_loaders(
    config: dict[str, Any],
    train_csv: Path,
    val_csv: Path,
    label_mapping_path: Path | None,
    task_mode: str,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    if task_mode == "dual_branch":
        train_loader = build_dual_branch_dataloader(
            csv_file=train_csv,
            mode="train",
            config=config,
            label_mapping_path=label_mapping_path,
            project_root=PROJECT_ROOT,
        )
        val_loader = build_dual_branch_dataloader(
            csv_file=val_csv,
            mode="val",
            config=config,
            label_mapping_path=label_mapping_path,
            project_root=PROJECT_ROOT,
            shuffle=False,
        )
        return train_loader, val_loader

    train_loader = build_dataloader(
        csv_file=train_csv,
        mode="train",
        config=config,
        label_mapping_path=label_mapping_path,
        project_root=PROJECT_ROOT,
    )
    val_loader = build_dataloader(
        csv_file=val_csv,
        mode="val",
        config=config,
        label_mapping_path=label_mapping_path,
        project_root=PROJECT_ROOT,
        shuffle=False,
    )
    return train_loader, val_loader


def _log_dual_branch_dataset_stats(split_name: str, dataset: Any) -> None:
    if not hasattr(dataset, "get_statistics"):
        return
    stats = dataset.get_statistics()
    logging.info(
        "%s images=%s total_patches=%s avg_patches_per_image=%.2f min_patches=%s max_patches=%s avg_selected_patches=%.2f min_selected=%s max_selected=%s",
        split_name,
        stats.get("num_images", 0),
        stats.get("total_patches", 0),
        stats.get("avg_patches_per_image", 0.0),
        stats.get("min_patches_per_image", 0),
        stats.get("max_patches_per_image", 0),
        stats.get("avg_selected_patches_per_image", 0.0),
        stats.get("min_selected_patches_per_image", 0),
        stats.get("max_selected_patches_per_image", 0),
    )


def run_training(config: dict[str, Any]) -> Path:
    seed = int(config.get("train", {}).get("seed", 42))
    seed_everything(seed)

    run_dir = build_run_dir(config)
    setup_logging(run_dir / "train.log")
    save_config_snapshot(config, run_dir)

    device = select_device(config)
    use_amp = bool(config.get("train", {}).get("mixed_precision", True) and device.type == "cuda")
    task_mode = _get_task_mode(config)
    logging.info("Run directory: %s", run_dir)
    logging.info("Device: %s", device)
    logging.info("AMP enabled: %s", use_amp)
    logging.info("Task mode: %s", task_mode)

    split_dir = resolve_path(config.get("build_patch_splits", {}).get("output_dir", "data/splits"))
    label_mapping_path = resolve_path(config.get("build_patch_splits", {}).get("label_mapping_path", "data/splits/label_mapping.json"))
    train_csv = split_dir / "train.csv"
    val_csv = split_dir / "val.csv"
    label_names = load_label_names(label_mapping_path, num_classes=config.get("data", {}).get("num_classes"))
    num_classes = len(label_names) if label_names is not None else config.get("data", {}).get("num_classes")

    train_loader, val_loader = _build_train_and_val_loaders(
        config=config,
        train_csv=train_csv,
        val_csv=val_csv,
        label_mapping_path=label_mapping_path,
        task_mode=task_mode,
    )

    model = build_model(config).to(device)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer, num_epochs=int(config.get("train", {}).get("epochs", 20)))
    if task_mode == "dual_branch":
        class_counts = compute_image_class_counts(train_csv, num_classes=num_classes, label_names=label_names)
    else:
        class_counts = compute_patch_class_counts(train_csv, num_classes=num_classes, label_names=label_names)
    criterion = build_loss(config, class_counts=class_counts, device=device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    logging.info("Model: %s", config.get("model", {}).get("name", "resnet18"))
    logging.info("Loss: %s", config.get("loss", {}).get("name", "cross_entropy"))
    if task_mode == "dual_branch":
        dual_branch_config = config.get("dual_branch", {})
        logging.info(
            "Dual branch: whole_backbone=%s patch_backbone=%s fusion=%s patch_pooling=%s max_patches_per_image=%s whole_image_size=%s",
            dual_branch_config.get("whole_backbone", config.get("model", {}).get("name", "resnet18")),
            dual_branch_config.get("patch_backbone", config.get("model", {}).get("name", "resnet18")),
            dual_branch_config.get("fusion", "concat"),
            dual_branch_config.get("patch_pooling", "mean"),
            dual_branch_config.get("max_patches_per_image", "all"),
            dual_branch_config.get("whole_image_size", 512),
        )
        logging.info("Train images: %s | Val images: %s", len(train_loader.dataset), len(val_loader.dataset))
        _log_dual_branch_dataset_stats("Train", train_loader.dataset)
        _log_dual_branch_dataset_stats("Val", val_loader.dataset)
        logging.info("Validation: direct image-level metrics from model logits")
    else:
        logging.info("Train samples: %s | Val samples: %s", len(train_loader.dataset), len(val_loader.dataset))
        logging.info(
            "Evaluation: level=%s aggregation=%s",
            config.get("evaluation", {}).get("level", "both"),
            config.get("evaluation", {}).get("aggregation", "mean_prob"),
        )

    num_epochs = int(config.get("train", {}).get("epochs", 20))
    history: list[dict[str, float | int | str | None]] = []
    best_metric = float("-inf")

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scaler=scaler,
            use_amp=use_amp,
            task_mode=task_mode,
        )
        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            use_amp=use_amp,
            evaluation_config=config,
            label_names=label_names,
            task_mode=task_mode,
        )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["macro_f1"])
            else:
                scheduler.step()

        learning_rate = float(optimizer.param_groups[0]["lr"])
        epoch_record: dict[str, Any] = {
            "epoch": epoch,
            "lr": learning_rate,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        _append_validation_metrics(epoch_record, val_metrics)
        history.append(epoch_record)

        logging.info(
            "Epoch %03d | train_loss=%.4f train_acc=%.4f train_f1=%.4f | val_loss=%.4f val_acc=%.4f val_f1=%.4f | lr=%.6f | val_primary=%s",
            epoch,
            train_metrics["loss"],
            train_metrics["accuracy"],
            train_metrics["macro_f1"],
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["macro_f1"],
            learning_rate,
            val_metrics.get("primary_metric_level", "patch"),
        )
        if "patch_accuracy" in val_metrics and val_metrics.get("primary_metric_level") != "patch":
            logging.info(
                "Epoch %03d | val_patch_acc=%.4f val_patch_f1=%.4f",
                epoch,
                val_metrics["patch_accuracy"],
                val_metrics["patch_macro_f1"],
            )
        if "image_accuracy" in val_metrics:
            logging.info(
                "Epoch %03d | val_image_acc=%.4f val_image_f1=%.4f aggregation=%s",
                epoch,
                val_metrics["image_accuracy"],
                val_metrics["image_macro_f1"],
                val_metrics.get("aggregation", "model_direct"),
            )

        save_checkpoint(run_dir / "last_model.pth", model, optimizer, epoch, epoch_record, config)
        if val_metrics["macro_f1"] >= best_metric:
            best_metric = val_metrics["macro_f1"]
            save_checkpoint(run_dir / "best_model.pth", model, optimizer, epoch, epoch_record, config)
            logging.info(
                "Updated best model at epoch %03d with %s_macro_f1=%.4f",
                epoch,
                val_metrics.get("primary_metric_level", "patch"),
                best_metric,
            )

    history_path = run_dir / "history.json"
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Saved training history to %s", history_path)

    visualization_outputs = export_training_visualizations(history_path, run_dir)
    if visualization_outputs:
        logging.info("Saved loss curve to %s", visualization_outputs.get("loss_curve"))
        logging.info("Saved metric curve to %s", visualization_outputs.get("metric_curve"))
    else:
        logging.warning("Training history is empty. Skipping visualization export.")

    return run_dir


def main() -> None:
    args = parse_args()
    config = load_yaml(resolve_path(args.config))
    run_training(config)


if __name__ == "__main__":
    main()
