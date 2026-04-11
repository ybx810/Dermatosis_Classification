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

from src.datasets import build_whole_image_dataloader
from src.engine import train_one_epoch, validate
from src.losses import build_loss
from src.models import build_model
from src.utils.io import load_yaml
from src.utils.seed import seed_everything
from src.utils.visualize import export_training_visualizations

SUPPORTED_TASK_MODE = "whole_image"
SUPPORTED_PRIMARY_METRICS = {"accuracy", "precision", "recall", "macro_f1", "auc_ovr"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a whole-image medical image classifier.")
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
    project_name = str(config.get("project", {}).get("name", "whole-image-classification"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = resolve_path(config.get("project", {}).get("output_dir", "outputs"))
    run_dir = output_root / project_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_run_dir(
    config: dict[str, Any],
    run_dir: str | Path | None = None,
    run_name: str | None = None,
) -> Path:
    base_run_dir = resolve_path(run_dir) if run_dir is not None else build_run_dir(config)
    final_run_dir = base_run_dir / run_name if run_name else base_run_dir
    final_run_dir.mkdir(parents=True, exist_ok=True)
    return final_run_dir


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


def _validate_task_mode(config: dict[str, Any]) -> None:
    task_mode = str(config.get("task", {}).get("mode", SUPPORTED_TASK_MODE)).lower()
    if task_mode != SUPPORTED_TASK_MODE:
        raise ValueError("This project only supports task.mode=whole_image.")


def _resolve_primary_metric(config: dict[str, Any]) -> str:
    primary_metric = str(config.get("evaluation", {}).get("primary_metric", "macro_f1")).lower()
    if primary_metric not in SUPPORTED_PRIMARY_METRICS:
        raise ValueError(
            f"Unsupported evaluation.primary_metric: {primary_metric}. "
            f"Expected one of {sorted(SUPPORTED_PRIMARY_METRICS)}."
        )
    return primary_metric


def compute_class_counts(
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



def _log_train_sampling_summary(train_loader: Any) -> dict[str, Any]:
    summary = getattr(train_loader, "sampling_summary", {}) or {}
    strategy = str(summary.get("strategy", "none")).lower()
    target_epoch_size = summary.get("target_epoch_size")
    replacement = summary.get("replacement")
    class_counts = summary.get("class_counts", {})

    logging.info("Train sampling summary | strategy=%s", strategy)
    if target_epoch_size is not None:
        logging.info(
            "Train sampling summary | target_epoch_size=%s replacement=%s",
            target_epoch_size,
            replacement,
        )
    if summary.get("log_distribution", True) and class_counts:
        logging.info("Train sampling summary | original_class_counts=%s", class_counts)

    for warning in summary.get("warnings", []):
        logging.warning("Train sampling summary | %s", warning)
    return summary


def _warn_double_rebalancing(config: dict[str, Any], sampling_summary: dict[str, Any]) -> None:
    strategy = str(sampling_summary.get("strategy", "none")).lower()
    if not bool(sampling_summary.get("enabled", False)) or strategy == "none":
        return

    loss_config = config.get("loss", {})
    use_class_weights = bool(loss_config.get("use_class_weights", False))
    use_focal_alpha = bool(loss_config.get("use_alpha_from_class_counts", False))
    if not (use_class_weights or use_focal_alpha):
        return

    logging.warning(
        "Detected potential double class rebalancing: train.sampling.strategy=%s with "
        "loss.use_class_weights=%s and loss.use_alpha_from_class_counts=%s. "
        "This may over-emphasize minority classes.",
        strategy,
        use_class_weights,
        use_focal_alpha,
    )


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


def _resolve_split_dir(config: dict[str, Any]) -> Path:
    return resolve_path(
        config.get("data", {}).get("split_dir") or config.get("build_image_splits", {}).get("output_dir") or "data/splits"
    )


def _resolve_label_mapping_path(config: dict[str, Any], split_dir: Path) -> Path:
    return resolve_path(config.get("build_image_splits", {}).get("label_mapping_path") or split_dir / "label_mapping.json")


def _resolve_split_csvs(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    split_dir = _resolve_split_dir(config)
    label_mapping_path = _resolve_label_mapping_path(config, split_dir)
    whole_image_config = config.get("whole_image", {})
    train_csv = resolve_path(whole_image_config.get("train_csv") or split_dir / "train_images.csv")
    val_csv = resolve_path(whole_image_config.get("val_csv") or split_dir / "val_images.csv")
    return train_csv, val_csv, label_mapping_path


def run_training(
    config: dict[str, Any],
    run_dir: str | Path | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    _validate_task_mode(config)
    primary_metric = _resolve_primary_metric(config)

    seed = int(config.get("train", {}).get("seed", 42))
    seed_everything(seed)

    resolved_run_dir = resolve_run_dir(config, run_dir=run_dir, run_name=run_name)
    setup_logging(resolved_run_dir / "train.log")
    save_config_snapshot(config, resolved_run_dir)

    device = select_device(config)
    use_amp = bool(config.get("train", {}).get("mixed_precision", True) and device.type == "cuda")
    whole_image_config = config.get("whole_image", {})
    cache_config = whole_image_config.get("cache", {})
    logging.info("Run directory: %s", resolved_run_dir)
    logging.info("Device: %s", device)
    logging.info("AMP enabled: %s", use_amp)
    logging.info("Task mode: %s", SUPPORTED_TASK_MODE)
    logging.info("Primary validation metric: %s", primary_metric)
    logging.info(
        "Whole-image config | image_size=%s interpolation=%s cache_enabled=%s use_cached_for_training=%s allow_raw_fallback=%s",
        whole_image_config.get("image_size", 512),
        whole_image_config.get("interpolation", "area"),
        bool(cache_config.get("enabled", False)),
        bool(cache_config.get("use_cached_for_training", True)),
        bool(cache_config.get("allow_raw_fallback", False)),
    )

    train_csv, val_csv, label_mapping_path = _resolve_split_csvs(config)
    label_names = load_label_names(label_mapping_path, num_classes=config.get("data", {}).get("num_classes"))
    num_classes = len(label_names) if label_names is not None else config.get("data", {}).get("num_classes")

    train_loader = build_whole_image_dataloader(
        csv_file=train_csv,
        mode="train",
        config=config,
        label_mapping_path=label_mapping_path,
        project_root=PROJECT_ROOT,
    )
    val_loader = build_whole_image_dataloader(
        csv_file=val_csv,
        mode="val",
        config=config,
        label_mapping_path=label_mapping_path,
        project_root=PROJECT_ROOT,
        shuffle=False,
    )

    train_sampling_summary = _log_train_sampling_summary(train_loader)
    _warn_double_rebalancing(config, train_sampling_summary)

    model = build_model(config).to(device)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer, num_epochs=int(config.get("train", {}).get("epochs", 20)))
    class_counts = compute_class_counts(train_csv, num_classes=num_classes, label_names=label_names)
    criterion = build_loss(config, class_counts=class_counts, device=device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    whole_image_config = config.get("whole_image", {})
    cache_config = whole_image_config.get("cache", {})
    logging.info("Model: %s", config.get("model", {}).get("name", "resnet18"))
    logging.info("Loss: %s", config.get("loss", {}).get("name", "cross_entropy"))
    logging.info(
        "Whole image config | image_size=%s cache_size=%s interpolation=%s cache_enabled=%s cached_inputs=%s",
        whole_image_config.get("image_size", 512),
        cache_config.get("size"),
        whole_image_config.get("interpolation", "area"),
        cache_config.get("enabled", False),
        cache_config.get("use_cached_for_training", True),
    )
    logging.info("Train images: %s | Val images: %s", len(train_loader.dataset), len(val_loader.dataset))

    num_epochs = int(config.get("train", {}).get("epochs", 20))
    history: list[dict[str, float | int | str | None]] = []
    best_metric = float("-inf")
    best_epoch: int | None = None
    best_metrics: dict[str, Any] | None = None

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
        )
        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            use_amp=use_amp,
            label_names=label_names,
        )

        scheduler_metric = val_metrics.get(primary_metric)
        if scheduler_metric is None:
            scheduler_metric = val_metrics["macro_f1"]
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(float(scheduler_metric))
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
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_auc_ovr": val_metrics.get("auc_ovr"),
        }
        history.append(epoch_record)

        logging.info(
            "Epoch %03d | train_loss=%.4f train_acc=%.4f train_f1=%.4f | val_loss=%.4f val_acc=%.4f val_f1=%.4f val_auc=%s | lr=%.6f",
            epoch,
            train_metrics["loss"],
            train_metrics["accuracy"],
            train_metrics["macro_f1"],
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["macro_f1"],
            "N/A" if val_metrics.get("auc_ovr") is None else f"{val_metrics['auc_ovr']:.4f}",
            learning_rate,
        )

        save_checkpoint(resolved_run_dir / "last_model.pth", model, optimizer, epoch, epoch_record, config)
        current_metric = val_metrics.get(primary_metric)
        if current_metric is None:
            current_metric = val_metrics["macro_f1"]
        if float(current_metric) >= best_metric:
            best_metric = float(current_metric)
            best_epoch = int(epoch)
            best_metrics = {
                "loss": val_metrics.get("loss"),
                "accuracy": val_metrics.get("accuracy"),
                "precision": val_metrics.get("precision"),
                "recall": val_metrics.get("recall"),
                "macro_f1": val_metrics.get("macro_f1"),
                "auc_ovr": val_metrics.get("auc_ovr"),
                "num_samples": val_metrics.get("num_samples"),
                "sample_level": val_metrics.get("sample_level"),
            }
            save_checkpoint(resolved_run_dir / "best_model.pth", model, optimizer, epoch, epoch_record, config)
            logging.info("Updated best model at epoch %03d with %s=%.4f", epoch, primary_metric, best_metric)

    history_path = resolved_run_dir / "history.json"
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Saved training history to %s", history_path)

    visualization_outputs = export_training_visualizations(history_path, resolved_run_dir)
    if visualization_outputs:
        logging.info("Saved loss curve to %s", visualization_outputs.get("loss_curve"))
        logging.info("Saved metric curve to %s", visualization_outputs.get("metric_curve"))
    else:
        logging.warning("Training history is empty. Skipping visualization export.")

    best_model_path = resolved_run_dir / "best_model.pth"
    if best_epoch is None or not best_model_path.exists():
        raise RuntimeError(
            "Training did not produce a best model checkpoint. "
            "Please check train.epochs and validation metric configuration."
        )

    return {
        "run_dir": str(resolved_run_dir),
        "best_model_path": str(best_model_path),
        "best_epoch": int(best_epoch),
        "primary_metric": primary_metric,
        "best_metric_value": float(best_metric),
        "final_history_path": str(history_path),
        "best_metrics": best_metrics or {},
    }


def main() -> None:
    args = parse_args()
    config = load_yaml(resolve_path(args.config))
    run_training(config)


if __name__ == "__main__":
    main()
