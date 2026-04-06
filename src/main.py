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
from src.models import (
    build_model,
    get_backbone_modules,
    get_backbone_parameters,
    get_classifier_module,
    get_classifier_parameters,
    set_backbone_trainable,
)
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


def save_config_snapshot(config: dict[str, Any], run_dir: Path) -> None:
    config_path = run_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def select_device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("train", {}).get("device", "cuda")).lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _to_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return float(value)


def _resolve_optimizer_options(config: dict[str, Any]) -> dict[str, float | str | bool]:
    optimizer_config = config.get("optimizer", {})
    name = str(optimizer_config.get("name", "adam")).lower()
    lr = float(optimizer_config.get("lr", 3e-4))
    weight_decay = float(optimizer_config.get("weight_decay", 0.0))
    differential_lr = bool(optimizer_config.get("differential_lr", False))

    classifier_lr = _to_optional_float(optimizer_config.get("classifier_lr"))
    if classifier_lr is None:
        classifier_lr = lr

    backbone_lr = _to_optional_float(optimizer_config.get("backbone_lr"))
    if backbone_lr is None:
        backbone_lr = lr * 0.1

    backbone_weight_decay = _to_optional_float(optimizer_config.get("backbone_weight_decay"))
    if backbone_weight_decay is None:
        backbone_weight_decay = weight_decay

    classifier_weight_decay = _to_optional_float(optimizer_config.get("classifier_weight_decay"))
    if classifier_weight_decay is None:
        classifier_weight_decay = weight_decay

    momentum = float(optimizer_config.get("momentum", 0.9))

    return {
        "name": name,
        "lr": lr,
        "weight_decay": weight_decay,
        "differential_lr": differential_lr,
        "backbone_lr": backbone_lr,
        "classifier_lr": classifier_lr,
        "backbone_weight_decay": backbone_weight_decay,
        "classifier_weight_decay": classifier_weight_decay,
        "momentum": momentum,
    }


def _build_optimizer_with_param_groups(
    optimizer_options: dict[str, float | str | bool],
    param_groups: list[dict[str, Any]],
) -> torch.optim.Optimizer:
    name = str(optimizer_options["name"])
    momentum = float(optimizer_options["momentum"])

    if name == "adam":
        return torch.optim.Adam(param_groups)
    if name == "adamw":
        return torch.optim.AdamW(param_groups)
    if name == "sgd":
        return torch.optim.SGD(param_groups, momentum=momentum)

    raise ValueError(f"Unsupported optimizer: {name}")


def build_optimizer(
    config: dict[str, Any],
    model: torch.nn.Module,
    backbone_name: str,
    backbone_trainable: bool,
) -> torch.optim.Optimizer:
    optimizer_options = _resolve_optimizer_options(config)
    differential_lr = bool(optimizer_options["differential_lr"])

    if not backbone_trainable:
        classifier_parameters = get_classifier_parameters(model, backbone_name)
        head_lr = float(optimizer_options["classifier_lr"]) if differential_lr else float(optimizer_options["lr"])
        head_weight_decay = (
            float(optimizer_options["classifier_weight_decay"])
            if differential_lr
            else float(optimizer_options["weight_decay"])
        )
        param_groups = [
            {
                "params": classifier_parameters,
                "lr": head_lr,
                "weight_decay": head_weight_decay,
                "group_name": "classifier",
            }
        ]
        return _build_optimizer_with_param_groups(optimizer_options, param_groups)

    if differential_lr:
        param_groups = [
            {
                "params": get_backbone_parameters(model, backbone_name),
                "lr": float(optimizer_options["backbone_lr"]),
                "weight_decay": float(optimizer_options["backbone_weight_decay"]),
                "group_name": "backbone",
            },
            {
                "params": get_classifier_parameters(model, backbone_name),
                "lr": float(optimizer_options["classifier_lr"]),
                "weight_decay": float(optimizer_options["classifier_weight_decay"]),
                "group_name": "classifier",
            },
        ]
    else:
        param_groups = [
            {
                "params": list(model.parameters()),
                "lr": float(optimizer_options["lr"]),
                "weight_decay": float(optimizer_options["weight_decay"]),
                "group_name": "all",
            }
        ]

    return _build_optimizer_with_param_groups(optimizer_options, param_groups)


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


def _rebuild_scheduler_with_previous_state(
    config: dict[str, Any],
    previous_scheduler: torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
) -> torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    if previous_scheduler is None:
        return None

    rebuilt_scheduler = build_scheduler(config, optimizer, num_epochs)
    if rebuilt_scheduler is None:
        return None

    try:
        rebuilt_scheduler.load_state_dict(previous_scheduler.state_dict())
    except Exception as error:  # pragma: no cover - defensive branch for scheduler internals.
        logging.warning(
            "Failed to restore scheduler state after optimizer rebuild: %s. Using a fresh scheduler instead.",
            error,
        )

    return rebuilt_scheduler


def _collect_optimizer_param_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    parameter_ids: set[int] = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            parameter_ids.add(id(parameter))
    return parameter_ids


def _add_backbone_group_to_optimizer(
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    backbone_name: str,
) -> None:
    optimizer_options = _resolve_optimizer_options(config)
    differential_lr = bool(optimizer_options["differential_lr"])
    existing_parameter_ids = _collect_optimizer_param_ids(optimizer)
    backbone_parameters = [
        parameter
        for parameter in get_backbone_parameters(model, backbone_name)
        if id(parameter) not in existing_parameter_ids
    ]
    if not backbone_parameters:
        return

    backbone_lr = float(optimizer_options["backbone_lr"]) if differential_lr else float(optimizer_options["lr"])
    backbone_weight_decay = (
        float(optimizer_options["backbone_weight_decay"])
        if differential_lr
        else float(optimizer_options["weight_decay"])
    )
    optimizer.add_param_group(
        {
            "params": backbone_parameters,
            "lr": backbone_lr,
            "weight_decay": backbone_weight_decay,
            "group_name": "backbone",
        }
    )


def _resolve_finetune_config(config: dict[str, Any]) -> dict[str, Any]:
    finetune_config = config.get("finetune", {}) or {}
    freeze_backbone_epochs = int(finetune_config.get("freeze_backbone_epochs", 0))
    if freeze_backbone_epochs < 0:
        raise ValueError("finetune.freeze_backbone_epochs must be >= 0")

    unfreeze_strategy = str(finetune_config.get("unfreeze_strategy", "all")).lower()
    if unfreeze_strategy != "all":
        raise ValueError("Only finetune.unfreeze_strategy=all is supported in current training pipeline.")

    return {
        "enabled": bool(finetune_config.get("enabled", False)),
        "freeze_backbone_epochs": freeze_backbone_epochs,
        "unfreeze_strategy": unfreeze_strategy,
        "reinit_optimizer_on_unfreeze": bool(finetune_config.get("reinit_optimizer_on_unfreeze", True)),
        "reset_scheduler_on_unfreeze": bool(finetune_config.get("reset_scheduler_on_unfreeze", True)),
        "train_bn_when_frozen": bool(finetune_config.get("train_bn_when_frozen", False)),
    }


def configure_model_train_mode(
    model: torch.nn.Module,
    backbone_name: str,
    backbone_trainable: bool,
    train_bn_when_frozen: bool,
) -> None:
    model.train()
    get_classifier_module(model, backbone_name).train()

    for backbone_module in get_backbone_modules(model, backbone_name):
        if backbone_trainable or train_bn_when_frozen:
            backbone_module.train()
        else:
            backbone_module.eval()


def _find_param_group_lr(optimizer: torch.optim.Optimizer, group_name: str) -> float | None:
    for param_group in optimizer.param_groups:
        if param_group.get("group_name") == group_name:
            return float(param_group["lr"])
    return None


def _resolve_stage_learning_rates(
    optimizer: torch.optim.Optimizer,
    backbone_trainable: bool,
) -> tuple[float | None, float]:
    default_lr = float(optimizer.param_groups[0]["lr"])
    classifier_lr = _find_param_group_lr(optimizer, "classifier")
    if classifier_lr is None:
        classifier_lr = default_lr

    backbone_lr = _find_param_group_lr(optimizer, "backbone")
    if backbone_lr is None:
        backbone_lr = default_lr if backbone_trainable else None

    return backbone_lr, classifier_lr


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


def run_training(config: dict[str, Any]) -> Path:
    _validate_task_mode(config)
    primary_metric = _resolve_primary_metric(config)

    seed = int(config.get("train", {}).get("seed", 42))
    seed_everything(seed)

    run_dir = build_run_dir(config)
    setup_logging(run_dir / "train.log")
    save_config_snapshot(config, run_dir)

    device = select_device(config)
    use_amp = bool(config.get("train", {}).get("mixed_precision", True) and device.type == "cuda")
    whole_image_config = config.get("whole_image", {})
    cache_config = whole_image_config.get("cache", {})
    logging.info("Run directory: %s", run_dir)
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

    num_epochs = int(config.get("train", {}).get("epochs", 20))
    backbone_name = str(config.get("model", {}).get("name", "resnet18")).lower()
    finetune_config = _resolve_finetune_config(config)
    optimizer_options = _resolve_optimizer_options(config)
    use_frozen_head_stage = bool(finetune_config["enabled"]) and int(finetune_config["freeze_backbone_epochs"]) > 0
    backbone_trainable = not use_frozen_head_stage
    backbone_unfrozen = not use_frozen_head_stage

    model = build_model(config).to(device)
    set_backbone_trainable(
        model,
        backbone_name,
        trainable=backbone_trainable,
        train_bn_when_frozen=bool(finetune_config["train_bn_when_frozen"]),
    )

    optimizer = build_optimizer(
        config=config,
        model=model,
        backbone_name=backbone_name,
        backbone_trainable=backbone_trainable,
    )
    scheduler = build_scheduler(config, optimizer, num_epochs=num_epochs)
    class_counts = compute_class_counts(train_csv, num_classes=num_classes, label_names=label_names)
    criterion = build_loss(config, class_counts=class_counts, device=device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    whole_image_config = config.get("whole_image", {})
    cache_config = whole_image_config.get("cache", {})
    logging.info("Model: %s", backbone_name)
    logging.info("Loss: %s", config.get("loss", {}).get("name", "cross_entropy"))
    logging.info(
        "Optimizer config | name=%s differential_lr=%s lr=%.6f backbone_lr=%.6f classifier_lr=%.6f",
        optimizer_options["name"],
        optimizer_options["differential_lr"],
        float(optimizer_options["lr"]),
        float(optimizer_options["backbone_lr"]),
        float(optimizer_options["classifier_lr"]),
    )
    logging.info(
        "Finetune config | enabled=%s freeze_backbone_epochs=%s unfreeze_strategy=%s reinit_optimizer_on_unfreeze=%s reset_scheduler_on_unfreeze=%s train_bn_when_frozen=%s",
        finetune_config["enabled"],
        finetune_config["freeze_backbone_epochs"],
        finetune_config["unfreeze_strategy"],
        finetune_config["reinit_optimizer_on_unfreeze"],
        finetune_config["reset_scheduler_on_unfreeze"],
        finetune_config["train_bn_when_frozen"],
    )
    logging.info(
        "Whole image config | image_size=%s cache_size=%s interpolation=%s cache_enabled=%s cached_inputs=%s",
        whole_image_config.get("image_size", 512),
        cache_config.get("size"),
        whole_image_config.get("interpolation", "area"),
        cache_config.get("enabled", False),
        cache_config.get("use_cached_for_training", True),
    )
    logging.info("Train images: %s | Val images: %s", len(train_loader.dataset), len(val_loader.dataset))

    history: list[dict[str, float | int | str | bool | None]] = []
    best_metric = float("-inf")

    for epoch in range(1, num_epochs + 1):
        should_unfreeze_now = (
            use_frozen_head_stage
            and not backbone_trainable
            and epoch == int(finetune_config["freeze_backbone_epochs"]) + 1
        )
        if should_unfreeze_now:
            backbone_trainable = True
            backbone_unfrozen = True
            set_backbone_trainable(
                model,
                backbone_name,
                trainable=True,
                train_bn_when_frozen=bool(finetune_config["train_bn_when_frozen"]),
            )

            if bool(finetune_config["reinit_optimizer_on_unfreeze"]):
                previous_scheduler = scheduler
                optimizer = build_optimizer(
                    config=config,
                    model=model,
                    backbone_name=backbone_name,
                    backbone_trainable=True,
                )
                if bool(finetune_config["reset_scheduler_on_unfreeze"]):
                    scheduler = build_scheduler(config, optimizer, num_epochs=num_epochs)
                else:
                    scheduler = _rebuild_scheduler_with_previous_state(
                        config=config,
                        previous_scheduler=previous_scheduler,
                        optimizer=optimizer,
                        num_epochs=num_epochs,
                    )
            else:
                _add_backbone_group_to_optimizer(
                    config=config,
                    optimizer=optimizer,
                    model=model,
                    backbone_name=backbone_name,
                )
                if bool(finetune_config["reset_scheduler_on_unfreeze"]):
                    scheduler = build_scheduler(config, optimizer, num_epochs=num_epochs)

            logging.info("Backbone unfrozen at epoch %03d", epoch)

        stage = "full_finetune" if backbone_trainable else "frozen_head_only"

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scaler=scaler,
            use_amp=use_amp,
            train_mode_configurer=lambda current_model: configure_model_train_mode(
                model=current_model,
                backbone_name=backbone_name,
                backbone_trainable=backbone_trainable,
                train_bn_when_frozen=bool(finetune_config["train_bn_when_frozen"]),
            ),
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

        backbone_lr, classifier_lr = _resolve_stage_learning_rates(optimizer, backbone_trainable=backbone_trainable)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        epoch_record: dict[str, Any] = {
            "epoch": epoch,
            "lr": learning_rate,
            "stage": stage,
            "backbone_lr": backbone_lr,
            "classifier_lr": classifier_lr,
            "backbone_trainable": bool(backbone_trainable),
            "backbone_unfrozen": bool(backbone_unfrozen),
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
            "Epoch %03d | stage=%s backbone_trainable=%s unfrozen=%s | train_loss=%.4f train_acc=%.4f train_f1=%.4f | val_loss=%.4f val_acc=%.4f val_f1=%.4f val_auc=%s | backbone_lr=%s classifier_lr=%.6f",
            epoch,
            stage,
            backbone_trainable,
            backbone_unfrozen,
            train_metrics["loss"],
            train_metrics["accuracy"],
            train_metrics["macro_f1"],
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["macro_f1"],
            "N/A" if val_metrics.get("auc_ovr") is None else f"{val_metrics['auc_ovr']:.4f}",
            "N/A" if backbone_lr is None else f"{backbone_lr:.6f}",
            classifier_lr,
        )

        save_checkpoint(run_dir / "last_model.pth", model, optimizer, epoch, epoch_record, config)
        current_metric = val_metrics.get(primary_metric)
        if current_metric is None:
            current_metric = val_metrics["macro_f1"]
        if float(current_metric) >= best_metric:
            best_metric = float(current_metric)
            save_checkpoint(run_dir / "best_model.pth", model, optimizer, epoch, epoch_record, config)
            logging.info("Updated best model at epoch %03d with %s=%.4f", epoch, primary_metric, best_metric)

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
