from __future__ import annotations

import json
from pathlib import Path
from typing import Any



def _get_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for training-curve export. Install it with `pip install matplotlib`."
        ) from exc

    plt.style.use("seaborn-v0_8-whitegrid")
    return plt



def _load_history(history: list[dict[str, Any]] | str | Path) -> list[dict[str, Any]]:
    if isinstance(history, list):
        return history

    history_path = Path(history)
    if not history_path.exists():
        return []

    payload = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return payload



def _prepare_output_path(output_dir: str | Path, filename: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename



def plot_loss_curves(
    history: list[dict[str, Any]] | str | Path,
    output_dir: str | Path,
    filename: str = "learning_curve_loss.png",
) -> Path | None:
    records = _load_history(history)
    if not records:
        return None

    plt = _get_pyplot()
    epochs = [record.get("epoch") for record in records]
    train_loss = [record.get("train_loss") for record in records]
    val_loss = [record.get("val_loss") for record in records]

    output_path = _prepare_output_path(output_dir, filename)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    ax.plot(epochs, train_loss, label="Train Loss", linewidth=2.2, color="#1f77b4")
    ax.plot(epochs, val_loss, label="Val Loss", linewidth=2.2, color="#d62728")
    ax.set_title("Training and Validation Loss", fontsize=14)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.legend(frameon=True)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path



def plot_metric_curves(
    history: list[dict[str, Any]] | str | Path,
    output_dir: str | Path,
    filename: str = "learning_curve_metric.png",
) -> Path | None:
    records = _load_history(history)
    if not records:
        return None

    plt = _get_pyplot()
    epochs = [record.get("epoch") for record in records]
    train_accuracy = [record.get("train_accuracy") for record in records]
    val_accuracy = [record.get("val_accuracy") for record in records]
    train_macro_f1 = [record.get("train_macro_f1") for record in records]
    val_macro_f1 = [record.get("val_macro_f1") for record in records]

    output_path = _prepare_output_path(output_dir, filename)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    ax.plot(epochs, train_accuracy, label="Train Accuracy", linewidth=2.2, color="#1f77b4")
    ax.plot(epochs, val_accuracy, label="Val Accuracy", linewidth=2.2, color="#17becf")
    ax.plot(epochs, train_macro_f1, label="Train Macro F1", linewidth=2.2, color="#ff7f0e")
    ax.plot(epochs, val_macro_f1, label="Val Macro F1", linewidth=2.2, color="#2ca02c")
    ax.set_title("Accuracy and Macro F1 Curves", fontsize=14)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=True, ncol=2)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path



def export_training_visualizations(
    history: list[dict[str, Any]] | str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    records = _load_history(history)
    if not records:
        return {}

    outputs: dict[str, str] = {}
    loss_path = plot_loss_curves(records, output_dir)
    metric_path = plot_metric_curves(records, output_dir)

    if loss_path is not None:
        outputs["loss_curve"] = str(loss_path)
    if metric_path is not None:
        outputs["metric_curve"] = str(metric_path)
    return outputs
