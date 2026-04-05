from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def _resolve_classification_label_space(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    label_names: list[str] | None = None,
) -> tuple[list[int], list[str]]:
    inferred_num_classes = None
    if y_prob is not None:
        if y_prob.ndim != 2 or y_prob.shape[0] != y_true.shape[0]:
            raise ValueError("probabilities must have shape [num_samples, num_classes].")
        inferred_num_classes = int(y_prob.shape[1])

    if label_names is not None:
        if inferred_num_classes is not None and len(label_names) != inferred_num_classes:
            raise ValueError(
                "label_names length must match the probability dimension: "
                f"{len(label_names)} vs {inferred_num_classes}."
            )
        return list(range(len(label_names))), list(label_names)

    if inferred_num_classes is not None:
        labels = list(range(inferred_num_classes))
        return labels, [str(label) for label in labels]

    labels = sorted({*y_true.tolist(), *y_pred.tolist()})
    return labels, [str(label) for label in labels]


def compute_per_class_metrics(
    targets: list[int] | np.ndarray,
    predictions: list[int] | np.ndarray,
    label_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    y_true = np.asarray(targets, dtype=np.int64)
    y_pred = np.asarray(predictions, dtype=np.int64)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("targets and predictions must have the same length.")

    labels, resolved_label_names = _resolve_classification_label_space(y_true, y_pred, label_names=label_names)
    if not labels:
        return []

    precisions, recalls, f1_scores, supports = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    confusion = confusion_matrix(y_true, y_pred, labels=labels)
    predicted_counts = confusion.sum(axis=0)
    true_counts = confusion.sum(axis=1)
    total_samples = int(confusion.sum())

    per_class_metrics: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(resolved_label_names):
        tp = int(confusion[class_index, class_index])
        fp = int(predicted_counts[class_index] - tp)
        fn = int(true_counts[class_index] - tp)
        tn = total_samples - tp - fp - fn
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        one_vs_rest_accuracy = float((tp + tn) / total_samples) if total_samples > 0 else 0.0
        per_class_metrics.append(
            {
                "class_index": int(labels[class_index]),
                "class_name": str(class_name),
                "precision": float(precisions[class_index]),
                "recall": float(recalls[class_index]),
                "f1": float(f1_scores[class_index]),
                "support": int(supports[class_index]),
                "predicted_count": int(predicted_counts[class_index]),
                "true_count": int(true_counts[class_index]),
                "specificity": specificity,
                "one_vs_rest_accuracy": one_vs_rest_accuracy,
            }
        )
    return per_class_metrics


def compute_classification_metrics(
    targets: list[int] | np.ndarray,
    predictions: list[int] | np.ndarray,
    probabilities: list[list[float]] | np.ndarray | None = None,
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    y_true = np.asarray(targets, dtype=np.int64)
    y_pred = np.asarray(predictions, dtype=np.int64)
    y_prob = None if probabilities is None else np.asarray(probabilities, dtype=np.float64)
    labels, resolved_label_names = _resolve_classification_label_space(y_true, y_pred, y_prob=y_prob, label_names=label_names)

    if y_true.size == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "macro_f1": 0.0,
            "auc_ovr": None,
            "confusion_matrix": [],
            "labels": resolved_label_names,
            "per_class_metrics": compute_per_class_metrics([], [], label_names=resolved_label_names),
        }

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": resolved_label_names,
        "auc_ovr": None,
        "per_class_metrics": compute_per_class_metrics(y_true, y_pred, label_names=resolved_label_names),
    }

    if y_prob is not None:
        unique_classes = np.unique(y_true)
        full_label_space_present = unique_classes.size == len(labels)
        if unique_classes.size > 1 and full_label_space_present:
            try:
                y_true_bin = label_binarize(y_true, classes=labels)
                metrics["auc_ovr"] = float(
                    roc_auc_score(
                        y_true_bin,
                        y_prob,
                        average="macro",
                        multi_class="ovr",
                    )
                )
            except (ValueError, IndexError):
                metrics["auc_ovr"] = None

    return metrics


def build_image_level_metrics(
    targets: list[int] | np.ndarray,
    predictions: list[int] | np.ndarray,
    probabilities: list[list[float]] | np.ndarray | None = None,
    label_names: list[str] | None = None,
    loss: float | None = None,
) -> dict[str, Any]:
    metrics = compute_classification_metrics(
        targets=targets,
        predictions=predictions,
        probabilities=probabilities,
        label_names=label_names,
    )
    metrics["loss"] = float(loss) if loss is not None else None
    metrics["num_samples"] = int(len(targets))
    metrics["sample_level"] = "image"
    return metrics


def _save_confusion_matrix_with_matplotlib(
    confusion_array: np.ndarray,
    label_names: list[str],
    output_path: Path,
    normalize: bool,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False

    figure_size = max(6, int(np.ceil(len(label_names) * 0.8)))
    fig, ax = plt.subplots(figsize=(figure_size, figure_size))
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_array, display_labels=label_names)
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format=".2f" if normalize else "d")
    ax.set_title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def _save_confusion_matrix_with_pillow(
    confusion_array: np.ndarray,
    label_names: list[str],
    output_path: Path,
) -> None:
    cell_size = 64
    label_area = 180
    grid_size = max(1, len(label_names))
    width = label_area + (cell_size * grid_size)
    height = label_area + (cell_size * grid_size)

    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    max_value = float(confusion_array.max()) if confusion_array.size > 0 else 1.0
    max_value = max(max_value, 1.0)

    for row in range(grid_size):
        for col in range(grid_size):
            x0 = label_area + col * cell_size
            y0 = label_area + row * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size

            value = float(confusion_array[row, col]) if confusion_array.size > 0 else 0.0
            intensity = int(255 - (180 * (value / max_value)))
            fill = (intensity, intensity, 255)
            draw.rectangle([x0, y0, x1, y1], fill=fill, outline="black")
            draw.text((x0 + 18, y0 + 22), str(int(value)), fill="black", font=font)

    for index, label in enumerate(label_names):
        draw.text((label_area + index * cell_size + 8, label_area - 24), str(label), fill="black", font=font)
        draw.text((8, label_area + index * cell_size + 22), str(label), fill="black", font=font)

    draw.text((label_area, 20), "Predicted label", fill="black", font=font)
    draw.text((20, 20), "True label", fill="black", font=font)
    image.save(output_path)


def save_confusion_matrix_figure(
    confusion: list[list[int]] | np.ndarray,
    label_names: list[str],
    output_path: str | Path,
    normalize: bool = False,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    confusion_array = np.asarray(confusion)
    if normalize and confusion_array.size > 0:
        row_sums = confusion_array.sum(axis=1, keepdims=True)
        confusion_array = np.divide(
            confusion_array,
            np.where(row_sums == 0, 1, row_sums),
            dtype=np.float64,
        )

    used_matplotlib = _save_confusion_matrix_with_matplotlib(
        confusion_array=confusion_array,
        label_names=label_names,
        output_path=output_path,
        normalize=normalize,
    )
    if not used_matplotlib:
        _save_confusion_matrix_with_pillow(confusion_array=confusion_array, label_names=label_names, output_path=output_path)
    return output_path


def save_metrics_json(metrics: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path