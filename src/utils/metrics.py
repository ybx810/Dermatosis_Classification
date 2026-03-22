from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

DEFAULT_EVALUATION_CONFIG = {
    "level": "both",
    "aggregation": "mean_prob",
    "report_patch_metrics": True,
    "report_image_metrics": True,
}
VALID_EVALUATION_LEVELS = {"patch", "image", "both"}
VALID_AGGREGATIONS = {"mean_prob", "majority_vote", "max_prob"}


def compute_classification_metrics(
    targets: list[int] | np.ndarray,
    predictions: list[int] | np.ndarray,
    probabilities: list[list[float]] | np.ndarray | None = None,
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute stable multi-class classification metrics."""

    y_true = np.asarray(targets, dtype=np.int64)
    y_pred = np.asarray(predictions, dtype=np.int64)
    y_prob = None if probabilities is None else np.asarray(probabilities, dtype=np.float64)

    if y_true.size == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "macro_f1": 0.0,
            "auc_ovr": None,
            "confusion_matrix": [],
            "labels": label_names or [],
        }

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
        labels = list(range(len(label_names)))
    elif inferred_num_classes is not None:
        labels = list(range(inferred_num_classes))
        label_names = [str(label) for label in labels]
    else:
        labels = sorted({*y_true.tolist(), *y_pred.tolist()})
        label_names = [str(label) for label in labels]

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": label_names,
        "auc_ovr": None,
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


def resolve_evaluation_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    evaluation_config = dict(DEFAULT_EVALUATION_CONFIG)
    if config is not None:
        source_config = config.get("evaluation", config)
        evaluation_config.update(source_config)

    level = str(evaluation_config.get("level", "both")).lower()
    aggregation = str(evaluation_config.get("aggregation", "mean_prob")).lower()
    if level not in VALID_EVALUATION_LEVELS:
        raise ValueError(f"Unsupported evaluation.level: {level}. Expected one of {sorted(VALID_EVALUATION_LEVELS)}")
    if aggregation not in VALID_AGGREGATIONS:
        raise ValueError(f"Unsupported evaluation.aggregation: {aggregation}. Expected one of {sorted(VALID_AGGREGATIONS)}")

    report_patch_metrics = bool(evaluation_config.get("report_patch_metrics", True))
    report_image_metrics = bool(evaluation_config.get("report_image_metrics", True))
    should_report_patch = report_patch_metrics or level in {"patch", "both"}
    should_report_image = report_image_metrics or level in {"image", "both"}
    primary_metric_level = "image" if level == "image" else "patch"

    return {
        "level": level,
        "aggregation": aggregation,
        "report_patch_metrics": report_patch_metrics,
        "report_image_metrics": report_image_metrics,
        "should_report_patch": should_report_patch,
        "should_report_image": should_report_image,
        "primary_metric_level": primary_metric_level,
    }


def aggregate_patch_predictions_to_image(
    targets: list[int] | np.ndarray,
    predictions: list[int] | np.ndarray,
    probabilities: list[list[float]] | np.ndarray | None,
    source_images: list[str],
    aggregation: str = "mean_prob",
    patch_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Aggregate patch predictions into image-level predictions by source_image."""

    y_true = np.asarray(targets, dtype=np.int64)
    y_pred = np.asarray(predictions, dtype=np.int64)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("targets and predictions must have the same length for image-level aggregation.")
    if y_true.shape[0] != len(source_images):
        raise ValueError("source_images length must match the number of patch predictions.")

    y_prob = None if probabilities is None else np.asarray(probabilities, dtype=np.float64)
    if y_prob is not None and y_prob.shape[0] != y_true.shape[0]:
        raise ValueError("probabilities length must match the number of patch predictions.")
    if aggregation in {"mean_prob", "max_prob"} and y_prob is None:
        raise ValueError(f"Aggregation '{aggregation}' requires per-patch probabilities.")
    if patch_paths is not None and len(patch_paths) != y_true.shape[0]:
        raise ValueError("patch_paths length must match the number of patch predictions.")

    grouped: dict[str, dict[str, Any]] = {}
    for index, source_image in enumerate(source_images):
        normalized_source = str(source_image).strip()
        if not normalized_source:
            patch_hint = ""
            if patch_paths is not None:
                patch_hint = f" (patch_path={patch_paths[index]})"
            raise ValueError(f"Missing source_image for patch prediction at index {index}{patch_hint}.")

        record = grouped.setdefault(
            normalized_source,
            {
                "target": int(y_true[index]),
                "predictions": [],
                "probabilities": [],
                "patch_paths": [],
            },
        )
        target_value = int(y_true[index])
        if record["target"] != target_value:
            raise ValueError(
                "Inconsistent labels found for source_image "
                f"'{normalized_source}': {record['target']} vs {target_value}."
            )

        record["predictions"].append(int(y_pred[index]))
        if y_prob is not None:
            record["probabilities"].append(y_prob[index])
        if patch_paths is not None:
            record["patch_paths"].append(str(patch_paths[index]))

    image_targets: list[int] = []
    image_predictions: list[int] = []
    image_probabilities: list[list[float]] = []
    ordered_source_images: list[str] = []
    patch_counts: list[int] = []

    for source_image, record in grouped.items():
        patch_predictions = np.asarray(record["predictions"], dtype=np.int64)
        patch_probabilities = None
        if record["probabilities"]:
            patch_probabilities = np.asarray(record["probabilities"], dtype=np.float64)

        if aggregation == "mean_prob":
            aggregated_probabilities = np.mean(patch_probabilities, axis=0)
            aggregated_prediction = int(np.argmax(aggregated_probabilities))
        elif aggregation == "max_prob":
            aggregated_probabilities = np.max(patch_probabilities, axis=0)
            probability_sum = float(np.sum(aggregated_probabilities))
            if probability_sum > 0:
                aggregated_probabilities = aggregated_probabilities / probability_sum
            aggregated_prediction = int(np.argmax(aggregated_probabilities))
        elif aggregation == "majority_vote":
            vote_counter = Counter(patch_predictions.tolist())
            max_votes = max(vote_counter.values()) if vote_counter else 0
            tied_labels = sorted(label for label, count in vote_counter.items() if count == max_votes)
            if len(tied_labels) == 1:
                aggregated_prediction = int(tied_labels[0])
            elif patch_probabilities is not None:
                mean_probabilities = np.mean(patch_probabilities, axis=0)
                aggregated_prediction = int(max(tied_labels, key=lambda label: (mean_probabilities[label], -label)))
            else:
                aggregated_prediction = int(tied_labels[0])

            if patch_probabilities is not None:
                aggregated_probabilities = np.mean(patch_probabilities, axis=0)
            else:
                num_classes = max(patch_predictions.tolist()) + 1 if patch_predictions.size > 0 else 1
                aggregated_probabilities = np.zeros(num_classes, dtype=np.float64)
                for label, count in vote_counter.items():
                    aggregated_probabilities[int(label)] = count / max(1, patch_predictions.size)
        else:
            raise ValueError(f"Unsupported aggregation strategy: {aggregation}")

        ordered_source_images.append(source_image)
        image_targets.append(int(record["target"]))
        image_predictions.append(aggregated_prediction)
        image_probabilities.append(aggregated_probabilities.astype(np.float64).tolist())
        patch_counts.append(int(patch_predictions.size))

    return {
        "source_images": ordered_source_images,
        "targets": image_targets,
        "predictions": image_predictions,
        "probabilities": image_probabilities,
        "patch_counts": patch_counts,
    }


def compute_multilevel_classification_metrics(
    targets: list[int] | np.ndarray,
    predictions: list[int] | np.ndarray,
    probabilities: list[list[float]] | np.ndarray | None,
    label_names: list[str] | None = None,
    source_images: list[str] | None = None,
    patch_paths: list[str] | None = None,
    loss: float | None = None,
    evaluation_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_config = resolve_evaluation_config(evaluation_config)

    patch_metrics = compute_classification_metrics(
        targets=targets,
        predictions=predictions,
        probabilities=probabilities,
        label_names=label_names,
    )
    patch_metrics["loss"] = float(loss) if loss is not None else None
    patch_metrics["num_samples"] = int(len(targets))
    patch_metrics["sample_level"] = "patch"

    image_metrics: dict[str, Any] | None = None
    if resolved_config["should_report_image"] or resolved_config["primary_metric_level"] == "image":
        if len(targets) == 0:
            image_metrics = compute_classification_metrics([], [], [], label_names=label_names)
            image_metrics["num_samples"] = 0
            image_metrics["sample_level"] = "image"
            image_metrics["aggregation"] = resolved_config["aggregation"]
            image_metrics["patches_per_image"] = {"min": 0, "max": 0, "mean": 0.0}
        else:
            if source_images is None:
                raise ValueError("Image-level evaluation requires source_images for every patch.")

            aggregated = aggregate_patch_predictions_to_image(
                targets=targets,
                predictions=predictions,
                probabilities=probabilities,
                source_images=source_images,
                aggregation=resolved_config["aggregation"],
                patch_paths=patch_paths,
            )
            image_metrics = compute_classification_metrics(
                targets=aggregated["targets"],
                predictions=aggregated["predictions"],
                probabilities=aggregated["probabilities"],
                label_names=label_names,
            )
            image_metrics["num_samples"] = int(len(aggregated["targets"]))
            image_metrics["sample_level"] = "image"
            image_metrics["aggregation"] = resolved_config["aggregation"]
            if aggregated["patch_counts"]:
                patch_counts = np.asarray(aggregated["patch_counts"], dtype=np.float64)
                image_metrics["patches_per_image"] = {
                    "min": int(np.min(patch_counts)),
                    "max": int(np.max(patch_counts)),
                    "mean": float(np.mean(patch_counts)),
                }
            else:
                image_metrics["patches_per_image"] = {"min": 0, "max": 0, "mean": 0.0}

    primary_metrics = patch_metrics
    if resolved_config["primary_metric_level"] == "image":
        if image_metrics is None:
            raise ValueError("evaluation.level is set to image but image_metrics could not be computed.")
        primary_metrics = image_metrics

    results: dict[str, Any] = {
        "loss": float(loss) if loss is not None else patch_metrics.get("loss"),
        "num_samples": int(len(targets)),
        "evaluation_level": resolved_config["level"],
        "aggregation": resolved_config["aggregation"],
        "primary_metric_level": resolved_config["primary_metric_level"],
        "accuracy": float(primary_metrics["accuracy"]),
        "precision": float(primary_metrics["precision"]),
        "recall": float(primary_metrics["recall"]),
        "macro_f1": float(primary_metrics["macro_f1"]),
        "auc_ovr": primary_metrics.get("auc_ovr"),
        "confusion_matrix": primary_metrics["confusion_matrix"],
        "labels": primary_metrics["labels"],
    }

    if resolved_config["should_report_patch"]:
        results["patch_metrics"] = patch_metrics
        results["patch_accuracy"] = float(patch_metrics["accuracy"])
        results["patch_precision"] = float(patch_metrics["precision"])
        results["patch_recall"] = float(patch_metrics["recall"])
        results["patch_macro_f1"] = float(patch_metrics["macro_f1"])
        results["patch_auc_ovr"] = patch_metrics.get("auc_ovr")
        results["patch_confusion_matrix"] = patch_metrics["confusion_matrix"]

    if image_metrics is not None and resolved_config["should_report_image"]:
        results["image_metrics"] = image_metrics
        results["image_accuracy"] = float(image_metrics["accuracy"])
        results["image_precision"] = float(image_metrics["precision"])
        results["image_recall"] = float(image_metrics["recall"])
        results["image_macro_f1"] = float(image_metrics["macro_f1"])
        results["image_auc_ovr"] = image_metrics.get("auc_ovr")
        results["image_confusion_matrix"] = image_metrics["confusion_matrix"]
        results["num_images"] = int(image_metrics["num_samples"])

    return results


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
            text = str(int(value))
            draw.text((x0 + 18, y0 + 22), text, fill="black", font=font)

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
    """Save a confusion matrix image for later inspection and reporting."""

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
