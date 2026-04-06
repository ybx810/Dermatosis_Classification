from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

from src.utils.metrics import compute_classification_metrics, save_confusion_matrix_figure, save_metrics_json


PER_CLASS_COLUMNS = [
    "class_index",
    "class_name",
    "precision",
    "recall",
    "f1",
    "support",
    "predicted_count",
    "true_count",
    "specificity",
    "one_vs_rest_accuracy",
]


def get_prediction_scores(model: Any, features: np.ndarray) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        try:
            scores = model.predict_proba(features)
            return np.asarray(scores, dtype=np.float64)
        except Exception:
            pass

    if hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(features)
            score_array = np.asarray(scores, dtype=np.float64)
            if score_array.ndim == 1:
                score_array = np.column_stack([-score_array, score_array])
            return score_array
        except Exception:
            return None

    return None


def _resolve_label_indices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str] | None,
) -> list[int]:
    if label_names is not None:
        return list(range(len(label_names)))
    return sorted({*y_true.tolist(), *y_pred.tolist()})


def compute_auc_ovr(
    y_true: np.ndarray,
    score_matrix: np.ndarray | None,
    label_indices: list[int],
) -> float | None:
    if score_matrix is None:
        return None

    if y_true.size == 0 or np.unique(y_true).size < 2:
        return None

    scores = np.asarray(score_matrix, dtype=np.float64)
    classes = np.asarray(label_indices, dtype=np.int64)

    if scores.ndim == 1:
        if classes.size != 2:
            return None
        y_true_binary = (y_true == classes[1]).astype(np.int64)
        try:
            return float(roc_auc_score(y_true_binary, scores))
        except ValueError:
            return None

    if scores.ndim != 2 or scores.shape[1] != classes.size:
        return None

    try:
        y_true_bin = label_binarize(y_true, classes=classes)
        if y_true_bin.shape[1] == 1:
            positive_scores = scores[:, 1] if scores.shape[1] > 1 else scores[:, 0]
            return float(roc_auc_score(y_true_bin.ravel(), positive_scores))
        return float(roc_auc_score(y_true_bin, scores, average="macro", multi_class="ovr"))
    except (ValueError, IndexError):
        return None


def compute_metrics_for_split(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str] | None,
    score_matrix: np.ndarray | None = None,
) -> dict[str, Any]:
    y_true_array = np.asarray(y_true, dtype=np.int64)
    y_pred_array = np.asarray(y_pred, dtype=np.int64)

    metrics = compute_classification_metrics(
        targets=y_true_array,
        predictions=y_pred_array,
        probabilities=None,
        label_names=label_names,
    )
    label_indices = _resolve_label_indices(y_true_array, y_pred_array, label_names)
    metrics["auc_ovr"] = compute_auc_ovr(y_true_array, score_matrix, label_indices)
    return metrics


def save_per_class_metrics_csv(metrics: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame(metrics.get("per_class_metrics", []))
    if dataframe.empty:
        dataframe = pd.DataFrame(columns=PER_CLASS_COLUMNS)
    else:
        dataframe = dataframe.reindex(columns=PER_CLASS_COLUMNS)
    dataframe.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def save_evaluation_artifacts(metrics: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = save_metrics_json(metrics, output_dir / "metrics.json")
    per_class_path = save_per_class_metrics_csv(metrics, output_dir / "per_class_metrics.csv")
    confusion_path = save_confusion_matrix_figure(
        confusion=metrics.get("confusion_matrix", []),
        label_names=metrics.get("labels", []),
        output_path=output_dir / "confusion_matrix.png",
    )
    return {
        "metrics": metrics_path,
        "per_class_metrics": per_class_path,
        "confusion_matrix": confusion_path,
    }


def _safe_column_suffix(value: str) -> str:
    return (
        str(value)
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("-", "_")
    )


def build_predictions_dataframe(
    source_images: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str] | None,
    score_matrix: np.ndarray | None = None,
) -> pd.DataFrame:
    y_true_array = np.asarray(y_true, dtype=np.int64)
    y_pred_array = np.asarray(y_pred, dtype=np.int64)
    if y_true_array.shape[0] != len(source_images):
        raise ValueError("source_images and y_true lengths do not match.")

    dataframe = pd.DataFrame(
        {
            "source_image": source_images,
            "y_true": y_true_array,
            "y_pred": y_pred_array,
        }
    )

    if label_names is not None:
        dataframe["y_true_label"] = [label_names[int(index)] for index in y_true_array.tolist()]
        dataframe["y_pred_label"] = [label_names[int(index)] for index in y_pred_array.tolist()]

    if score_matrix is not None:
        scores = np.asarray(score_matrix, dtype=np.float64)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])

        if scores.ndim == 2:
            if label_names is not None and len(label_names) == scores.shape[1]:
                class_labels = [str(label) for label in label_names]
            else:
                class_labels = [str(index) for index in range(scores.shape[1])]
            for column_index, class_label in enumerate(class_labels):
                column_name = f"score_{_safe_column_suffix(class_label)}"
                dataframe[column_name] = scores[:, column_index]

    return dataframe
